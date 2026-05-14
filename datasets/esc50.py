# Author: Jiayu Xiong
# Reference: STAR-MAE / Distribution-aware Loss Reweighting (DLR), PR-D-25-06353_R2.
import os
import csv
import json
try:
    import torchaudio
except ImportError:
    torchaudio = None
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random
from utils.kaldi_fbank import KaldiFbankTransform


def cfg(args, name, default=None):
    if isinstance(args, dict):
        return args.get(name, default)
    return getattr(args, name, default)


def cfg_any(args, names, default=None):
    for name in names:
        value = cfg(args, name, None)
        if value is not None:
            return value
    return default


class ESC50Dataset(Dataset):
    index_dict = None
    label_num = None

    def __init__(self, test_fold, args, is_test=False):
        """
        Parameters:
            test_fold (int): Specifies which fold to use as the test set (0~4).
            args: Parsed runtime arguments or a backward-compatible config dict.
            is_test (bool): Indicates whether the dataset is for testing.
        """
        self.root = cfg(args, 'root')
        self.test_fold = test_fold
        self.is_test = is_test
        self.args = args

        # Define paths
        self.data_folder = self.root 
        self.label_csv = os.path.join(self.root, 'labelvocabulary.csv')
        self.main_audio_folder = os.path.join(self.root, '48000')  # Main audio folder

        self.fbank_transform = KaldiFbankTransform(
            num_mel_bins=cfg(args, 'H', 128),  # Frequency dimension of fbank
            target_length=cfg(args, 'W', 512) * cfg(args, 'T', 1),
            mixup_rate=cfg(args, 'mixup', 0.0),
            norm_mean=cfg(args, 'mean', 0.0),
            norm_std=cfg(args, 'std', 1.0),
            skip_norm=cfg(args, 'skip_norm', False),
            noise=cfg(args, 'noise', False),
            mel_fmin=cfg(args, 'mel_fmin', 50.0),
            mel_fmax=cfg(args, 'mel_fmax', 8000.0),
        )

        # Create label mapping (only during the first initialization)
        if ESC50Dataset.index_dict is None:
            self._create_label_mapping()

        self.index_dict = ESC50Dataset.index_dict
        self.label_num = ESC50Dataset.label_num

        # Load JSON data and split into training or testing set
        self.data = self._load_json_data()

        dataset_type = 'Test Set' if self.is_test else 'Training Set'
        print(f"Dataset type: {dataset_type}")
        print(f"Number of samples: {len(self.data)}")
        print(f"Number of labels: {self.label_num}")

    def _create_label_mapping(self):
        """
        Create a mapping from labels to indices by reading labels from labelvocabulary.csv.
        """
        index_lookup = {}
        with open(self.label_csv, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header row
            for line in reader:
                if len(line) >= 2:
                    try:
                        idx = int(line[0])
                        label = line[1].strip()
                        index_lookup[label] = idx
                    except ValueError:
                        print(f"Warning: Invalid row format {line}")
        ESC50Dataset.index_dict = index_lookup
        ESC50Dataset.label_num = len(index_lookup)

    def _load_json_data(self):
        """
        Load fold00~fold04.json files and split into training or testing sets based on test_fold.
        """
        data = []
        for fold in range(5):
            fold_name = f"fold0{fold}"
            json_path = os.path.join(self.data_folder, f"{fold_name}.json")
            if not os.path.isfile(json_path):
                print(f"Warning: JSON file does not exist: {json_path}")
                continue
            with open(json_path, 'r', encoding='utf-8') as f:
                try:
                    fold_data = json.load(f)
                except json.JSONDecodeError as e:
                    print(f"Error: Cannot parse JSON file {json_path}: {e}")
                    continue
                for audio_file, labels in fold_data.items():
                    audio_path = os.path.join(self.main_audio_folder, fold_name, audio_file)
                    if not os.path.isfile(audio_path):
                        print(f"Warning: Audio file does not exist: {audio_path}")
                        continue
                    # Select data based on whether it is a test set
                    if self.is_test and fold == self.test_fold:
                        data.append({
                            'audio_path': audio_path,
                            'labels': labels
                        })
                    elif not self.is_test and fold != self.test_fold:
                        data.append({
                            'audio_path': audio_path,
                            'labels': labels
                        })
        return data

    def __getitem__(self, index):
        """
        Retrieve a data sample.

        Args:
            index (int): Index of the sample.

        Returns:
            tuple: (frames, label_indices)
        """
        # Get the data sample
        datum = self.data[index]

        audio_path = datum['audio_path']

        # Load audio
        if torchaudio is None:
            raise ImportError('torchaudio is required for wav datasets, but it is not installed.')
        waveform, sr = torchaudio.load(audio_path)

        # Convert labels to one-hot encoding
        label_indices = np.zeros(self.label_num, dtype=np.float32)
        for label_str in datum['labels']:
            if label_str in self.index_dict:
                idx = self.index_dict[label_str]
                label_indices[idx] = 1.0
        label_indices = torch.FloatTensor(label_indices)

        # Decide whether to perform mixup
        do_mixup = (random.random() < self.fbank_transform.mixup_rate) and not self.is_test
        if do_mixup:
            # Randomly select another sample for mixup
            mix_idx = random.randint(0, len(self.data) - 1)
            mix_datum = self.data[mix_idx]
            mix_audio_path = mix_datum['audio_path']
            mix_waveform, mix_sr = torchaudio.load(mix_audio_path)

            # Resample if necessary
            if mix_sr != sr:
                resampler = torchaudio.transforms.Resample(mix_sr, sr)
                mix_waveform = resampler(mix_waveform)

            # Compute fbank
            fbank, mix_lambda = self.fbank_transform(waveform, sr, waveform_2=mix_waveform)

            # Combine labels
            mix_label = np.zeros(self.label_num, dtype=np.float32)
            for label_str in mix_datum['labels']:
                if label_str in self.index_dict:
                    idx = self.index_dict[label_str]
                    mix_label[idx] = 1.0
            mix_label = torch.FloatTensor(mix_label)

            label_indices = mix_lambda * label_indices + (1.0 - mix_lambda) * mix_label
        else:
            # Do not perform mixup
            fbank, _ = self.fbank_transform(waveform, sr, waveform_2=None)

        # Convert [time, freq] to (T, freq, W) or (T, H, W)
        frames_3d = self.fbank_to_spatial(fbank)  # (T, freq, W)
        return frames_3d, label_indices

    def fbank_to_spatial(self, fbank: torch.Tensor):
        """
        Split [time, freq] into T frames, each with (freq, W), and stack them into (T, freq, W).

        Args:
            fbank (torch.Tensor): Fbank tensor of shape [time, freq].

        Returns:
            torch.Tensor: 3D tensor of shape (T, freq, W).
        """
        freq = fbank.shape[1]
        total_time = fbank.shape[0]

        T = cfg(self.args, 'T', 1)
        W = cfg(self.args, 'W', 256)

        required = T * W
        # Pad or truncate the time dimension
        if total_time < required:
            pad_len = required - total_time
            fbank = torch.nn.functional.pad(fbank, (0, 0, 0, pad_len))
        elif total_time > required:
            fbank = fbank[:required, :]

        frames = []
        for i in range(T):
            start = i * W
            end = (i + 1) * W
            # Slice: [W, freq]
            frame = fbank[start:end, :]
            # Transpose to [freq, W]
            frame = frame.transpose(0, 1)
            frames.append(frame)

        frames_3d = torch.stack(frames, dim=0)  # shape = (T, freq, W)
        return frames_3d

    def __len__(self):
        return len(self.data)


def get_ESC50_dataloader(args, test_fold):
    """
    Create DataLoaders for the ESC-50 dataset.

    Args:
        args: Parsed runtime arguments or a backward-compatible config dict.
        test_fold (int): Specifies which fold to use as the test set (0~4).

    Returns:
        tuple: (train_loader, test_loader)
    """
    # Create training dataset
    train_dataset = ESC50Dataset(
        test_fold=test_fold,
        args=args,
        is_test=False
    )

    # Create testing dataset
    test_dataset = ESC50Dataset(
        test_fold=test_fold,
        args=args,
        is_test=True
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg_any(args, ('batchsize', 'batch_size'), 32),
        num_workers=cfg(args, 'workers', 4),
        shuffle=True,
        drop_last=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg_any(args, ('batchsize', 'batch_size'), 32),
        num_workers=cfg(args, 'workers', 4),
        shuffle=False,
        drop_last=False
    )

    return train_loader, test_loader
