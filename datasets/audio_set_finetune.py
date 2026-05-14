# Author: Jiayu Xiong
# Reference: STAR-MAE / Distribution-aware Loss Reweighting (DLR), PR-D-25-06353_R2.
import os
import csv
import random

import numpy as np
import torch
try:
    import torchaudio
except ImportError:
    torchaudio = None
from torch.utils.data import Dataset, DataLoader

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


class AudiosetDataset(Dataset):
    """
    Complete Example:
      1) During initialization, read the CSV file and store row information in self.data
      2) If index_dict is not created, collect all labels based on all_csv_paths, sort them, and assign an index to each label
      3) In __getitem__: perform mixup based on args, apply KaldiFbankTransform, and output (T, freq, W)
    """

    # Can also be set as class variables; here we demonstrate storing them within __init__
    # index_dict = None
    # label_num = None

    def __init__(self, dataset_type, csv_path, data_folder, args, all_csv_paths):
        """
        Parameters:
            dataset_type: Type of dataset ('bal', 'eval', etc.), controls whether to perform mixup, etc.
            csv_path: Path to the current dataset's CSV file
            data_folder: Root directory containing .wav audio files
            args: Parsed runtime arguments or a backward-compatible config dict.
            all_csv_paths: List of CSV files used to collect all labels
        """
        self.dataset_type = dataset_type
        self.csv_path = csv_path
        self.data_folder = data_folder
        self.args = args
        self.all_csv_paths = all_csv_paths

        # ---------- (A) Load current dataset CSV ----------
        self.data = self._load_csv_data(self.csv_path)
        
        # ---------- (B) Build or retrieve label-to-index mapping based on all CSV files ----------
        self.index_dict = self._build_or_get_label_mapping(self.all_csv_paths)
        self.label_num = len(self.index_dict)

        # ---------- (C) Initialize the Kaldi fbank transform replicated from the old version ----------
        # Set freqm / timem based on your required parameters. To align with the old version, you can also use fixed values
        self.fbank_transform = KaldiFbankTransform(
            num_mel_bins=cfg(args, 'H', 128),  # Frequency dimension of fbank
            target_length=cfg(args, 'W', 1024) * cfg(args, 'T', 1),
            freqm=cfg(args, 'freqm', 48),
            timem=cfg(args, 'timem', 48),
            mixup_rate=cfg(args, 'mixup', 0.0),
            norm_mean=cfg(args, 'mean', 0.0),
            norm_std=cfg(args, 'std', 1.0),
            skip_norm=cfg(args, 'skip_norm', False),
            noise=cfg(args, 'noise', False),
            mel_fmin=cfg(args, 'mel_fmin', 50.0),
            mel_fmax=cfg(args, 'mel_fmax', 8000.0),
        )

        print(f"Dataset type: {self.dataset_type}, Samples: {len(self.data)}, Num classes: {self.label_num}")

    def _load_csv_data(self, csv_file):
        """
        Read data from a CSV file and store it in self.data
        Assumes the CSV format is:
            video_id, start_time, end_time, label1, label2, ...
        where columns after the third are labels. Modify according to your actual format.
        """
        data_list = []
        with open(csv_file, 'r') as f:
            reader = csv.reader(f, skipinitialspace=True)
            for line in reader:
                if len(line) < 4:
                    continue
                video_id = line[0].strip()
                # Take line[3:] as label columns
                labels_str = ','.join(line[3:]).strip().strip('"')
                labels = [lab.strip() for lab in labels_str.split(',')]
                data_list.append({
                    'video_id': video_id,
                    'labels': labels
                })
        return data_list

    def _build_or_get_label_mapping(self, csv_paths):
        """
        Collect all possible labels from all_csv_paths, sort them, and assign an index to each label.
        To reuse the same mapping, you can modify this to use class variables.
        """
        all_labels = set()
        for csv_path in csv_paths:
            if not os.path.isfile(csv_path):
                continue
            with open(csv_path, 'r') as f:
                reader = csv.reader(f, skipinitialspace=True)
                for line in reader:
                    if len(line) < 4:
                        continue
                    labels_str = ','.join(line[3:]).strip().strip('"')
                    labels = [lab.strip() for lab in labels_str.split(',')]
                    all_labels.update(labels)
        
        sorted_labels = sorted(list(all_labels))
        index_dict = {label: idx for idx, label in enumerate(sorted_labels)}
        return index_dict

    def __getitem__(self, index):
        """
        Core Logic:
          1) Read audio waveform
          2) Process labels
          3) Randomly decide whether to perform mixup
          4) Apply KaldiFbankTransform to obtain fbank
          5) If mixup is performed, combine labels
          6) Convert fbank to (T, freq, W) or other required output
        """
        datum = self.data[index]
        audio_filename = 'Y' + datum['video_id'] + '.wav'
        audio_path = os.path.join(self.data_folder, audio_filename)

        # Load audio
        if torchaudio is None:
            raise ImportError('torchaudio is required for wav datasets, but it is not installed.')
        waveform, sr = torchaudio.load(audio_path)
        # waveform shape: [channels, num_samples], typically [1, N]

        # Convert labels to one-hot encoding
        label_indices = np.zeros(self.label_num, dtype=np.float32)
        for label_str in datum['labels']:
            if label_str in self.index_dict:
                idx = self.index_dict[label_str]
                label_indices[idx] = 1.0
        label_indices = torch.FloatTensor(label_indices)

        # If dataset_type != 'eval' and a random number falls below mixup_rate, perform mixup
        do_mixup = (random.random() < self.fbank_transform.mixup_rate) and (self.dataset_type != 'eval')
        if do_mixup:
            # Randomly select another sample
            mix_idx = random.randint(0, len(self.data) - 1)
            mix_datum = self.data[mix_idx]
            mix_audio_filename = 'Y' + mix_datum['video_id'] + '.wav'
            mix_audio_path = os.path.join(self.data_folder, mix_audio_filename)
            mix_waveform, mix_sr = torchaudio.load(mix_audio_path)

            # If sampling rates differ, resample to sr
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


def get_dataset_20K(args):
    """
    Example of constructing training and evaluation DataLoaders.
    args should include:
      {
        'root': 'path/to/dataset_root',
        'batchsize': 32,
        'workers': 4,
        'T': 4,
        'W': 224,
        'H': 128,
        ...
      }
    """
    root = cfg_any(args, ('root', 'dataset_root'))
    batch_size = cfg_any(args, ('batchsize', 'batch_size'), 32)
    num_workers = cfg(args, 'workers', 4)

    # CSV paths
    csv_path_bal = os.path.join(root, 'train_index_cleaned.csv')
    csv_path_eval = os.path.join(root, 'eval_index_cleaned.csv')

    # Collect all CSVs to build a unified label mapping
    all_csv_paths = [csv_path_bal, csv_path_eval]

    # Create Datasets
    bal_dataset = AudiosetDataset(
        dataset_type='bal',
        csv_path=csv_path_bal,
        data_folder=os.path.join(root, 'bal'),   # Assuming balanced set wav files are in .../bal directory
        args=args,
        all_csv_paths=all_csv_paths
    )

    eval_dataset = AudiosetDataset(
        dataset_type='eval',
        csv_path=csv_path_eval,
        data_folder=os.path.join(root, 'eval'),  # Assuming evaluation set wav files are in .../eval directory
        args=args,
        all_csv_paths=all_csv_paths
    )

    # Create DataLoaders
    bal_loader = DataLoader(
        bal_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,   # Typically shuffle training set
        drop_last=True
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,  # Typically do not shuffle evaluation set
        drop_last=False
    )

    return bal_loader, eval_loader
