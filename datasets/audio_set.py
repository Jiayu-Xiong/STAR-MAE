# Author: Jiayu Xiong
# Reference: STAR-MAE / Distribution-aware Loss Reweighting (DLR), PR-D-25-06353_R2.
import os
import csv
try:
    import torchaudio
except ImportError:
    torchaudio = None
import numpy as np
import torch
from torch.utils.data import Dataset
import random
from utils.audio_to_video import AudioToVideoTransform


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
    index_dict = None
    label_num = None

    def __init__(self, dataset_type, csv_path, data_folder, args, all_csv_paths):
        """
        Parameters:
            dataset_type: Type of dataset, one of 'unbal', 'bal', 'eval'
            csv_path: Path to the current dataset's CSV file
            data_folder: Root directory where audio files are stored
            args: Parsed runtime arguments or a backward-compatible config dict
            all_csv_paths: List of all dataset CSV file paths used to create a unified label mapping
        """
        self.dataset_type = dataset_type
        self.csv_path = csv_path
        self.data_folder = data_folder
        self.args = args
        self.all_csv_paths = all_csv_paths

        # Load and parse the current dataset's CSV file
        self.data = self._load_csv_data(self.csv_path)

        # Create label mapping (only during the first initialization)
        if AudiosetDataset.index_dict is None:
            self._create_label_mapping()

        # Use class variables
        self.index_dict = AudiosetDataset.index_dict
        self.label_num = AudiosetDataset.label_num

        # Initialize the audio transformer based on dataset type
        if self.dataset_type != 'eval':
            self.transform = AudioToVideoTransform(
                sample_rate=cfg(self.args, 'sample_rate', 16000),
                n_fft=cfg(self.args, 'n_fft', 2048),
                H=cfg(self.args, 'H', 224),
                W=cfg(self.args, 'W', 224),
                T=cfg(self.args, 'T', 4),
                overlap_rate=cfg(self.args, 'overlap_rate', 0.2),
                per_frame_overlap=cfg(self.args, 'per_frame_overlap', False),
                use_mel=cfg(self.args, 'use_mel', True),
                hop_length=cfg(self.args, 'hop_length', None),
                mel_fmin=cfg(self.args, 'mel_fmin', 50.0),
                mel_fmax=cfg(self.args, 'mel_fmax', 8000.0),
            )
        else:
            self.transform = AudioToVideoTransform(
                sample_rate=cfg(self.args, 'sample_rate', 16000),
                n_fft=cfg(self.args, 'n_fft', 2048),
                H=cfg(self.args, 'H', 224),
                W=cfg(self.args, 'W', 224),
                T=cfg(self.args, 'T', 4),
                overlap_rate=0.0,
                per_frame_overlap=cfg(self.args, 'per_frame_overlap', False),
                use_mel=cfg(self.args, 'use_mel', True),
                hop_length=100,
                mel_fmin=cfg(self.args, 'mel_fmin', 50.0),
                mel_fmax=cfg(self.args, 'mel_fmax', 8000.0),
            )

        # Retrieve additional parameters
        self.mixup = cfg(self.args, 'mixup', 0.0)
        self.mixup_alpha = cfg(self.args, 'mixup_alpha', 0.4)
        self.norm_mean = cfg(self.args, 'mean', 0.0)
        self.norm_std = cfg(self.args, 'std', 1.0)
        self.skip_norm = cfg(self.args, 'skip_norm', False)
        self.noise = cfg(self.args, 'noise', False)

        print(f"Dataset type: {self.dataset_type}")
        print(f"Number of samples: {len(self.data)}")
        print(f"Number of labels: {self.label_num}")

    def _create_label_mapping(self):
        """
        Create a mapping from labels to indices by extracting labels from all dataset CSV files
        and ensuring a consistent order.
        """
        all_labels = set()
        for csv_path in self.all_csv_paths:
            with open(csv_path, 'r') as f:
                reader = csv.reader(f, skipinitialspace=True)
                for line in reader:
                    if len(line) < 4:
                        continue
                    labels_str = ','.join(line[3:]).strip()
                    labels_str = labels_str.strip('"')
                    labels = labels_str.split(',')
                    labels = [label.strip() for label in labels]
                    all_labels.update(labels)
        labels = sorted(all_labels)
        index_lookup = {label: idx for idx, label in enumerate(labels)}
        AudiosetDataset.index_dict = index_lookup
        AudiosetDataset.label_num = len(index_lookup)

    def _load_csv_data(self, csv_path):
        """
        Load data from a CSV file.

        Args:
            csv_path (str): Path to the CSV file.

        Returns:
            list: A list of dictionaries with 'video_id' and 'labels'.
        """
        data = []
        with open(csv_path, 'r') as f:
            reader = csv.reader(f, skipinitialspace=True)
            for line in reader:
                if len(line) < 4:
                    continue
                video_id = line[0].strip()
                labels_str = ','.join(line[3:]).strip()
                labels_str = labels_str.strip('"')
                labels = labels_str.split(',')
                labels = [label.strip() for label in labels]
                data.append({
                    'video_id': video_id,
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

        # Construct the audio file path
        audio_filename = 'Y' + datum['video_id'] + '.wav'
        audio_path = os.path.join(self.data_folder, audio_filename)

        # Load the audio
        if torchaudio is None:
            raise ImportError('torchaudio is required for wav datasets, but it is not installed.')
        waveform, sr = torchaudio.load(audio_path)

        # Check if the sampling rate matches the target
        target_sr = cfg(self.args, 'sample_rate', 16000)
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            waveform = resampler(waveform)
            sr = target_sr

        # Process labels into one-hot encoding
        label_indices = np.zeros(self.label_num)
        for label_str in datum['labels']:
            if label_str in self.index_dict:
                idx = self.index_dict[label_str]
                label_indices[idx] = 1.0

        label_indices = torch.FloatTensor(label_indices)

        # Decide whether to perform mixup
        if random.random() < self.mixup and self.dataset_type != 'eval':
            # Randomly select another sample for mixup
            mix_idx = random.randint(0, len(self.data) - 1)
            mix_datum = self.data[mix_idx]
            mix_audio_filename = 'Y' + mix_datum['video_id'] + '.wav'
            mix_audio_path = os.path.join(self.data_folder, mix_audio_filename)
            mix_waveform, mix_sr = torchaudio.load(mix_audio_path)

            # Resample if necessary
            if mix_sr != sr:
                resampler = torchaudio.transforms.Resample(mix_sr, sr)
                mix_waveform = resampler(mix_waveform)

            # Align lengths
            min_length = min(waveform.shape[1], mix_waveform.shape[1])
            waveform = waveform[:, :min_length]
            mix_waveform = mix_waveform[:, :min_length]

            # Generate mixup coefficient
            mix_lambda = np.random.beta(self.mixup_alpha, self.mixup_alpha)

            # Perform mixup
            waveform = mix_lambda * waveform + (1 - mix_lambda) * mix_waveform

            # Combine labels
            mix_label_indices = np.zeros(self.label_num)
            for label_str in mix_datum['labels']:
                if label_str in self.index_dict:
                    idx = self.index_dict[label_str]
                    mix_label_indices[idx] = 1.0

            label_indices = mix_lambda * label_indices + (1 - mix_lambda) * torch.FloatTensor(mix_label_indices)

        # Apply audio transformation
        frames = self.transform(waveform)

        # Normalize
        if not self.skip_norm:
            frames = (frames - self.norm_mean) / (self.norm_std * 2)

        # Add noise for augmentation
        if self.noise:
            frames = frames + torch.rand_like(frames) * np.random.rand() / 10
            frames = torch.roll(frames, shifts=np.random.randint(-10, 10), dims=0)

        return frames, label_indices

    def __len__(self):
        return len(self.data)


def test_dataset():
    # Assume you have the following CSV file paths
    csv_path_unbal = 'AudioSet2M/un_train_index_cleaned.csv'
    csv_path_bal = 'AudioSet2M/train_index_cleaned.csv'
    csv_path_eval = 'AudioSet2M/eval_index_cleaned.csv'

    # List of all dataset CSV file paths
    all_csv_paths = [csv_path_unbal, csv_path_bal, csv_path_eval]
    # all_csv_paths = [csv_path_bal, csv_path_eval]

    args = type('Args', (), {
        'sample_rate': 16000,
        'mel_fmin': 50.0,
        'mel_fmax': 8000.0,
        'n_fft': 1024,
        'H': 224,
        'W': 224,
        'T': 4,
        'overlap_rate': 0.2,
        'per_frame_overlap': True,
        'use_mel': True,
        'mixup': 0.5,
        'mixup_alpha': 0.4,
        'mean': 0.0,
        'std': 1.0,
        'skip_norm': False,
        'noise': False,
    })()

    # Create dataset instances
    unbal_dataset = AudiosetDataset(
        dataset_type='unbal',
        csv_path=csv_path_unbal,
        data_folder='AudioSet2M/unbal',
        args=args,
        all_csv_paths=all_csv_paths
    )

    bal_dataset = AudiosetDataset(
        dataset_type='bal',
        csv_path=csv_path_bal,
        data_folder='AudioSet2M/bal',
        args=args,
        all_csv_paths=all_csv_paths
    )

    eval_dataset = AudiosetDataset(
        dataset_type='eval',
        csv_path=csv_path_eval,
        data_folder='AudioSet2M/eval',
        args=args,
        all_csv_paths=all_csv_paths
    )


from torch.utils.data import DataLoader


def get_dataset_2M(args):
    """
    Create DataLoaders for the 2M dataset.

    Args:
        args: Parsed runtime arguments or a backward-compatible config dict.

    Returns:
        tuple: (unbal_loader, eval_loader)
    """
    root = cfg_any(args, ('root', 'dataset_root'))
    batch_size = cfg_any(args, ('batchsize', 'batch_size'))
    num_workers = cfg(args, 'workers')

    # Construct CSV file paths using the root directory
    csv_path_unbal = os.path.join(root, 'un_train_index_cleaned.csv')
    csv_path_bal = os.path.join(root, 'train_index_cleaned.csv')
    csv_path_eval = os.path.join(root, 'eval_index_cleaned.csv')

    # All CSV file paths
    all_csv_paths = [csv_path_bal, csv_path_eval]

    # Create dataset instances
    unbal_dataset = AudiosetDataset(
        dataset_type='unbal',
        csv_path=csv_path_unbal,
        data_folder=os.path.join(root, 'unbal'),
        args=args,
        all_csv_paths=all_csv_paths
    )

    eval_dataset = AudiosetDataset(
        dataset_type='eval',
        csv_path=csv_path_eval,
        data_folder=os.path.join(root, 'eval'),
        args=args,
        all_csv_paths=all_csv_paths
    )

    # Create DataLoaders
    unbal_loader = DataLoader(
        unbal_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,  # Typically shuffle training set
        drop_last=True
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,  # Typically do not shuffle evaluation set
        drop_last=True
    )

    return unbal_loader, eval_loader


def get_eval_set(args):
    """
    Create a DataLoader for the evaluation set. For our distribution visualization.

    Args:
        args: Parsed runtime arguments or a backward-compatible config dict.

    Returns:
        tuple: (None, eval_loader)
    """
    root = cfg_any(args, ('root', 'dataset_root'))
    batch_size = cfg_any(args, ('batchsize', 'batch_size'))
    num_workers = cfg(args, 'workers')

    # Construct CSV file paths using the root directory
    csv_path_unbal = os.path.join(root, 'un_train_index_cleaned.csv')
    csv_path_bal = os.path.join(root, 'train_index_cleaned.csv')
    csv_path_eval = os.path.join(root, 'eval_index_cleaned.csv')

    # All CSV file paths
    all_csv_paths = [csv_path_bal, csv_path_eval]

    # Create evaluation dataset
    eval_dataset = AudiosetDataset(
        dataset_type='eval',
        csv_path=csv_path_eval,
        data_folder=os.path.join(root, 'eval'),
        args=args,
        all_csv_paths=all_csv_paths
    )

    # Create DataLoader for evaluation
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,  # Typically do not shuffle evaluation set
        drop_last=False
    )

    return None, eval_loader
