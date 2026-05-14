# Author: Jiayu Xiong
# Reference: STAR-MAE / Distribution-aware Loss Reweighting (DLR), PR-D-25-06353_R2.
import os
import random
import numpy as np
import torch
try:
    import torchaudio
    from torchaudio.datasets import SPEECHCOMMANDS
except ImportError:
    torchaudio = None

    class SPEECHCOMMANDS:
        def __init__(self, *args, **kwargs):
            raise ImportError('torchaudio is required for SpeechCommands datasets, but it is not installed.')

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


class SubsetSC(SPEECHCOMMANDS):
    """
    This class is similar to the official torchaudio documentation.
    It splits the SPEECHCOMMANDS dataset into training, validation, and testing subsets.
    """
    def __init__(self, subset: str = None, root: str = "./", download: bool = False):
        super().__init__(root, download=download)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as f:
                return [
                    os.path.normpath(os.path.join(self._path, line.strip()))
                    for line in f
                ]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]

        # If subset=None, retain all data without splitting


class SpeechCommandsDataset(Dataset):
    """
    Extends torchaudio's SubsetSC with additional custom logic:
     - transform (KaldiFbankTransform)
     - mixup
     - add_noise
     - one_hot encoding
     - normalization
    """
    def __init__(self, subset: str, args):
        """
        Parameters:
          subset (str): 'train' / 'valid' / 'test'
          args: Parsed runtime arguments or a backward-compatible config dict
        """
        super().__init__()
        self.subset = subset
        self.args = args

        # Map external 'train' / 'valid' / 'test' to SubsetSC keywords
        subset_map = {'train': 'training', 'valid': 'validation', 'test': 'testing'}
        sc_subset = subset_map[self.subset]

        # Instantiate the torchaudio dataset
        # You can specify the root directory and whether to download via args
        self.base_dataset = SubsetSC(
            subset=sc_subset,
            root=cfg(args, "root", "./"),  
            download=cfg(args, "download", False)
        )

        self.fbank_transform = KaldiFbankTransform(
            num_mel_bins=cfg(args, 'H', 128),  # Frequency dimension of fbank
            target_length=cfg(args, 'W', 1024) * cfg(args, 'T', 1),
            mixup_rate=cfg(args, 'mixup', 0.0),
            norm_mean=cfg(args, 'mean', 0.0),
            norm_std=cfg(args, 'std', 1.0),
            skip_norm=cfg(args, 'skip_norm', False),
            noise=cfg(args, 'noise', False),
            mel_fmin=cfg(args, 'mel_fmin', 50.0),
            mel_fmax=cfg(args, 'mel_fmax', 8000.0),
        )

        # Retrieve all labels from the dataset (torchaudio's dataset is single-label)
        all_labels = [self.base_dataset[i][2] for i in range(len(self.base_dataset))]
        all_labels = sorted(list(set(all_labels)))
        self.label2idx = {label: idx for idx, label in enumerate(all_labels)}
        self.num_classes = len(all_labels)

    def __len__(self):
        return len(self.base_dataset)

    def to_one_hot(self, label_indices):
        """
        Convert label indices (or a list of indices) to a one-hot encoded vector.
        In the SpeechCommands official dataset, it's typically single-label,
        but can be modified for multi-label if needed.
        """
        one_hot = torch.zeros(self.num_classes, dtype=torch.float)
        one_hot[label_indices] = 1.0
        return one_hot

    def add_noise(self, waveform):
        """
        Add Gaussian noise, similar to the original logic.
        """
        noise_std = 0.005  # Noise standard deviation, adjustable as needed
        noise = torch.randn_like(waveform) * noise_std
        return waveform + noise

    def __getitem__(self, index):
        """
        Retrieve a data sample.

        Args:
            index (int): Index of the sample.

        Returns:
            tuple: (frames, label_indices)
        """
        # Get the data sample from the base dataset
        waveform, sr, label, _, _ = self.base_dataset[index]
        # Map label to index
        label_num = self.label2idx[label]
        label_indices = self.to_one_hot(label_num)
        # Add noise if required
        if cfg(self.args, 'noise', False) and self.subset == 'train':
            waveform = self.add_noise(waveform)

        # Apply transformation

        # Decide whether to perform mixup
        do_mixup = (random.random() < self.fbank_transform.mixup_rate) and (self.subset == 'train')
        if do_mixup:
            # Randomly select another sample for mixup
            mix_idx = random.randint(0, len(self.base_dataset) - 1)
            mix_waveform, sr, mix_label_, _, _ = self.base_dataset[mix_idx]

            # Compute fbank
            fbank, mix_lambda = self.fbank_transform(waveform, sr, waveform_2=mix_waveform)
            mix_label_ = self.label2idx[mix_label_]
            mix_label = self.to_one_hot(mix_label_)

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


def create_speechcommands_dataloaders(args):
    """
    Similar to your original create_speechcommands_dataloaders function.
    Returns three DataLoaders: train_loader, valid_loader, test_loader.

    Args:
        args: Parsed runtime arguments or a backward-compatible config dict.

    Returns:
        tuple: (train_loader, valid_loader, test_loader)
    """
    batch_size = cfg_any(args, ('batchsize', 'batch_size'), 32)
    num_workers = cfg(args, 'workers', 4)

    train_dataset = SpeechCommandsDataset(subset='train', args=args)
    valid_dataset = SpeechCommandsDataset(subset='valid', args=args)
    test_dataset = SpeechCommandsDataset(subset='test', args=args)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, valid_loader, test_loader
