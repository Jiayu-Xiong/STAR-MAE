# Author: Jiayu Xiong
# Reference: STAR-MAE / Distribution-aware Loss Reweighting (DLR), PR-D-25-06353_R2.
import torch
try:
    import torchaudio
except ImportError:
    torchaudio = None
import numpy as np
import random

class KaldiFbankTransform:
    """
    The purpose of this class: Given a waveform (optional mixup), output fbank consistent with the old version,
    and perform the same zero-padding/truncation/SpecAugment/normalization/noise augmentation.
    """

    def __init__(self,
                 num_mel_bins=128,
                 target_length=1024,
                 freqm=0,
                 timem=0,
                 mixup_rate=0.0,
                 norm_mean=0.0,
                 norm_std=1.0,
                 skip_norm=False,
                 noise=False,
                 mel_fmin=50.0,
                 mel_fmax=8000.0):
        """
        Parameters are consistent with the old version.
        """
        self.num_mel_bins = num_mel_bins
        self.target_length = target_length  # Perform cropping or zero-padding on the time dimension
        self.freqm = freqm  # Size of Frequency Masking
        self.timem = timem  # Size of Time Masking
        self.mixup_rate = mixup_rate
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.skip_norm = skip_norm
        self.noise = noise
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax

        if torchaudio is None:
            raise ImportError('torchaudio is required for wav/fbank preprocessing, but it is not installed.')
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freqm)
        self.time_mask = torchaudio.transforms.TimeMasking(timem)

    def __call__(self, 
                 waveform_1: torch.Tensor,
                 sr: int,
                 waveform_2: torch.Tensor = None):
        """
        Args:
            waveform_1 (torch.Tensor): [1, n_samples], must be single-channel.
            sr (int): Sample rate of the waveform.
            waveform_2 (torch.Tensor, optional): Another waveform for mixup. Pass None if not performing mixup.

        Returns:
            tuple:
                fbank (torch.Tensor): Shape [time, freq].
                mix_lambda (float): If mixup was performed, returns the sampled lambda; otherwise, returns 0.
        """
        if waveform_2 is not None:
            # If the sample rate of the second waveform is different, it should be resampled to sr first. Omitted here.
            waveform_2 = waveform_2 - waveform_2.mean()

            # Align lengths
            len1 = waveform_1.shape[1]
            len2 = waveform_2.shape[1]
            if len1 != len2:
                if len1 > len2:
                    # Padding waveform_2
                    tmp = torch.zeros(1, len1)
                    tmp[0, :len2] = waveform_2
                    waveform_2 = tmp
                else:
                    # Truncating waveform_2
                    waveform_2 = waveform_2[:, :len1]

            # Sample lambda from Beta distribution
            # Alternatively, it can be uniform random
            mix_lambda = np.random.beta(10, 10)
            mix_waveform = mix_lambda * waveform_1 + (1 - mix_lambda) * waveform_2
            waveform = mix_waveform - mix_waveform.mean()

        else:
            # Do not perform mixup
            waveform = waveform_1 - waveform_1.mean()
            mix_lambda = 0.0

        fbank = torchaudio.compliance.kaldi.fbank(
            waveform, 
            htk_compat=True, 
            sample_frequency=sr, 
            use_energy=False,
            window_type='hanning', 
            num_mel_bins=self.num_mel_bins, 
            dither=0.0, 
            frame_shift=10,  # 10ms
            low_freq=self.mel_fmin,
            high_freq=self.mel_fmax,
        )
        # fbank shape: [time_frames, num_mel_bins]

        n_frames = fbank.shape[0]
        p = self.target_length - n_frames
        if p > 0:
            # Need to pad with zeros
            fbank = torch.nn.functional.pad(fbank, (0, 0, 0, p))  # (left, right, top, bottom)
        elif p < 0:
            # Truncate
            fbank = fbank[:self.target_length, :]

        # Since the old version applies freq_mask/time_mask on [time, freq], need to first convert to [1, freq, time]
        fbank = fbank.transpose(0, 1).unsqueeze(0)  # [1, freq, time]
        if self.freqm > 0:
            fbank = self.freq_mask(fbank)
        if self.timem > 0:
            fbank = self.time_mask(fbank)
        fbank = fbank.squeeze(0).transpose(0, 1)    # Convert back to [time, freq]

        # Consistent with the old version: (fbank - mean) / std
        if not self.skip_norm:
            fbank = (fbank - self.norm_mean) / self.norm_std

        if self.noise:
            fbank = fbank + torch.rand_like(fbank) * np.random.rand() / 10
            # Randomly roll along the time axis
            fbank = torch.roll(fbank, shifts=np.random.randint(-10, 10), dims=0)

        # Return shape = [time, freq]
        return fbank, mix_lambda
