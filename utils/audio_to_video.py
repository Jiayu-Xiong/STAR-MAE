# Author: Jiayu Xiong
# Reference: STAR-MAE / Distribution-aware Loss Reweighting (DLR), PR-D-25-06353_R2.
import torch
try:
    import torchaudio
except ImportError:
    torchaudio = None

class AudioToVideoTransform:
    def __init__(self, sample_rate, n_fft, H, W, T, overlap_rate=0.0,
                 per_frame_overlap=False, use_mel=False, hop_length=None,
                 mel_fmin=50.0, mel_fmax=8000.0, start_sample=0,
                 overlap_rates=None):
        """
        Initialize the AudioToVideoTransform class with the given parameters.
        For pretrain, but something has changed while visualization step.
        
        Args:
            sample_rate (int): Sampling rate of the audio.
            n_fft (int): FFT window size.
            H (int): Number of frequency bins (height).
            W (int): Number of time steps per frame (width).
            T (int): Number of frames.
            overlap_rate (float, optional): Maximum overlap rate (between 0 and 1). Defaults to 0.0.
            per_frame_overlap (bool, optional): Whether to use different overlap rates for each frame. Defaults to False.
            use_mel (bool, optional): Whether to convert to Mel spectrogram. Defaults to False.
            hop_length (int): User-provided hop length for spectrogram.
            start_sample (int): Deterministic start sample selected by the user.
            overlap_rates (list[float], optional): User-provided per-frame overlap rates.
        """
        self.sample_rate = sample_rate  # Sampling rate
        self.n_fft = n_fft  # FFT window size
        self.H = H  # Number of frequency bins (height)
        self.W = W  # Number of time steps per frame (width)
        self.T = T  # Number of frames
        self.overlap_rate = overlap_rate  # Maximum overlap rate (between 0 and 1)
        self.per_frame_overlap = per_frame_overlap  # Whether to use different overlap rates for each frame
        self.use_mel = use_mel  # Whether to convert to Mel spectrogram
        self.hop_length = hop_length
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.start_sample = start_sample
        self.overlap_rates = overlap_rates

        self.warning = False
        if torchaudio is None:
            raise ImportError('torchaudio is required for wav preprocessing, but it is not installed.')

        # Check the relationship between H and n_fft
        n_freqs = self.n_fft // 2 + 1
        if self.H > n_freqs:
            print(f"Warning: Specified number of frequency bins H={self.H} is greater than n_fft // 2 + 1 = {n_freqs}. Adjusting H to {n_freqs}.")
            self.H = n_freqs

    def __call__(self, waveform):
        """
        Transform the input waveform into frames with fbank features.
        
        Args:
            waveform (torch.Tensor): Tensor of shape (channel, samples).
        
        Returns:
            torch.Tensor: Tensor of shape (T, H, W).
        """
        # Get the length of the input audio
        _, num_samples = waveform.shape

        # Calculate the hop_length range
        max_hop_length = int((num_samples - self.n_fft) / (self.T * self.W - 1))
        max_hop_length = max(max_hop_length, 10)  # Ensure at least 10
        max_hop_length = min(max_hop_length, self.n_fft)
        # Set minimum hop_length
        min_hop_length = min(100, max_hop_length)
        if self.hop_length is None:
            raise ValueError(
                f'hop_length must be provided by the caller. Valid range for this input is '
                f'[{min_hop_length}, {max_hop_length}].')
        elif self.hop_length > max_hop_length:
            hop_length = max_hop_length
            if not self.warning:
                print(f"The chosen hop_length is too large and has been automatically adjusted to the maximum value of {max_hop_length}.")
                self.warning = True
        elif self.hop_length < min_hop_length:
            hop_length = max_hop_length
            if not self.warning:
                print(f"The chosen hop_length is too small and has been automatically adjusted to the minimum value of {min_hop_length}.")
                self.warning = True
        else:
            hop_length = self.hop_length

        if self.per_frame_overlap:
            if self.overlap_rates is None:
                raise ValueError('overlap_rates must be provided when per_frame_overlap=True.')
            if len(self.overlap_rates) != self.T:
                raise ValueError(f'overlap_rates must contain T={self.T} values.')
            overlap_rates = self.overlap_rates
            frame_steps = [max(int(self.W * (1 - orate)), 1) for orate in overlap_rates]
            # Calculate the starting position for each frame
            start_positions = [0]
            for i in range(1, self.T):
                start_pos = start_positions[i-1] + frame_steps[i-1]
                start_positions.append(start_pos)
            N_required = start_positions[-1] + self.W
        else:
            overlap_rate_value = self.overlap_rate
            frame_step = max(int(self.W * (1 - overlap_rate_value)), 1)
            start_positions = [i * frame_step for i in range(self.T)]
            N_required = start_positions[-1] + self.W

        # Calculate the total number of spectrogram time steps required
        total_samples_required = (N_required - 1) * hop_length + self.n_fft

        # Calculate the maximum starting sample
        max_start_sample = num_samples - total_samples_required
        if max_start_sample < 0:
            # Input audio length is insufficient, perform padding
            padding = -max_start_sample
            waveform = torch.nn.functional.pad(waveform, (0, padding))
            num_samples += padding
            max_start_sample = 0  # After padding, the starting sample can only be 0

        start_sample = int(self.start_sample)
        if start_sample < 0 or start_sample > max_start_sample:
            raise ValueError(f'start_sample must be in [0, {max_start_sample}], got {start_sample}.')
        end_sample = start_sample + total_samples_required

        # Extract the required audio segment
        waveform = waveform[:, start_sample:end_sample]

        # Initialize spectrogram transformation
        if self.use_mel:
            spectrogram_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=hop_length,
                n_mels=self.H,
                f_min=self.mel_fmin,
                f_max=self.mel_fmax,
            )
        else:
            spectrogram_transform = torchaudio.transforms.Spectrogram(
                n_fft=self.n_fft,
                hop_length=hop_length
            )

        # Compute spectrogram
        spectrogram = spectrogram_transform(waveform)
        # If there are batch or channel dimensions, squeeze them
        spectrogram = spectrogram.squeeze()

        # For Mel spectrogram, take the logarithm
        if self.use_mel:
            spectrogram = torchaudio.functional.amplitude_to_DB(
                spectrogram,
                multiplier=10,
                amin=1e-10,
                db_multiplier=0
            )

        freq_bins, time_steps = spectrogram.shape

        # Adjust frequency dimension to H
        if freq_bins != self.H:
            spectrogram = torch.nn.functional.interpolate(
                spectrogram.unsqueeze(0).unsqueeze(0),  # Add batch and channel dimensions
                size=(self.H, time_steps),
                mode='bilinear',
                align_corners=False
            ).squeeze(0).squeeze(0)

        # Ensure the spectrogram has the required number of time steps
        if time_steps < N_required:
            padding = N_required - time_steps
            spectrogram = torch.nn.functional.pad(spectrogram, (0, padding))
            time_steps = N_required
        elif time_steps > N_required:
            spectrogram = spectrogram[:, :N_required]
            time_steps = N_required

        # Extract frames
        frames = []
        for start in start_positions:
            end = start + self.W
            if end > time_steps:
                pad_size = end - time_steps
                frame = torch.nn.functional.pad(spectrogram[:, start:], (0, pad_size))
            else:
                frame = spectrogram[:, start:end]
            frames.append(frame)

        # Stack frames
        frames = torch.stack(frames, dim=0)  # Shape: (T, H, W)

        return frames

def test_transform():
    """
    Test the AudioToVideoTransform class with a user-provided audio signal.
    """
    # Set test parameters
    duration = 10.0  # Audio duration in seconds
    sample_rate = 16000  # Sample rate 16kHz
    n_fft = 1024  # FFT window length
    H = 224  # Number of frequency bins (height)
    W = 224  # Number of time steps per frame (width)
    T = 4  # Number of frames
    overlap_rate = 0.2  # Maximum overlap rate
    per_frame_overlap = True  # Whether to use different overlap rates for each frame
    use_mel = False  # Use Mel spectrogram
    hop_length = 160
    overlap_rates = [0.2, 0.2, 0.2, 0.2]

    # Initialize AudioToVideoTransform class
    transform = AudioToVideoTransform(
        sample_rate, n_fft, H, W, T, overlap_rate, per_frame_overlap, use_mel,
        hop_length=hop_length, overlap_rates=overlap_rates)

    # Replace this with a user-provided waveform in real use.
    num_samples = int(sample_rate * duration)
    waveform = torch.zeros(1, num_samples)

    print(waveform.shape)
    # Apply transformation
    frames = transform(waveform)

    # Print output
    print("Output shape:", frames.shape)
    print("Expected shape: (T, H, W) =", (T, H, W))

# Uncomment the following line to run the test function
# test_transform()
