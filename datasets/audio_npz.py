# -*- coding: utf-8 -*-
# Author: Jiayu Xiong
# Reference: STAR-MAE / Distribution-aware Loss Reweighting (DLR), PR-D-25-06353_R2.
import os
import csv
import time
import argparse
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
try:
    import torchaudio
except ImportError:
    torchaudio = None


def cfg(args, name: str, default=None):
    if isinstance(args, dict):
        return args.get(name, default)
    return getattr(args, name, default)


def cfg_any(args, names, default=None):
    for name in names:
        value = cfg(args, name, None)
        if value is not None:
            return value
    return default


# -------------------------
# CSV / label helpers
# -------------------------
def make_index_dict(label_csv: str) -> Dict[str, int]:
    index_dict: Dict[str, int] = {}
    with open(label_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # AudioSet canonical csv usually has: index, mid, display_name
            if 'mid' in row and 'index' in row:
                index_dict[row['mid']] = int(row['index'])
            elif 'label' in row and 'index' in row:
                index_dict[row['label']] = int(row['index'])
            else:
                vals = list(row.values())
                if len(vals) >= 2:
                    index_dict[str(vals[1]).strip()] = int(vals[0])
    return index_dict


def infer_label_num(label_csv: str) -> int:
    index_dict = make_index_dict(label_csv)
    if len(index_dict) == 0:
        raise RuntimeError(f'empty label index csv: {label_csv}')
    return max(index_dict.values()) + 1


# -------------------------
# metadata loader
# keep old label logic: datum['labels'] is a comma-joined string,
# label_indices is built by index_dict[label_str]
# -------------------------
def load_audio_meta(csv_path: str, wav_root: str) -> List[dict]:
    data: List[dict] = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f, skipinitialspace=True)
        for row in reader:
            if not row:
                continue
            first = row[0].strip().lower()
            if first in {'video_id', 'ytid', 'id', 'wav'}:
                continue

            raw0 = row[0].strip()
            if raw0.endswith('.wav') or '/' in raw0 or '\\' in raw0:
                wav_path = raw0 if os.path.isabs(raw0) else os.path.join(wav_root, raw0)
            else:
                wav_path = os.path.join(wav_root, 'Y' + raw0 + '.wav')

            labels = ''
            if len(row) >= 4:
                # old AudioSet style: video_id, start, end, label1, label2, ...
                labels = ','.join([x.strip() for x in row[3:] if x.strip() != ''])

            data.append({'wav': wav_path, 'labels': labels})
    return data


# -------------------------
# feature helpers
# -------------------------
def waveform_to_fbank(waveform: torch.Tensor,
                      sample_rate: int,
                      num_mel_bins: int,
                      mel_fmin: float,
                      mel_fmax: float) -> torch.Tensor:
    if torchaudio is None:
        raise ImportError('torchaudio is required for wav preprocessing, but it is not installed.')
    return torchaudio.compliance.kaldi.fbank(
        waveform,
        htk_compat=True,
        sample_frequency=sample_rate,
        use_energy=False,
        window_type='hanning',
        num_mel_bins=num_mel_bins,
        dither=0.0,
        frame_shift=10,
        low_freq=mel_fmin,
        high_freq=mel_fmax,
    )


def normalize_fbank(fbank: torch.Tensor, mean: float, std: float, skip_norm: bool) -> torch.Tensor:
    fbank = fbank.to(torch.float32)
    if skip_norm:
        return fbank.contiguous()
    denom = std if abs(std) > 1e-12 else 1.0
    return ((fbank - mean) / denom).contiguous()


def pad_or_crop_fbank(fbank: torch.Tensor, target_length: int) -> torch.Tensor:
    t = int(fbank.shape[0])
    if t < target_length:
        fbank = F.pad(fbank, (0, 0, 0, target_length - t))
    elif t > target_length:
        fbank = fbank[:target_length, :]
    return fbank.contiguous()


def atomic_save_npz(path: str, **arrays) -> None:
    tmp_path = path + '.tmp'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(tmp_path, 'wb') as f:
        np.savez(f, **arrays)
    os.replace(tmp_path, path)


# -------------------------
# preprocess dataset
# old label logic preserved
# x: normalized variable-length fbank
# y: one-hot from index_dict
# return: (mean, std, nframes), label_t
# -------------------------
class AudiosetNPZPreprocessor(Dataset):
    def __init__(self, csv_path: str, args, label_csv: str):
        self.csv_path = csv_path
        self.args = args
        self.wav_root = cfg(args, 'wav_root')
        self.root = cfg_any(args, ('root', 'npz_root'))
        self.sample_rate = int(cfg(args, 'sample_rate', 32000))
        self.melbins = int(cfg(args, 'melbins', 128))
        self.mel_fmin = float(cfg(args, 'mel_fmin', 50.0))
        self.mel_fmax = float(cfg(args, 'mel_fmax', 14000.0))
        self.norm_mean = float(cfg(args, 'mean', 0.0))
        self.norm_std = float(cfg(args, 'std', 1.0))
        self.skip_norm = bool(cfg(args, 'skip_norm', False))

        self.data = load_audio_meta(self.csv_path, self.wav_root)
        self.index_dict = make_index_dict(label_csv)
        self.label_num = infer_label_num(label_csv)
        self._resampler_cache = {}

        os.makedirs(self.root, exist_ok=True)
        print('--------------- the npz preprocessor ---------------')
        print(f'csv={self.csv_path}')
        print(f'wav_root={self.wav_root}')
        print(f'npz_root={self.root}')
        print(f'label_csv={label_csv}')
        print(f'melbins={self.melbins}, sample_rate={self.sample_rate}, '
              f'mel_fmin={self.mel_fmin}, mel_fmax={self.mel_fmax}')
        print(f'skip_norm={self.skip_norm}, mean={self.norm_mean:.6f}, std={self.norm_std:.6f}')
        print('save order: wav -> fbank -> optional norm -> save variable-length npz')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        datum = self.data[index]
        label_indices = np.zeros(self.label_num, dtype=np.float32)
        if datum['labels']:
            for label_str in datum['labels'].split(','):
                label_str = label_str.strip()
                if label_str == '':
                    continue
                label_indices[int(self.index_dict[label_str])] = 1.0
        label_t = torch.from_numpy(label_indices)

        rel = os.path.relpath(datum['wav'], self.wav_root)
        out_base = os.path.join(self.root, os.path.splitext(rel)[0])
        data_path = out_base + '.npz'
        os.makedirs(os.path.dirname(data_path), exist_ok=True)

        if os.path.exists(data_path):
            return (np.float64(0), np.float64(0), np.float64(1)), label_t

        try:
            if torchaudio is None:
                raise ImportError('torchaudio is required for wav preprocessing, but it is not installed.')
            waveform, sr = torchaudio.load(datum['wav'], backend='sox')
        except Exception:
            print(datum['wav'])
            return (np.float64(0), np.float64(0), np.float64(1)), label_t

        if waveform.ndim == 2 and waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != self.sample_rate:
            if sr not in self._resampler_cache:
                self._resampler_cache[sr] = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = self._resampler_cache[sr](waveform)

        fbank = waveform_to_fbank(
            waveform,
            self.sample_rate,
            self.melbins,
            self.mel_fmin,
            self.mel_fmax,
        )

        x64 = fbank.to(torch.float64)
        n = x64.numel()
        s = x64.sum()
        ss = (x64 * x64).sum()
        mean = (s / n).item()
        var = (ss / n - (s / n) ** 2)
        var = torch.clamp_min(var, 0.0).item()
        std = float(var ** 0.5)
        nframes = int(fbank.shape[0])

        # preprocess save: no pad here, only optional norm
        fbank = normalize_fbank(fbank, self.norm_mean, self.norm_std, self.skip_norm)

        data_np = fbank.cpu().numpy().astype(np.float32, copy=True)
        label_np = label_indices
        atomic_save_npz(data_path, x=data_np, y=label_np)

        return (mean, std, nframes), label_t


# -------------------------
# npz reader dataset
# read variable-length normalized npz, pad/crop here, return item+label
# -------------------------
class AudiosetSpec(Dataset):
    def __init__(self, csv_path: str, args, label_csv: Optional[str] = None):
        self.csv_path = csv_path
        self.args = args
        self.wav_root = cfg(args, 'wav_root')
        self.root = cfg_any(args, ('root', 'npz_root'))
        self.target_len = int(cfg(args, 'target_length'))
        self.melbins = int(cfg(args, 'melbins', 128))

        self.data = load_audio_meta(self.csv_path, self.wav_root)
        self.label_csv = label_csv
        self.label_num = infer_label_num(label_csv) if label_csv else None

        print('--------------- the {} dataloader (NPZ) ---------------'.format(cfg(args, 'mode', 'train')))
        print('read-only npz dataset')
        print('save order: optional norm -> variable length npz')
        print('read order: load npz -> pad/crop to target_length -> return fbank, label')
        print('mel_fmin/mel_fmax/sample_rate are fixed by the precomputed npz and are not applied while reading')
        print(f'target_length={self.target_len}, melbins={self.melbins}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        datum = self.data[index]
        rel = os.path.relpath(datum['wav'], self.wav_root)
        data_path = os.path.join(self.root, os.path.splitext(rel)[0] + '.npz')
        with np.load(data_path, allow_pickle=False) as f:
            x = f['x']
            y = f['y']

        fbank = torch.tensor(np.asarray(x, dtype=np.float32, order='C'), dtype=torch.float32)
        label = torch.tensor(np.asarray(y, dtype=np.float32, order='C'), dtype=torch.float32)
        fbank = pad_or_crop_fbank(fbank, self.target_len)
        if fbank.ndim != 2 or fbank.shape[0] != self.target_len or fbank.shape[1] != self.melbins:
            raise RuntimeError(
                f'Invalid NPZ feature shape after pad/crop: {tuple(fbank.shape)} from {data_path}; '
                f'expected ({self.target_len}, {self.melbins}).')
        if self.label_num is not None and label.numel() != self.label_num:
            label = torch.zeros(self.label_num, dtype=torch.float32)
        else:
            label = label.reshape(-1)
        return fbank.contiguous(), label.contiguous()


# -------------------------
# optional smoke test
# -------------------------
def smoke_test_loader(args) -> None:
    args.mode = 'train'
    dataset = AudiosetSpec(args.csv, args, label_csv=args.label_csv)
    if len(dataset) == 0:
        print('dataset is empty, skip smoke test')
        return

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        prefetch_factor=args.prefetch if args.workers > 0 else None,
        pin_memory=args.pin_memory,
        persistent_workers=True if args.workers > 0 else False,
        drop_last=args.drop_last,
    )
    xb, yb = next(iter(loader))
    print('--------------- dataloader smoke test ---------------')
    print(f'x shape: {tuple(xb.shape)}')
    print(f'y shape: {tuple(yb.shape)}')
    print(f'x dtype: {xb.dtype}, y dtype: {yb.dtype}')
    print('-----------------------------------------------------')


# -------------------------
# preprocess main
# iterate preprocessor dataset to write npz files
# -------------------------
def preprocess_dataset(args) -> None:
    args.mode = 'preprocess'
    dataset = AudiosetNPZPreprocessor(args.csv, args, label_csv=args.label_csv)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        prefetch_factor=args.prefetch if args.workers > 0 else None,
        pin_memory=args.pin_memory,
        persistent_workers=True if args.workers > 0 else False,
        drop_last=False,
    )

    done = 0
    skip = 0
    start = time.time()
    for i, (stats, labels) in enumerate(loader, start=1):
        nframes = stats[2]
        done += int((nframes > 1).sum().item())
        skip += int((nframes == 1).sum().item())
        if i % 50 == 0:
            print(f'[{i}/{len(loader)}] processed={done} skipped_or_existing={skip}')

    elapsed = time.time() - start
    print('--------------- preprocess finished ---------------')
    print(f'dataset size           : {len(dataset)}')
    print(f'processed              : {done}')
    print(f'skipped/existing/fail  : {skip}')
    print(f'time                   : {elapsed:.2f}s')


# -------------------------
# main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description='audiosetspec npz preprocess + dataset reader')
    parser.add_argument('--csv', type=str, required=True, help='Path to the metadata CSV file')
    parser.add_argument('--root', type=str, required=True, help='Root directory for output/read NPZ files')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--workers', type=int, default=24, help='Number of DataLoader workers')
    parser.add_argument('--prefetch', type=int, default=2, help='DataLoader prefetch_factor (>=2 when workers>0)')
    parser.add_argument('--pin-memory', action='store_true', help='Enable pin_memory for faster HtoD copies')
    parser.add_argument('--iters', type=int, default=10, help='Unused in preprocess main. Kept for compatibility.')
    parser.add_argument('--warmup', type=int, default=3, help='Unused in preprocess main. Kept for compatibility.')
    parser.add_argument('--target-length', type=int, default=992, help='Target spectrogram frames (time dimension)')
    parser.add_argument('--mean', type=float, default=0.0, help='Global mean for normalization')
    parser.add_argument('--std', type=float, default=1.0, help='Global std for normalization')
    parser.add_argument('--skip-norm', action='store_true', help='Skip normalization')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Unused in preprocess main. Kept for compatibility.')
    parser.add_argument('--drop-last', action='store_true', help='Drop last incomplete batch during smoke test')

    parser.add_argument('--wav-root', type=str, required=True, help='Root directory of source WAV files')
    # Mel reference presets:
    # PANNs: sr=32000, mel_fmin=50, mel_fmax=14000.
    # AudioMAE: sr=16000, mel_fmin=50, mel_fmax=8000.
    parser.add_argument(
        '--sample-rate',
        type=int,
        default=32000,
        help='Audio sample rate. Presets: PANNs=32000, AudioMAE=16000.')
    parser.add_argument('--melbins', type=int, default=128, help='Number of mel bins')
    parser.add_argument(
        '--mel-fmin',
        type=float,
        default=50.0,
        help='Minimum mel frequency in Hz. Presets: PANNs=50, AudioMAE=50.')
    parser.add_argument(
        '--mel-fmax',
        type=float,
        default=14000.0,
        help='Maximum mel frequency in Hz. Presets: PANNs=14000, AudioMAE=8000.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing npz files')
    parser.add_argument('--no-smoke-test', action='store_true', help='Skip the post-preprocess dataloader smoke test')
    parser.add_argument('--label-csv', type=str, required=True, help='Label index CSV used by the one-hot label logic')
    args = parser.parse_args()

    preprocess_dataset(args)
    if not args.no_smoke_test:
        smoke_test_loader(args)


if __name__ == '__main__':
    main()
