# Author: Jiayu Xiong
# Reference: STAR-MAE / Distribution-aware Loss Reweighting (DLR), PR-D-25-06353_R2.
import argparse
import os
import sys

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def parse_args():
    parser = argparse.ArgumentParser(description='Validate separation for user-provided STAR-MAE encoder/decoder masks.')
    parser.add_argument('--encoder-mask-path', required=True, help='Path to encoder mask tensor (.pt/.pth/.npy/.npz).')
    parser.add_argument('--decoder-mask-path', required=True, help='Path to decoder mask tensor (.pt/.pth/.npy/.npz).')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def load_mask(path, device):
    if path.endswith(('.pt', '.pth')):
        mask = torch.load(path, map_location=device)
    elif path.endswith('.npy'):
        mask = torch.from_numpy(np.load(path))
    elif path.endswith('.npz'):
        with np.load(path) as data:
            if len(data.files) != 1:
                raise ValueError(f'{path} must contain exactly one array, found {data.files}')
            mask = torch.from_numpy(data[data.files[0]])
    else:
        raise ValueError(f'Unsupported mask file extension: {path}')
    if isinstance(mask, dict):
        raise ValueError(f'{path} must store a tensor directly, not a dict.')
    return mask.to(device=device, dtype=torch.bool)


def main():
    args = parse_args()
    device = torch.device(args.device)
    en_mask = load_mask(args.encoder_mask_path, device)
    de_mask = load_mask(args.decoder_mask_path, device)
    if en_mask.shape != de_mask.shape:
        raise ValueError(f'encoder and decoder masks must have the same shape: {tuple(en_mask.shape)} vs {tuple(de_mask.shape)}')
    decoder_targets = ~de_mask
    encoder_visible = ~en_mask
    overlap = decoder_targets & encoder_visible
    print(
        f'overlap={int(overlap.sum().item())}, '
        f'targets={decoder_targets.sum(dim=1).tolist()}, '
        f'encoder_visible={encoder_visible.sum(dim=1).tolist()}, '
        f'encoder_masked={en_mask.sum(dim=1).tolist()}')
    assert not overlap.any(), 'Decoder targets include encoder-visible patches.'


if __name__ == '__main__':
    main()
