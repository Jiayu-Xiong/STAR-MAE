# Author: Jiayu Xiong
# Reference: STAR-MAE / Distribution-aware Loss Reweighting (DLR), PR-D-25-06353_R2.
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dlr.reweighting import DistributionAwareLossReweighting
from train_star_mae_dlr import dlr_weighted_mse


def parse_args():
    parser = argparse.ArgumentParser(description='Validate DLR gradients, weights, and interval refinement.')
    parser.add_argument('--pred-path', required=True, help='Path to user-provided prediction tensor (.pt/.pth/.npy/.npz).')
    parser.add_argument('--target-path', required=True, help='Path to user-provided target tensor (.pt/.pth/.npy/.npz).')
    parser.add_argument('--steps', type=int, default=32)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def load_tensor(path, device):
    if path.endswith(('.pt', '.pth')):
        tensor = torch.load(path, map_location=device)
    elif path.endswith('.npy'):
        import numpy as np
        tensor = torch.from_numpy(np.load(path))
    elif path.endswith('.npz'):
        import numpy as np
        with np.load(path) as data:
            if len(data.files) != 1:
                raise ValueError(f'{path} must contain exactly one array, found {data.files}')
            tensor = torch.from_numpy(data[data.files[0]])
    else:
        raise ValueError(f'Unsupported tensor file extension: {path}')
    if isinstance(tensor, dict):
        raise ValueError(f'{path} must store a tensor directly, not a dict.')
    return tensor.to(device=device, dtype=torch.float32)


def main():
    args = parse_args()
    device = torch.device(args.device)
    pred_input = load_tensor(args.pred_path, device)
    target_input = load_tensor(args.target_path, device)
    if pred_input.shape != target_input.shape:
        raise ValueError(f'pred and target shapes must match: {tuple(pred_input.shape)} vs {tuple(target_input.shape)}')
    if pred_input.ndim != 3:
        raise ValueError(f'pred and target must have shape (batch, patches, dim), got {tuple(pred_input.shape)}')

    dlr = DistributionAwareLossReweighting(
        total_iter=args.steps,
        left=0.25,
        right=0.75,
        with_ratio=True,
        momentum=0.99,
    ).to(device)
    optimizer = torch.optim.AdamW(dlr.dual_flow.online_flow.parameters(), lr=1e-3)

    first_mle = None
    last_mle = None
    first_interval = None
    last_interval = None
    last_grad_norm = 0.0
    last_weight_mean = 0.0

    for step in range(args.steps):
        pred = pred_input.detach().clone().requires_grad_(True)
        target = target_input.detach().clone()
        loss, mle_loss = dlr_weighted_mse(pred, target, dlr, optimizer, step)
        loss.backward()
        grad_norm = 0.0
        for param in dlr.dual_flow.online_flow.parameters():
            if param.grad is not None:
                grad_norm += float(param.grad.detach().norm())
        with torch.no_grad():
            patch_mse = ((pred - target) ** 2).mean(dim=-1)
            log_patch_mse = torch.log(patch_mse.clamp_min(1e-12)).reshape(-1, 1)
            weights = dlr.generate_weight(log_patch_mse, step)

        if first_mle is None:
            first_mle = float(mle_loss)
            first_interval = (dlr.igr.left, dlr.igr.right)
        last_mle = float(mle_loss)
        last_interval = (dlr.igr.left, dlr.igr.right)
        last_grad_norm = grad_norm
        last_weight_mean = float(weights.mean())

    print('DLR validation')
    print(f'first_mle={first_mle:.6f}, last_mle={last_mle:.6f}')
    print(f'first_interval=({first_interval[0]:.6f}, {first_interval[1]:.6f})')
    print(f'last_interval=({last_interval[0]:.6f}, {last_interval[1]:.6f})')
    print(f'last_interval_width={last_interval[1] - last_interval[0]:.6f}')
    print(f'last_dnf_grad_norm={last_grad_norm:.6f}')
    print(f'last_weight_mean={last_weight_mean:.6f}')
    assert last_grad_norm > 0.0, 'DNF did not receive gradients.'
    assert 0.0 <= last_interval[0] < last_interval[1] <= 1.0, 'DLR interval is invalid.'
    assert last_interval[1] - last_interval[0] < first_interval[1] - first_interval[0], 'DLR interval did not contract.'


if __name__ == '__main__':
    main()
