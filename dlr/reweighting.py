# Author: Jiayu Xiong
# Reference: STAR-MAE / Distribution-aware Loss Reweighting (DLR), PR-D-25-06353_R2.
import torch
import torch.nn as nn
import random
import numpy as np
import argparse

try:
    from .normalizing_flow import DualNormalizingFlow
except ImportError:
    from normalizing_flow import DualNormalizingFlow

class IntervalGenerator:
    def __init__(self):
        self.left = 0.0
        self.right = 1.0

    def generate(self, forward_step: float, backward_step: float, forward_step_2: float = 0., backward_step_2: float = 0.):
        if forward_step_2 != 0:
            delta_left = random.uniform(-backward_step, forward_step)
            delta_right = random.uniform(-forward_step_2, backward_step_2)
        else:
            delta_left = random.uniform(-backward_step, forward_step)
            delta_right = random.uniform(-forward_step, backward_step)

        l = self.left + delta_left
        r = self.right + delta_right
        left = min(l, r)
        right = max(l, r)
        left = max(0.0, left)
        right = min(1.0, right)
        if left >= right:
            left, right = 0.0, 1.0

        self.left = left
        self.right = right
        return left, right


class WeightGenerator:
    def __init__(self, dual_flow: DualNormalizingFlow):
        self.dual_flow = dual_flow
        self.interval_gen = IntervalGenerator()

    def generate(self, x: torch.Tensor, left: float, right: float) -> torch.Tensor:
        with torch.no_grad():
            if x.dim() == 1:
                x = x.unsqueeze(-1)

            cdf_vals = self.dual_flow.cdf(x, use_momentum=True)

            w = torch.ones_like(cdf_vals)
            in_mask = (cdf_vals >= left) & (cdf_vals <= right)
            w[in_mask] = 2.0

        return w.squeeze(-1)


class DistributionAwareLossReweighting(nn.Module):
    def __init__(self, total_iter:int, left:np.float64, right:np.float64, allow_back:bool=False, with_ratio:bool = False, start_iter:int=-1, momentum:float=0.99):
        super().__init__()
        if start_iter == -1:
            start_iter = int(0.6*total_iter)
        self.start_iter = start_iter
        self.total_iter = total_iter
        self.left_range = left/total_iter
        self.right_range = (1-right)/total_iter
        self.back = allow_back
        self.ratio = with_ratio
        self.dual_flow = DualNormalizingFlow(hidden_dim=32, momentum=momentum)
        self.weight_gen = WeightGenerator(self.dual_flow)
        self.igr = IntervalGenerator()
    def pss_module(self, cur_iter:int):
        if cur_iter < self.start_iter:
            return 0
        period = np.pi * 5 *(cur_iter/self.total_iter)
        return np.abs(np.sin(period))
    
    def generate_weight(self, x: torch.Tensor, cur_iter: int):
        if self.back and self.ratio:
            l, r = self.igr.generate(self.left_range*3, self.left_range, self.right_range*3, self.left_range)
        elif self.ratio:
            l, r = self.igr.generate(self.left_range*2, 0, self.right_range*2, 0)
        elif self.back:
            l, r = self.igr.generate(self.left_range*3, self.left_range) 
        else:
            l, r = self.igr.generate(self.left_range*2, 0)
        w = self.generate_weight_for_mse(x, l, r)
        w = 1 + self.pss_module(cur_iter)*(w / w.mean() - 1)
        return w

    def forward(self, x: torch.Tensor, cur_iter: int) -> torch.Tensor:
        return self.generate_weight(x, cur_iter)

    def train_dnf_for_mle(self, x: torch.Tensor) -> torch.Tensor:
        return self.dual_flow.compute_mle_loss(x)

    def update_momentum_flow(self):
        self.dual_flow.update_momentum_flow()

    def generate_weight_for_mse(self, x: torch.Tensor,
                                left: float,
                                right: float) -> torch.Tensor:
        return self.weight_gen.generate(x, left, right)


DLR = DistributionAwareLossReweighting
    

def load_input(path):
    if path.endswith(('.pt', '.pth')):
        tensor = torch.load(path, map_location='cpu')
    elif path.endswith('.npy'):
        tensor = torch.from_numpy(np.load(path))
    elif path.endswith('.npz'):
        with np.load(path) as data:
            if len(data.files) != 1:
                raise ValueError(f'{path} must contain exactly one array, found {data.files}')
            tensor = torch.from_numpy(data[data.files[0]])
    else:
        raise ValueError(f'Unsupported input tensor file extension: {path}')
    if isinstance(tensor, dict):
        raise ValueError(f'{path} must store a tensor directly, not a dict.')
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(-1)
    if tensor.ndim != 2:
        raise ValueError(f'input tensor must have shape (N,) or (N, 1), got {tuple(tensor.shape)}')
    return tensor.to(dtype=torch.float32)


def parse_args():
    parser = argparse.ArgumentParser(description='Run DLR on a user-provided loss/error tensor.')
    parser.add_argument('--input-path', required=True, help='Path to user-provided tensor (.pt/.pth/.npy/.npz).')
    parser.add_argument('--total-iter', type=int, default=100)
    parser.add_argument('--left', type=float, default=0.2)
    parser.add_argument('--right', type=float, default=0.8)
    parser.add_argument('--allow-back', action='store_true')
    parser.add_argument('--with-ratio', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--start-iter', type=int, default=-1)
    parser.add_argument('--momentum', type=float, default=0.99)
    return parser.parse_args()


def main():
    args = parse_args()

    x = load_input(args.input_path)

    dlr = DistributionAwareLossReweighting(
        total_iter=args.total_iter,
        left=args.left,
        right=args.right,
        allow_back=args.allow_back,
        with_ratio=args.with_ratio,
        start_iter=args.start_iter,
        momentum=args.momentum
    )

    for cur_iter in range(args.total_iter):
        loss = dlr.train_dnf_for_mle(x)
        dlr.update_momentum_flow()
        w = dlr.generate_weight(x, cur_iter)
        if cur_iter % 5 == 0:
            print(f"Iter {cur_iter:2d} | "
                  f"IG.left = {dlr.igr.left:.4f}, IG.right = {dlr.igr.right:.4f}, "
                  f"Mean(w) = {w.mean().item():.4f}, MLE-loss = {loss.item():.4f}")

if __name__ == "__main__":
    main()
