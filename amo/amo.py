import torch
import torch.nn as nn
import random
import numpy as np

from momentum import DualProbEnc

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
    def __init__(self, dual_enc: DualProbEnc):
        self.dual_enc = dual_enc
        self.interval_gen = IntervalGenerator()

    def generate(self, x: torch.Tensor, left: float, right: float) -> torch.Tensor:
        with torch.no_grad():
            if x.dim() == 1:
                x = x.unsqueeze(-1)

            cdf_vals = self.dual_enc.cdf(x, use_momentum=True)

            w = torch.ones_like(cdf_vals)
            in_mask = (cdf_vals >= left) & (cdf_vals <= right)
            w[in_mask] = 2.0

        return w.squeeze(-1)


class AMO(nn.Module):
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
        self.dual_enc = DualProbEnc(momentum=momentum)
        self.weight_gen = WeightGenerator(self.dual_enc)
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

    def train_dual_encoder_for_mle(self, x: torch.Tensor) -> torch.Tensor:
        return self.dual_enc.compute_mle_loss(x)

    def update_momentum_flow(self):
        self.dual_enc.update_momentum_flow()

    def generate_weight_for_mse(self, x: torch.Tensor,
                                left: float,
                                right: float) -> torch.Tensor:
        return self.weight_gen.generate(x, left, right)
    

def main():

    total_iter = 100
    left = 0.2
    right = 0.8
    allow_back = False
    with_ratio = True
    start_iter = -1
    momentum = 0.99

    amo = AMO(
        total_iter=total_iter,
        left=left,
        right=right,
        allow_back=allow_back,
        with_ratio=with_ratio,
        start_iter=start_iter,
        momentum=momentum
    )

    for cur_iter in range(total_iter):
        batch_size = 128
        x = torch.randn(batch_size, 1)
        loss = amo.train_dual_encoder_for_mle(x)
        amo.update_momentum_flow()
        w = amo.generate_weight(x, cur_iter)
        if cur_iter % 5 == 0:
            print(f"Iter {cur_iter:2d} | "
                  f"IG.left = {amo.igr.left:.4f}, IG.right = {amo.igr.right:.4f}, "
                  f"Mean(w) = {w.mean().item():.4f}, MLE-loss = {loss.item():.4f}")

if __name__ == "__main__":
    main()
