# Author: Jiayu Xiong
# Reference: STAR-MAE / Distribution-aware Loss Reweighting (DLR), PR-D-25-06353_R2.
import torch
import torch.nn as nn
import torch.nn.functional as F

class DLRWeightedMSELoss(nn.Module):
    def __init__(self, dlr: nn.Module):
        """
        dlr: Distribution-aware Loss Reweighting module.
        """
        super(DLRWeightedMSELoss, self).__init__()
        self.dlr = dlr

    def forward(self, pred, target, iter):
        """
        pred, target: [B, N, C]
        """
        base_mse = F.mse_loss(pred, target, reduction='none').mean(dim=-1)  # [B, N]

        base_mse_flat = torch.log(base_mse.detach().clamp_min(1e-12)).reshape(-1, 1)

        weights_flat = self.dlr(base_mse_flat, iter)  # [B*N]

        weights = weights_flat.view_as(base_mse)

        weighted_mse = base_mse * weights

        loss = weighted_mse.mean()
        return loss


WeightedMSELoss = DLRWeightedMSELoss
