import torch
import torch.nn as nn

class WeightedMSELoss(nn.Module):
    def __init__(self, amo: nn.Module):
        """
        amo: generator
        """
        super(WeightedMSELoss, self).__init__()
        self.amo = amo

    def forward(self, pred, target, iter):
        """
        pred, target: [B, N, C]
        """
        base_mse = ((pred - target)**2).mean(dim=-1)  # [B, N]

        base_mse_flat = base_mse.view(-1)  # [B*N]

        weights_flat = self.amo.generate_weight(base_mse_flat, iter)  # [B*N]

        weights = weights_flat.view_as(base_mse)

        weighted_mse = base_mse * weights

        loss = weighted_mse.mean()
        return loss
