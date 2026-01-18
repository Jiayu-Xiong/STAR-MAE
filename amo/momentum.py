import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

class CouplingLayer(nn.Module):

    def __init__(self, hidden_dim=64):
        super(CouplingLayer, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2)  # scale, shift
        )

    def forward(self, x, mask):
        x_t = x * mask
        out = self.net(x_t)
        s = out[:, 0:1]  # (batch, 1)
        t = out[:, 1:2]  # (batch, 1)

        s = torch.sigmoid(out[:,0:1]) * 2 - 1  # smooth

        y = x * (1 - mask) * torch.exp(s) + t * (1 - mask) + x_t

        log_det = torch.sum((1 - mask) * s, dim=1, keepdim=True)
        return y, log_det


class RealNVPFlow(nn.Module):
    def __init__(self, num_layers=4, hidden_dim=64):

        super(RealNVPFlow, self).__init__()
        self.num_layers = num_layers
        layers = []
        for i in range(num_layers):
            layer = CouplingLayer(hidden_dim=hidden_dim)
            layers.append(layer)
        self.layers = nn.ModuleList(layers)
        self.base_dist = torch.distributions.Normal(0.0, 1.0)

    def forward_transform(self, x):
        z = x
        log_det_sum = 0.0
        for i, layer in enumerate(self.layers):
            mask = (i % 2) * torch.ones_like(z)  # 0 or 1
            z, log_det = layer(z, mask)
            log_det_sum = log_det_sum + log_det
        return z, log_det_sum

    def inverse_transform(self, z):
        x = z
        log_det_sum = 0.0
        for i, layer in reversed(list(enumerate(self.layers))):
            mask = (i % 2) * torch.ones_like(x)
            x_t = x * mask
            out = layer.net(x_t)  # (batch,2)
            s = torch.tanh(out[:,0:1])
            t = out[:,1:2]
            cur_y = x

            x_in = (cur_y - x_t - t*(1-mask)) * torch.exp(-s) * (1-mask)
            x = x_t + x_in

            log_det = torch.sum((1 - mask)*(-s), dim=1, keepdim=True)
            log_det_sum += log_det
        return x, log_det_sum

    def log_prob(self, x):
        z, log_det = self.forward_transform(x)
        log_p_z = self.base_dist.log_prob(z).sum(dim=1, keepdim=True)  # sum(dim=1)
        return log_p_z + log_det

    def cdf(self, x):
        z, _ = self.forward_transform(x)  # (batch,1)
        cdf_vals = 0.5*(1.0 + torch.erf(z / math.sqrt(2.0)))  #  base_dist = N(0,1), cdf
        return cdf_vals

    def sample(self, num_samples=1):
        z_samples = self.base_dist.sample((num_samples,1))  # (num_samples,1)
        x, _ = self.inverse_transform(z_samples)
        return x


class DualProbEnc(nn.Module):
    def __init__(self, num_layers=4, hidden_dim=64, momentum=0.9):
        super(DualProbEnc, self).__init__()
        self.momentum = 1 - momentum

        self.online_flow = RealNVPFlow(num_layers=num_layers, hidden_dim=hidden_dim)
        self.momentum_flow = RealNVPFlow(num_layers=num_layers, hidden_dim=hidden_dim)

        self.initialize_momentum_flow()

    def initialize_momentum_flow(self):
        for p_online, p_mom in zip(self.online_flow.parameters(), self.momentum_flow.parameters()):
            p_mom.data.copy_(p_online.data)

    def update_momentum_flow(self):
        with torch.no_grad():
            for p_online, p_mom in zip(self.online_flow.parameters(), self.momentum_flow.parameters()):
                delta = p_online.data - p_mom.data
                p_mom.data.add_(self.momentum * delta)

    def compute_mle_loss(self, x):
        log_p = self.online_flow.log_prob(x)  # (batch,1)
        # MLE
        nll = -log_p.mean()
        loss = nll
        return loss

    def forward(self, x, use_momentum=False):
        if use_momentum:
            return self.momentum_flow.log_prob(x)
        else:
            return self.online_flow.log_prob(x)

    def cdf(self, x, use_momentum=False):
        if use_momentum:
            return self.momentum_flow.cdf(x)
        else:
            return self.online_flow.cdf(x)
