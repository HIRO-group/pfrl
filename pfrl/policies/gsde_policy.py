import torch
from torch import nn
from pfrl.distributions import StateDependentNoiseDistribution

class gSDEHead(nn.Module):
    """Head module for a deterministic policy."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.distribution = StateDependentNoiseDistribution(in_dim, out_dim)

    def reset_noise(self):
        self.distribution.reset_noise()

    def forward(self, x):
        return self.distribution.update_distribution_parameter(x)
