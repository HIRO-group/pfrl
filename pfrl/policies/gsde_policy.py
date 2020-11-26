import numpy as np
import torch
from torch import nn
from pfrl.distributions import StateDependentNoiseDistribution

class gSDEHead(nn.Module):
    """Head module for a gSDE policy."""

    # distribution: torch.distributions.Distribution

    def __init__(self,
        in_dim: int,
        out_dim: int,
        log_std_min=-20,
        log_std_max=2,
        log_std_init=2.0,
        full_std=True,
        ) -> None:

        super(gSDEHead, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self._mu = nn.Linear(in_dim, out_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.full_std = full_std

        # Reduce the number of parameters if needed
        _log_std = torch.ones(in_dim, out_dim) if self.full_std else torch.ones(in_dim, 1)
        self._log_std = nn.Parameter(_log_std * log_std_init, requires_grad=True)

    def reset_noise(self, batch_size: int = 1) -> None:
        self._reset_noise(self._log_std, batch_size)

    def _reset_noise(self, log_std: torch.Tensor, batch_size: int) -> None:
        """
        Sample weights for the noise exploration matrix,
        using a centered Gaussian distribution.
        :param log_std:
        :param batch_size:
        """
        std = self._std(log_std)
        weights_dist = torch.distributions.Normal(torch.zeros_like(std), std)
        # Reparametrization trick to pass gradients
        self.exploration_mat = weights_dist.rsample()
        # Pre-compute matrices in case of parallel exploration
        self.exploration_matrices = weights_dist.rsample((batch_size,))

    def forward(self, x: torch.Tensor) -> torch.distributions.Distribution:
        # Forward through mean and std layers
        mean_actions = self._mu(x)
        log_std = self._log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        # return self.distribution.update_distribution_parameter(x)
        return StateDependentNoiseDistribution(
            x,
            self.in_dim,
            self.out_dim,
            full_std=self.full_std,
            )
