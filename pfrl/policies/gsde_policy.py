import numpy as np
import torch
from torch import nn
import pfrl


class gSDEHead(nn.Module):
    """Head module for a gSDE policy."""


    def __init__(self,
        in_dim: int,
        out_dim: int,
        log_std_init=2.0,
        full_std=True,
        use_expln=False,
        epsilon: float = 1e-6,
        ) -> None:

        super(gSDEHead, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.full_std = full_std
        self.use_expln = use_expln
        self.epsilon = epsilon
        self.initialized = False

        # Reduce the number of parameters if needed
        self._mu = nn.Linear(in_dim, out_dim)
        _log_std = torch.ones(in_dim, out_dim) if self.full_std else torch.ones(in_dim, 1)
        self._log_std = nn.Parameter(_log_std * log_std_init, requires_grad=True)
        # self.reset_noise()

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

    def _std(self, log_std: torch.Tensor) -> torch.Tensor:
        """
        Get the standard deviation from the learned parameter
        (log of it by default). This ensures that the std is positive.
        :param log_std:
        :return:
        """
        if self.use_expln:
            # From gSDE paper, it allows to keep variance
            # above zero and prevent it from growing too fast
            below_threshold = torch.exp(log_std) * (log_std <= 0)
            # Avoid NaN: zeros values that are below zero
            safe_log_std = log_std * (log_std > 0) + self.epsilon
            above_threshold = (torch.log1p(safe_log_std) + 1.0) * (log_std > 0)
            std = below_threshold + above_threshold
        else:
            # Use normal exponential
            std = torch.exp(log_std)

        if self.full_std:
            return std
        # Reduce the number of parameters:
        return torch.ones(self.in_dim, self.out_dim).to(log_std.device) * std

    def forward(self, x: torch.Tensor, deterministic: bool = False) -> torch.distributions.Distribution:
        if not self.initialized:
            self.reset_noise()

        # Forward through mean and std layers
        mean_actions = self._mu(x)
        std = self._std(self._log_std)

        return pfrl.distributions.StateDependentNoiseDistribution(
            x,
            mean_actions,
            std,
            self.in_dim,
            self.out_dim,
            self.epsilon,
            self.exploration_mat,
            self.exploration_matrices,
            deterministic
        )
