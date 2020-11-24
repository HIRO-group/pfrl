import pfrl
import torch
from torch import nn


class gSDEPolicy(nn.Module):
    def __init__(self, state_dim, goal_dim, action_dim, latent_dim=256, latent_sde_dim=None, full_std=True, log_std_init=-2.0, log_std_min=-20, log_std_max=2, **kwargs):
        super().__init__(**kwargs)
        self.latent_pi = nn.Sequential(
                nn.Linear(state_dim + goal_dim, latent_dim),
                nn.ReLU(),
                nn.Linear(latent_dim, latent_dim),
                nn.ReLU())
        self.dist = pfrl.distributions.StateDependentNoiseDistribution()

        self.mean_actions_net = nn.Linear(latent_dim, action_dim)
        # Reduce the number of parameters if needed
        log_std = torch.ones(latent_dim, self.action_dim) if self.full_std else torch.ones(latent_dim, 1)
        # Transform it to a parameter so it can be optimized
        self.log_std = nn.Parameter(log_std * log_std_init, requires_grad=True)
        self.dist.reset_noise(log_std)

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def _forward(self, mean):
        latent_pi = self.latent_pi(mean)
        mean_actions = self.mu(latent_pi)
        log_std = self.log_std(latent_pi)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        self.dist.set_latent_pi(latent_pi)
        self.dist.set_log_std(log_std)
        self.dist.set_latent_sde(latent_pi)

    def forward(self, mean):
        """
        Return Distribution
        """
        self._forward(mean)
        return self.dist
