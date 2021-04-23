from numbers import Number

import torch
from torch import nn
from torch.distributions import constraints, Distribution


def sum_independent_dims(tensor: torch.Tensor) -> torch.Tensor:
    """
    Continuous actions are usually considered to be independent,
    so we can sum components of the ``log_prob`` or the entropy.
    :param tensor: shape: (n_batch, n_actions) or (n_batch,)
    :return: shape: (n_batch,)
    """
    if len(tensor.shape) > 1:
        tensor = tensor.sum(dim=1)
    else:
        tensor = tensor.sum()
    return tensor


class StateDependentNoiseDistribution(Distribution):
    """Delta distribution.

    This is used

    Args:
        loc (float or Tensor): location of the distribution.
    """

    arg_constraints = {}
    # mypy complains about the type of `support` since it is initialized
    # as None in `torch.distributions.Distribution` as of torch==1.5.0.
    has_rsample = True

    def __init__(self,
        x,
        mean_actions: torch.Tensor,
        std: torch.Tensor,
        in_dim: int,
        out_dim: int,
        epsilon: float,
        exploration_mat: torch.Tensor,
        exploration_matrices: torch.Tensor,
        deterministic=False,
        validate_args=None):
        batch_shape = torch.Size()
        super(StateDependentNoiseDistribution, self).__init__(batch_shape, validate_args=validate_args)

        # Store current feature
        self.x = x
        self.mean_actions = mean_actions
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.exploration_mat = exploration_mat
        self.exploration_matrices = exploration_matrices
        self.deterministic = deterministic

        # Compute a new normal distribution with updated variance with noise
        variance = torch.mm(x ** 2, std ** 2)
        self._distribution = torch.distributions.Normal(mean_actions, torch.sqrt(variance + epsilon))

    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            return self.rsample(sample_shape)

    def rsample(self, sample_shape=torch.Size()):
        if self.deterministic:
            return self.mode()
        return self._sample()

    def mode(self):
        return self._distribution.mean

    def _sample(self):
        noise = self._get_noise(self.x)
        actions = self._distribution.mean + noise
        return actions

    def _get_noise(self, x) -> torch.Tensor:
        # Default case: only one exploration matrix
        if len(x) == 1 or len(x) != len(self.exploration_matrices):
            return torch.mm(x, self.exploration_mat)
        # Use batch matrix multiplication for efficient computation
        # (batch_size, n_features) -> (batch_size, 1, n_features)
        x = x.unsqueeze(1)
        # (batch_size, 1, n_actions)
        noise = torch.bmm(x, self.exploration_matrices)
        return noise.squeeze(1)

    def log_prob(self, actions) -> torch.Tensor:
        # log likelihood for a gaussian
        log_prob = self._distribution.log_prob(actions)
        # Sum along action dim
        log_prob = sum_independent_dims(log_prob)

        return log_prob
