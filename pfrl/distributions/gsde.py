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

    arg_constraints = {"distribution": constraints.real}
    # mypy complains about the type of `support` since it is initialized
    # as None in `torch.distributions.Distribution` as of torch==1.5.0.
    support = constraints.real  # type: ignore
    has_rsample = True

    def __init__(self,
        x,
        in_dim,
        out_dim,
        full_std=True,
        use_expln=False,
        epsilon: float = 1e-6,
        validate_args=None):
        batch_shape = torch.Size()
        super(StateDependentNoiseDistribution, self).__init__(batch_shape, validate_args=validate_args)

        # Store current feature
        self.x = x

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.full_std = full_std
        self.use_expln = use_expln
        self.epsilon = epsilon

        # Compute a new normal distribution with updated variance with noise
        variance = torch.mm(x ** 2, self._std(log_std) ** 2)
        self._distribution = torch.distributions.Normal(mean_actions, torch.sqrt(variance + self.epsilon))

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
