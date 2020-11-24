from numbers import Number

import torch
from torch.distributions import constraints, Distribution


def sum_independent_dims(tensor: th.Tensor) -> th.Tensor:
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


# TODO: Delete self.loc
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

    @property
    def mean(self):
        return self.distribution.mean

    @property
    def stddev(self):
        return torch.zeros_like(self.loc)

    @property
    def mode(self):
        return self.distribution.mean

    @property
    def variance(self):
        return torch.zeros_like(self.loc)

    def __init__(self, loc, validate_args=None):
        self.loc = loc
        if isinstance(loc, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super(StateDependentNoiseDistribution, self).__init__(batch_shape, validate_args=validate_args)

    def get_std(self, log_std):
        """
        Get the standard deviation from the learned parameter
        (log of it by default). This ensures that the std is positive.
        :param log_std:
        :return:
        """
        # TODO: Register use_expln to self.
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
        return torch.ones(self.latent_sde_dim, self.action_dim).to(log_std.device) * std

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(StateDependentNoiseDistribution, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        super(StateDependentNoiseDistribution, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def sample(self, sample_shape=torch.Size()):
        # TODO: This needs to be called after rsample ()
        noise = self.get_noise(self._latent_sde)
        actions = self.distribution.mean + noise
        return actions

    def reset_noise(self, log_std: th.Tensor, batch_size: int = 1) -> None:
        """
        Sample weights for the noise exploration matrix,
        using a centered Gaussian distribution.
        :param log_std:
        :param batch_size:
        """
        std = self.get_std(log_std)
        self.weights_dist = torch.distributions.Normal(torch.zeros_like(std), std)
        # Reparametrization trick to pass gradients
        self.exploration_mat = self.weights_dist.rsample()
        # Pre-compute matrices in case of parallel exploration
        self.exploration_matrices = self.weights_dist.rsample((batch_size,))

    def get_noise(self, latent_sde):
        # TODO: Register exploration_matrices and exploration_mat to self.

        # Default case: only one exploration matrix
        if len(latent_sde) == 1 or len(latent_sde) != len(self.exploration_matrices):
            return torch.mm(latent_sde, self.exploration_mat)
        # Use batch matrix multiplication for efficient computation
        # (batch_size, n_features) -> (batch_size, 1, n_features)
        latent_sde = latent_sde.unsqueeze(1)
        # (batch_size, 1, n_actions)
        noise = torch.bmm(latent_sde, self.exploration_matrices)
        return noise.squeeze(1)

    def rsample(self, sample_shape=torch.Size()):
        # Get these arguments someehow
        # 1. mean_actions
        # 2. log_std
        # 3. latent_sde

        # First, recompute a probability distribution
        variance = torch.mm(self.latent_sde ** 2, self.get_std(log_std) ** 2)
        self.distribution = torch.distributions.Normal(mean_actions, torch.sqrt(variance + self.epsilon))

        if self.deterministic:
            return self.mode()

        shape = self._extended_shape(sample_shape)
        return self.loc.expand(shape)

    def log_prob(self, actions):
        # log likelihood for a gaussian
        log_prob = self.distribution.log_prob(actions)
        # Sum along action dim
        log_prob = sum_independent_dims(log_prob)

        return log_prob

    def entropy(self):
        raise RuntimeError("Not defined")
