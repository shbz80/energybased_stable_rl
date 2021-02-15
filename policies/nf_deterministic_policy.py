import torch
from torch import nn
from torch.distributions import Normal
from torch.distributions.independent import Independent

from garage.torch.distributions import TanhNormal

from energybased_stable_rl.policies.stochastic_nf_policy import StochasticNFPolicy

import normflow_policy.normflow_ds as nfds

from dowel import logger, tabular


class DeterministicNormFlowPolicy(StochasticNFPolicy):
    """MLP whose outputs are fed into a Normal distribution..

    A policy that contains a MLP to make prediction based on a gaussian
    distribution.

    Args:
        env_spec (garage.envs.env_spec.EnvSpec): Environment specification.
        dim (int): Dimension of configuration
        n_flows (int) : Number of flow layers
        K (number or array): Scale of stiffness matrix
        D (number of array): Scale of damping matrix
        normal_distribution_cls (torch.distribution): normal distribution class
        to be constructed and returned by a call to forward. By default, is
        `torch.distributions.Normal`.
        init_std (number): initial standard deviation parameter
        name (str): Name of policy.

    """

    def __init__(self,
                 env_spec,
                 dim=2, n_flows=3, hidden_dim=8, K=None, D=None,
                 normal_distribution_cls=Normal,
                 init_std=1.0,
                 jac_damping=False,
                 init_func = nn.init.xavier_uniform_,
                 init_const = None,
                 name='GaussianNormFlowPolicy'):
        super().__init__(env_spec, name)
        self._obs_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.flat_dim
        self._module = nfds.NormalizingFlowDynamicalSystem(
            dim=self._obs_dim // 2,  # suppress dim with obs_dim
            n_flows=n_flows,
            hidden_dim=hidden_dim,
            K=K,
            D=D,
            device='cpu')
        self._module.init_phi(init_func, init_const)

        self._normal_distribution_cls = normal_distribution_cls
        # this is probably slightly different from GaussianMLP that has only one param for variance
        init_std_param = torch.Tensor([init_std]).log()
        self._init_std = torch.nn.Parameter(init_std_param)
        self._jac_damping = jac_damping

    def forward(self, observations):
        """Compute the action distributions from the observations.

        Args:
            observations (torch.Tensor): Batch of observations on default
                torch device.

        Returns:
            torch.distributions.Distribution: Batch distribution of actions.
            dict[str, torch.Tensor]: Additional agent_info, as torch Tensors

        """
        # logger.log('Obervations shape: {0}, {1}'.format(observations.shape[0], observations.shape[1]))
        # first flatten observations because jacobian_in_batch can only handle one batch dimension
        # should we use view to avoid create new tensors?
        obs_flatten = torch.reshape(observations, (-1, self._obs_dim))
        # logger.log('Obervations flatten shape: {0}, {1}'.format(obs_flatten.shape[0], obs_flatten.shape[1]))
        # might need to figure out a way for more axes
        x = obs_flatten[:, :self._obs_dim // 2]
        x_star = torch.zeros_like(x)  # assuming zero as the converging state
        x_dot = obs_flatten[:, self._obs_dim // 2:]

        # be sure we have grad for Jacobian evaluation
        x.requires_grad_()
        x_dot.requires_grad_()

        with torch.enable_grad():
            mean_flatten = self._module.forward_with_damping(x=x, x_star=x_star, x_dot=x_dot, inv=False,
                                                             jac_damping=self._jac_damping)

        # restore mean shape
        broadcast_shape = list(observations.shape[:-1]) + [self._action_dim]
        mean = torch.reshape(mean_flatten, broadcast_shape)

        uncentered_log_std = torch.zeros(*broadcast_shape) + self._init_std

        std = uncentered_log_std.exp()

        dist = self._normal_distribution_cls(mean, std)

        if not isinstance(dist, TanhNormal):
            # Makes it so that a sample from the distribution is treated as a
            # single sample and not dist.batch_shape samples.
            dist = Independent(dist, 1)

        return (dist, dict(mean=dist.mean, log_std=(dist.variance ** .5).log()))

    def _get_action(self, observations):
        (dist, dict) = self.forward(observations)
        return dist.mean