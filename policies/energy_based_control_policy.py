"""GaussianMLPPolicy."""
import torch
from torch import nn

from energybased_stable_rl.policies.energy_based_control_module import GaussianEnergyBasedModule
from energybased_stable_rl.policies.stochstic_ps_policy import StochasticPSPolicy
from torch.distributions import Normal, MultivariateNormal

class GaussianEnergyBasedPolicy(StochasticPSPolicy):
    """MLP whose parameters are distributed according to a Normal distribution..

    A policy that contains a MLP to make prediction based on a gaussian
    distribution.

    Args:


    """

    def __init__(self,
                 env_spec,
                 icnn_hidden_sizes=(32, 32),
                 damper_hidden_sizes=(32, 32),
                 w_init=nn.init.xavier_uniform_,
                 b_init=nn.init.zeros_,
                 icnn_hidden_nonlinearity=torch.relu,  # this has to a convex function e.g. relu
                 damper_hidden_nonlinearity=torch.tanh,
                 damper_full_mat=True,
                 learn_std=True,
                 init_std=1.0,
                 min_std=1e-6,
                 max_std=None,
                 full_std=False,
                 jac_update_rate=10,
                 std_parameterization='exp',
                 init_quad_pot=1.0,
                 min_quad_pot=1e-3,
                 max_quad_pot=1e1,
                 icnn_min_lr=1e-1,
                 name='GaussianMLPPolicy'):
        super().__init__(env_spec, name)
        self._obs_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.flat_dim
        self._module = GaussianEnergyBasedModule(
            coord_dim = self._obs_dim//2,
            icnn_hidden_sizes=icnn_hidden_sizes,
            damper_hidden_sizes=damper_hidden_sizes,
            w_init=w_init,
            b_init=b_init,
            icnn_hidden_nonlinearity=icnn_hidden_nonlinearity,  # this has to a convex function e.g. relu
            damper_hidden_nonlinearity=damper_hidden_nonlinearity,
            damper_full_mat=damper_full_mat,
            learn_std=learn_std,
            init_std=init_std,
            min_std=min_std,
            max_std=max_std,
            full_std=full_std,
            jac_update_rate=jac_update_rate,
            std_parameterization=std_parameterization,
            init_quad_pot=init_quad_pot,
            min_quad_pot=min_quad_pot,
            max_quad_pot=max_quad_pot,
            icnn_min_lr=icnn_min_lr)

    def forward(self, observations):
        """Compute the action distributions from the observations.

        Args:
            observations (torch.Tensor): Batch of observations on default
                torch device.

        Returns:
            torch.distributions.Distribution: Batch distribution of actions.
            dict[str, torch.Tensor]: Additional agent_info, as torch Tensors

        """
        dist = self._module(observations)
        return (dist, dict(mean=dist.mean, log_std=(dist.variance**.5).log()))

    def reset(self, do_resets=None):
        self._module._min_icnn()

