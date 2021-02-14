"""GaussianMLPPolicy."""
import torch

from energybased_stable_rl.policies.stochstic_eb_policy import StochasticEBPolicy
from energybased_stable_rl.policies.energy_based_init_module import EnergyBasedInitModule

class EnergyBasedInitPolicy(StochasticEBPolicy):
    """MLP whose parameters are distributed according to a Normal distribution..

    A policy that contains a MLP to make prediction based on a gaussian
    distribution.

    Args:


    """

    def __init__(self,
                 env_spec,
                 S_param,
                 D_param,
                 std,
                 name='EBInitPPolicy'
                 ):
        super().__init__(env_spec, name)
        self._obs_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.flat_dim
        self._module = EnergyBasedInitModule(
            S_param,
            D_param,
            std,
            self._obs_dim,
            self._action_dim
        )

    def forward(self, observations):
        """Compute the action distributions from the observations.

        Args:
            observations (torch.Tensor): Batch of observations on default
                torch device.

        Returns:
            torch.distributions.Distribution: Batch distribution of actions.
            dict[str, torch.Tensor]: Additional agent_info, as torch Tensors

        """

    def reset(self, do_resets=None):
        return
        # self._module.min_icnn()





