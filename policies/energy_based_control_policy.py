"""GaussianMLPPolicy."""
import torch
from torch import nn

from energybased_stable_rl.policies.energy_based_control_module import EnergyBasedControlModule
from energybased_stable_rl.policies.stochstic_eb_policy import StochasticEBPolicy

class EnergyBasedPolicy(StochasticEBPolicy):
    """MLP whose parameters are distributed according to a Normal distribution..

    A policy that contains a MLP to make prediction based on a gaussian
    distribution.

    Args:


    """

    def __init__(self,
                 env_spec,
                 icnn_hidden_sizes=(32, 32),
                 w_init_icnn_y=nn.init.xavier_uniform_,
                 b_init_icnn_y=nn.init.zeros_,
                 w_init_icnn_y_param=0.1,
                 w_init_icnn_z=nn.init.constant_,
                 w_init_icnn_z_param=0.1,
                 icnn_bias = False,
                 positive_type='log',
                 nonlinearity_icnn=torch.relu,
                 damper_hidden_sizes=(32, 32),
                 w_init_damper_offdiag=nn.init.xavier_uniform_,
                 b_init_damper_offdiag=nn.init.zeros_,
                 w_init_damper_diag=nn.init.xavier_uniform_,
                 b_init_damper_diag=nn.init.zeros_,
                 hidden_nonlinearity_damper=torch.tanh,
                 full_mat_damper=True,
                 damp_min=None,
                 init_quad_pot=1.0,
                 min_quad_pot=1e-3,
                 max_quad_pot=1e1,
                 icnn_min_lr=1e-1,
                 name='EBPPolicy'):
        super().__init__(env_spec, name)
        self._obs_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.flat_dim
        self._module = EnergyBasedControlModule(
            coord_dim = self._obs_dim//2,
            icnn_hidden_sizes=icnn_hidden_sizes,
            w_init_icnn_y=w_init_icnn_y,
            b_init_icnn_y=b_init_icnn_y,
            w_init_icnn_y_param=w_init_icnn_y_param,
            w_init_icnn_z=w_init_icnn_z,
            w_init_icnn_z_param=w_init_icnn_z_param,
            icnn_bias = icnn_bias,
            positive_type=positive_type,
            nonlinearity_icnn=nonlinearity_icnn,
            damper_hidden_sizes=damper_hidden_sizes,
            w_init_damper_offdiag=w_init_damper_offdiag,
            b_init_damper_offdiag=b_init_damper_offdiag,
            w_init_damper_diag=w_init_damper_diag,
            b_init_damper_diag=b_init_damper_diag,
            hidden_nonlinearity_damper=hidden_nonlinearity_damper,
            full_mat_damper=full_mat_damper,
            damp_min=damp_min,
            init_quad_pot=init_quad_pot,
            min_quad_pot=min_quad_pot,
            max_quad_pot=max_quad_pot,
            icnn_min_lr=icnn_min_lr,
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

    def set_param_values(self, state_dict):
        """Set the parameters to the policy.

        This method is included to ensure consistency with TF policies.

        Args:
            state_dict (dict): State dictionary.

        """
        if isinstance(state_dict, dict):
            self.load_state_dict(state_dict)
        else:
            print('policy param not dict')
            AssertionError



