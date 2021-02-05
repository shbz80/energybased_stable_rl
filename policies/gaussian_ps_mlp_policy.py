"""GaussianMLPPolicy."""
import torch
from torch import nn

from energybased_stable_rl.policies.gaussian_ps_mlp_module import GaussianPSMLPModule
from energybased_stable_rl.policies.stochstic_ps_policy import StochasticPSPolicy
from torch.distributions import Normal, MultivariateNormal

class GaussianPSMLPPolicy(StochasticPSPolicy):
    """MLP whose parameters are distributed according to a Normal distribution..

    A policy that contains a MLP to make prediction based on a gaussian
    distribution.

    Args:
        env_spec (EnvSpec): Environment specification.
        hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for mean. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        hidden_nonlinearity (callable): Activation function for intermediate
            dense layer(s). It should return a torch.Tensor. Set it to
            None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        output_nonlinearity (callable): Activation function for output dense
            layer. It should return a torch.Tensor. Set it to None to
            maintain a linear activation.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            torch.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            torch.Tensor.
        learn_std (bool): Is std trainable.
        init_std (float): Initial value for std.
            (plain value - not log or exponentiated).
        min_std (float): Minimum value for std.
        max_std (float): Maximum value for std.
        std_parameterization (str): How the std should be parametrized. There
            are two options:
            - exp: the logarithm of the std will be stored, and applied a
               exponential transformation
            - softplus: the std will be computed as log(1+exp(x))
        layer_normalization (bool): Bool for using layer normalization or not.
        name (str): Name of policy.

    """

    def __init__(self,
                 env_spec,
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=torch.tanh,
                 hidden_w_init=nn.init.xavier_uniform_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearity=None,
                 output_w_init=nn.init.xavier_uniform_,
                 output_b_init=nn.init.zeros_,
                 learn_std=True,
                 init_std=1.0,
                 min_std=1e-6,
                 max_std=None,
                 full_std = False,
                 std_parameterization='exp',
                 layer_normalization=False,
                 jac_update_rate = 10,
                 jac_batch_size = 64,
                 name='GaussianMLPPolicy'):
        super().__init__(env_spec, name)
        self._obs_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.flat_dim
        self._module = GaussianPSMLPModule(
            input_dim=self._obs_dim,
            output_dim=self._action_dim,
            hidden_sizes=hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            hidden_w_init=hidden_w_init,
            hidden_b_init=hidden_b_init,
            output_nonlinearity=output_nonlinearity,
            output_w_init=output_w_init,
            output_b_init=output_b_init,
            learn_std=learn_std,
            init_std=init_std,
            min_std=min_std,
            max_std=max_std,
            full_std=full_std,
            jac_update_rate = jac_update_rate,
            jac_batch_size = jac_batch_size,
            std_parameterization=std_parameterization,
            layer_normalization=layer_normalization)
        self.selected_param_key = []

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

    def set_param_values(self, state_dict):
        """Set the parameters to the policy.

        This method is included to ensure consistency with TF policies.

        Args:
            state_dict (dict): State dictionary.

        """
        if isinstance(state_dict, dict):
            self.load_state_dict(state_dict)

        if isinstance(state_dict, tuple):
            assert(isinstance(state_dict[0], dict))
            self.load_state_dict(state_dict[0])
            self.selected_param_key = state_dict[1]

    def set_param_keys(self, param_keys):
        self._module.param_keys = param_keys
        named_params = dict([param for param in self.named_parameters()])
        self._module.param_values = [[named_params[p_key] for p_key in param_key] for param_key in param_keys]

