import torch
from torch import nn
from energybased_stable_rl.policies.gaussian_ps_mlp_module import GaussianPSMLPModule

module = GaussianPSMLPModule(
            input_dim=4,
            output_dim=2,
            hidden_sizes=[16, 16],
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
            std_parameterization='exp',
            layer_normalization=False)

input = torch.rand(1,4)
output = module._get_mean_and_std(input)