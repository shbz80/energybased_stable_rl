
import torch
import copy
from energybased_stable_rl.policies.energy_based_control_policy import GaussianEnergyBasedPolicy
from energybased_stable_rl.policies.gaussian_ps_mlp_policy import GaussianPSMLPPolicy

def perturb_layers(pre_str, state_dict_in, state_dict_out, std, strs, layers):
    for i in range(layers):
        for s in strs:
            if pre_str+str(i)+s in state_dict_in:
                param = state_dict_in[pre_str+str(i)+s]
                perturbed_param = torch.normal(mean=param, std=torch.ones_like(param) * std)
                state_dict_out[pre_str + str(i) + s].copy_(perturbed_param)


def perturbTorchPolicy(policy, std):
    state_dict_in = policy.get_param_values()
    state_dict_out = copy.deepcopy(state_dict_in)

    assert(policy._module)
    module = policy._module

    if isinstance(policy, GaussianPSMLPPolicy):
        hidden_sizes = module._hidden_sizes
        pre_str = '_module._mean_module._layers.'
        strs = ['.linear.weight', '.linear.bias']
        perturb_layers(pre_str, state_dict_in, state_dict_out, std, strs, len(hidden_sizes))

        pre_str = '_module._mean_module._output_layers.'
        perturb_layers(pre_str, state_dict_in, state_dict_out, std, strs, 2)

    if isinstance(policy, GaussianEnergyBasedPolicy):
        hidden_sizes = module._icnn_hidden_sizes
        pre_str = '_module._icnn_module._y_layers.'
        strs = ['.weight', '.bias']
        perturb_layers(pre_str, state_dict_in, state_dict_out, std, strs, len(hidden_sizes)+1)

        pre_str = '_module._icnn_module._z_layers.'
        strs = ['.log_weight']
        perturb_layers(pre_str, state_dict_in, state_dict_out, std, strs, len(hidden_sizes))

        hidden_sizes = module._damper_hidden_sizes
        pre_str = '_module._damping_module._layers.'
        strs = ['.linear.weight', '.linear.bias']
        perturb_layers(pre_str, state_dict_in, state_dict_out, std, strs, len(hidden_sizes))

        pre_str = '_module._damping_module._output_layer.'
        strs = ['.weight', '.bias']
        perturb_layers(pre_str, state_dict_in, state_dict_out, std, strs, 1)

    return state_dict_out

def perturbTorchPolicyBatch(policy, std, batch_size=1):
    policy_list = []

    for i in range(batch_size):
        policy_list.append(perturbTorchPolicy(policy, std))

    return policy_list