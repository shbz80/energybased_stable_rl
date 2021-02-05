
import torch
import random
import copy
from energybased_stable_rl.policies.energy_based_control_policy import GaussianEnergyBasedPolicy
from energybased_stable_rl.policies.gaussian_ps_mlp_policy import GaussianPSMLPPolicy

# def perturb_layers(pre_str, state_dict_in, state_dict_out, std, strs, layers):
#     for i in range(layers):
#         for s in strs:
#             if pre_str+str(i)+s in state_dict_in:
#                 param = state_dict_in[pre_str+str(i)+s]
#                 perturbed_param = torch.normal(mean=param, std=torch.ones_like(param) * std)
#                 state_dict_out[pre_str + str(i) + s].copy_(perturbed_param)


def perturbTorchPolicy(policy, std, jac_batch_size=64):
    state_dict_in = policy.get_param_values()
    state_dict_out = copy.deepcopy(state_dict_in)

    state_dict_keys = list(state_dict_in.keys())

    if isinstance(policy, GaussianPSMLPPolicy):
        key_str = '_mean_module'

    param_keys_list = []
    selected_keys_list = []

    for str in state_dict_keys:
        if key_str in str:
            param_keys_list.append(str)

    rnd_idx = random.sample(list(range(0, len(param_keys_list))), k=len(param_keys_list))

    batch_size_count = 0
    while batch_size_count < jac_batch_size:
        param_key = param_keys_list[rnd_idx.pop()]
        param = state_dict_in[param_key]
        batch_size_count = batch_size_count + len(param.view(-1))
        selected_keys_list.append(param_key)
        perturbed_param = torch.normal(mean=param, std=torch.ones_like(param) * std)
        state_dict_out[param_key].copy_(perturbed_param)

    return (state_dict_out, selected_keys_list)




    #     hidden_sizes = module._hidden_sizes
    #     pre_str = '_module._mean_module._layers.'
    #     strs = ['.linear.weight', '.linear.bias']
    #     perturb_layers(pre_str, state_dict_in, state_dict_out, std, strs, len(hidden_sizes))
    #
    #     pre_str = '_module._mean_module._output_layers.'
    #     perturb_layers(pre_str, state_dict_in, state_dict_out, std, strs, 2)
    #
    # if isinstance(policy, GaussianEnergyBasedPolicy):
    #     hidden_sizes = module._icnn_hidden_sizes
    #     pre_str = '_module._icnn_module._y_layers.'
    #     strs = ['.weight', '.bias']
    #     perturb_layers(pre_str, state_dict_in, state_dict_out, std, strs, len(hidden_sizes)+1)
    #
    #     pre_str = '_module._icnn_module._z_layers.'
    #     strs = ['.log_weight']
    #     perturb_layers(pre_str, state_dict_in, state_dict_out, std, strs, len(hidden_sizes))
    #
    #     hidden_sizes = module._damper_hidden_sizes
    #     pre_str = '_module._damping_module._layers.'
    #     strs = ['.linear.weight', '.linear.bias']
    #     perturb_layers(pre_str, state_dict_in, state_dict_out, std, strs, len(hidden_sizes))
    #
    #     pre_str = '_module._damping_module._output_layer.'
    #     strs = ['.weight', '.bias']
    #     perturb_layers(pre_str, state_dict_in, state_dict_out, std, strs, 1)
    #
    # return state_dict_out

def perturbTorchPolicyBatch(policy, std, batch_size=1, jac_batch_size=64):
    policy_list = []

    for i in range(batch_size):
        policy_list.append(perturbTorchPolicy(policy, std, jac_batch_size=jac_batch_size))

    return policy_list