import numpy as np
import torch
import random
import copy
from energybased_stable_rl.policies.energy_based_control_policy import GaussianEnergyBasedPolicy
from energybased_stable_rl.policies.gaussian_ps_mlp_policy import GaussianPSMLPPolicy
from energybased_stable_rl.policies.energy_control_modules import desired_lognormal_mean

# def perturb_layers(pre_str, state_dict_in, state_dict_out, std, strs, layers):
#     for i in range(layers):
#         for s in strs:
#             if pre_str+str(i)+s in state_dict_in:
#                 param = state_dict_in[pre_str+str(i)+s]
#                 perturbed_param = torch.normal(mean=param, std=torch.ones_like(param) * std)
#                 state_dict_out[pre_str + str(i) + s].copy_(perturbed_param)


def perturbTorchPolicy(policy, jac_batch_size=64):

    state_dict_in = policy.get_param_values()
    state_dict_out = copy.deepcopy(state_dict_in)

    state_dict_keys = list(state_dict_in.keys())

    std = policy._module.get_std().detach()
    state_dict_out['_module._init_std'].copy_(std)  # todo is this the right place for this?
    if isinstance(policy, GaussianPSMLPPolicy):
        key_strs = ['_mean_module']

    if isinstance(policy, GaussianEnergyBasedPolicy):
        key_strs = ['_icnn_module', '_damping_module', '_quad_pot_param']
        # key_strs = ['_icnn_module']

    param_keys_list = []
    selected_keys_list = []

    for str in state_dict_keys:
        for key_str in key_strs:
            if key_str in str:
                param_keys_list.append(str)

    rnd_idx = random.sample(list(range(0, len(param_keys_list))), k=len(param_keys_list))

    batch_size_count = 0
    while batch_size_count < jac_batch_size and len(rnd_idx)>0:
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

def perturbTorchPolicyBatch(policy, batch_size=1, jac_batch_size=64):
    policy_list = []

    for i in range(batch_size):
        policy_list.append(perturbTorchPolicy(policy, jac_batch_size=jac_batch_size))

    return policy_list

def sample_params(mean_state_dict, std_dict, epoch):
    state_dict_in = mean_state_dict
    state_dict_out = copy.deepcopy(state_dict_in)

    state_dict_keys = list(state_dict_in.keys())

    key_strs = ['_icnn_module', '_damping_module', '_quad_pot_param'] #todo
    # key_strs = ['_mean_module']

    for key in state_dict_keys:
        for key_str in key_strs:
            if key_str in key:
                param = state_dict_in[key]
                std = std_dict[key]
                perturbed_param = torch.normal(param, std)
                state_dict_out[key].copy_(perturbed_param)
    return state_dict_out

def cem_init_std(state_dict, init_std, std_dict):
    state_dict_keys = list(state_dict.keys())
    key_strs = ['_icnn_module', '_damping_module', '_quad_pot_param']
    log_normal_strs = ['_z_layers', 'p_list']
    # key_strs = ['_mean_module']


    for key in state_dict_keys:
        for key_str in key_strs:
            if key_str in key:
                param = state_dict[key]
                if (log_normal_strs[0] in key)  and (log_normal_strs[1] in key):         # log normal todo bad implementation
                    desired_mean = torch.ones_like(param)*desired_lognormal_mean
                    # desired_std = torch.ones_like(param) * init_std   # todo
                    desired_std = torch.ones_like(param) * 0.001
                    log_std = torch.sqrt(torch.log(1 + (desired_std ** 2) / (desired_mean ** 2)))
                    std_dict[key] = log_std
                else:
                    std_dict[key] = torch.ones_like(param) * init_std
    print('cem init')



def cem_stat_compute(best_params, curr_mean, curr_std):
    state_dict_keys = list(curr_mean.keys())

    for key in state_dict_keys:
        if (('_icnn_module' in key) or ('_damping_module' in key) or ('_quad_pot_param' in key)):
        # if ('_mean_module' in key):
            param_list = [params[key].unsqueeze(-1) for params in best_params]
            stack_dim = len(param_list[0].shape)-1
            param_array = torch.cat(param_list, axis=stack_dim)
            curr_mean[key] = torch.mean(param_array, axis=stack_dim)
            curr_std[key] = torch.std(param_array, axis=stack_dim)




