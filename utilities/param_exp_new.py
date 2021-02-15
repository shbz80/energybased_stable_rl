import numpy as np
import torch
import random
import copy
from energybased_stable_rl.policies.energy_based_control_policy import EnergyBasedPolicy
from energybased_stable_rl.policies.nf_deterministic_policy import DeterministicNormFlowPolicy

def sample_params(state_dict, stat_dict):
    state_dict_in = state_dict
    state_dict_out = copy.deepcopy(state_dict_in)

    for param_key in stat_dict:
        mean = stat_dict[param_key]['mean']
        std = stat_dict[param_key]['std']
        assert (mean.shape == std.shape)
        perturbed_param = torch.normal(mean, std)
        state_dict_out[param_key].copy_(perturbed_param)

    return state_dict_out

def cem_init_std(state_dict, stat_dict, init_std, init_log_std, policy):
    state_dict_keys = list(state_dict.keys())

    if isinstance(policy, EnergyBasedPolicy):
        key_strs = ['_icnn_module', '_damping_module', '_quad_pot_param']
        log_normal_strs = ['_z_layers', 'log_weight']
        # key_strs = ['_mean_module']

        for key in state_dict_keys:
            for key_str in key_strs:
                if key_str in key:
                    param = state_dict[key]
                    if (log_normal_strs[0] in key)  and (log_normal_strs[1] in key):         # log normal todo bad implementation
                        param_std = torch.ones_like(param) * init_log_std
                        stat_dict[key] = {'mean': param, 'std': param_std}
                    else:
                        param_std = torch.ones_like(param) * init_std
                        stat_dict[key] = {'mean': param, 'std': param_std}

    if isinstance(policy, DeterministicNormFlowPolicy):
        key_strs = ['phi']
        for key in state_dict_keys:
            for key_str in key_strs:
                if key_str in key:
                    param = state_dict[key]
                    param_std  = torch.ones_like(param) * init_std
                    stat_dict[key] = {'mean': param, 'std': param_std}
    print('cem init')



def cem_stat_compute(best_params, curr_mean, curr_stat_dict):

    for param_key in curr_stat_dict:
        param_list = [params[param_key].unsqueeze(-1) for params in best_params]
        stack_dim = len(param_list[0].shape) - 1
        param_array = torch.cat(param_list, axis=stack_dim)
        curr_mean[param_key] = torch.mean(param_array, axis=stack_dim)
        curr_stat_dict[param_key]['mean'] = curr_mean[param_key]
        curr_stat_dict[param_key]['std'] = torch.std(param_array, axis=stack_dim)
