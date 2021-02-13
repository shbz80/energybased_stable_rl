import numpy as np
import torch
import random
import copy

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
                assert(param.shape==std.shape)
                perturbed_param = torch.normal(param, std)
                state_dict_out[key].copy_(perturbed_param)
    return state_dict_out

def cem_init_std(state_dict, std_dict, init_std, init_log_std):
    state_dict_keys = list(state_dict.keys())
    key_strs = ['_icnn_module', '_damping_module', '_quad_pot_param']
    log_normal_strs = ['_z_layers', 'log_weight']
    # key_strs = ['_mean_module']


    for key in state_dict_keys:
        for key_str in key_strs:
            if key_str in key:
                param = state_dict[key]
                if (log_normal_strs[0] in key)  and (log_normal_strs[1] in key):         # log normal todo bad implementation
                    std_dict[key] = torch.ones_like(param) * init_log_std
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




