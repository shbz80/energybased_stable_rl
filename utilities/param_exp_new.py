import numpy as np
import torch
import random
import copy
from energybased_stable_rl.policies.energy_based_control_policy import EnergyBasedPolicy
from energybased_stable_rl.policies.nf_deterministic_policy import DeterministicNormFlowPolicy

def sample_params(state_dict, stat_dict, std_scale, extr_std_weight, module_scale_dict=None):
    state_dict_in = state_dict
    state_dict_out = copy.deepcopy(state_dict_in)
    assert(std_scale>=0 and std_scale<=1.0)

    for param_key in stat_dict:
        mean = stat_dict[param_key]['mean']
        std = stat_dict[param_key]['std']

        if (module_scale_dict is not None) and (param_key in module_scale_dict.keys()):
            scale = module_scale_dict[param_key]
        else:
            scale = torch.ones_like(mean)
        # final_std = torch.sqrt(torch.square(scale * extr_std_weight) + (std_scale*std)**2) # todo
        final_std = scale * (extr_std_weight * (1.0-std_scale) + std_scale * std)
        assert (mean.shape == std.shape)
        # final_std = torch.ones_like(final_std)*0.2
        perturbed_param = torch.normal(mean, final_std)
        state_dict_out[param_key].copy_(perturbed_param)

    return state_dict_out


def cem_init_std(state_dict, stat_dict, init_std, init_log_std, policy):
    state_dict_keys = list(state_dict.keys())



    if isinstance(policy, EnergyBasedPolicy):
        key_strs = ['_icnn_module', '_damping_module', '._quad_pot_param']
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
        mean = torch.mean(param_array, axis=stack_dim)
        assert(curr_mean[param_key].shape == mean.shape)
        curr_mean[param_key].copy_(mean)
        curr_stat_dict[param_key]['mean'].copy_(mean)
        curr_stat_dict[param_key]['std'].copy_(torch.std(param_array, axis=stack_dim))


def scale_grad_params(grad_dict, sensitivity=False):
    global_max = 0
    scale_dict = copy.deepcopy(grad_dict)
    for key in grad_dict:
        param = grad_dict[key]
        max = torch.max(torch.abs(param))
        if max > global_max:
            global_max = max

    for key in grad_dict:
        if sensitivity:
            scale_dict[key] = 1.0 - torch.abs(grad_dict[key])/global_max
        else:
            scale_dict[key] = torch.ones_like(grad_dict[key])
    return scale_dict

def get_grad_theta(policy, obs0, sensitivity=False):
        grad_dict = {}
        key_strs = ['_icnn_module']

        with torch.enable_grad():

            for (key, param) in policy.named_parameters():
                for key_str in key_strs:
                    if key_str in key:
                        if sensitivity:
                            policy._module._icnn_module.zero_grad()
                            psi = policy._module._icnn_module(obs0)
                            grad = torch.autograd.grad(psi, param, None, retain_graph=False, create_graph=False)
                            grad_dict[key] = grad[0].detach()
                        else:
                            grad_dict[key] = torch.ones_like(param)
        return grad_dict