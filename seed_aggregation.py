import pickle
import numpy as np
from energybased_stable_rl.envs.block2D import T
import copy

base_filename = '/home/shahbaz/Software/garage36/energybased_stable_rl/data/local/experiment'

epoch_start = 0
epoch_num = 100
tm = range(T)
sample_num = 15

reward_mean_seed = np.zeros(epoch_num)
reward_std_seed = np.zeros(epoch_num)
success_mat_seed = np.zeros((epoch_num, sample_num))

def aggregrate_statistics(nll_list, n):
    nll_list = np.array(nll_list).T
    s0 = np.full((nll_list.shape[0]), n)
    mu = nll_list[:,0].reshape(-1)
    sigma = nll_list[:, 1].reshape(-1)
    sigma_sq = np.square(sigma)
    s1 = np.multiply(s0,mu)
    s2 = np.multiply(s0,(np.square(mu)+sigma_sq))
    S0 = np.sum(s0)
    S1 = np.sum(s1)
    S2 = np.sum(s2)
    Sigma_sq = np.divide(S2,S0) - np.divide(S1,S0)**2
    Sigma = np.sqrt(Sigma_sq)
    Mu = np.mean(mu)
    return Mu, Sigma

def aggregate_results(exps, file_name, success_dit, flag=False):
    if not isinstance(file_name, str):
        raise ValueError('Enter a string name for the output file')
    reward_means = []
    reward_stds = []
    success_stats = []
    results = {}
    for i in range(len(exps)):
        for ep in range(epoch_num):
            filename = base_filename + '/' + exps[i] + '/' + 'itr_' + str(ep) + '.pkl'
            infile = open(filename, 'rb')
            ep_data = pickle.load(infile)
            infile.close()
            epoch = ep_data['stats'].last_episode
            sample_num = 15
            if ep==8 and flag:
                sample_num = 14
            reward_mean_seed[ep] = np.mean([np.sum(epoch[s]['rewards']) for s in range(sample_num)])
            reward_std_seed[ep] = np.std([np.sum(epoch[s]['rewards']) for s in range(sample_num)])
            if ep==8 and flag:
                sample_num = 15
            for s in range(sample_num):
                sample = np.min(np.linalg.norm(epoch[s]['observations'][:, :2], axis=1), axis=0)
                success_mat_seed[ep, s] = sample < success_dit
        reward_means.append(copy.copy(reward_mean_seed))
        reward_stds.append(copy.copy(reward_std_seed))

        success_rate = np.sum(success_mat_seed, axis=1)*(100/sample_num)
        success_stats.append(copy.copy(success_rate))

    # reward stat
    reward_means = np.array(reward_means)
    reward_stds = np.array(reward_stds)
    rewards_mu_agg = np.zeros((epoch_num))
    rewards_std_agg = np.zeros((epoch_num))
    assert reward_means.shape == reward_stds.shape
    assert reward_stds.shape[1] == epoch_num
    for i in range(reward_stds.shape[1]):
        stat = [reward_means[:,i], reward_stds[:,i]]
        rewards_mu_agg[i], rewards_std_agg[i] = aggregrate_statistics(stat, sample_num)

    reward_stat = (rewards_mu_agg, rewards_std_agg)
    results['reward_stat'] = reward_stat


    # success stat
    success_stat = np.array(success_stats)
    success_mu = np.mean(success_stat, axis=0)
    success_std = np.std(success_stat, axis=0)
    success_stat = (success_mu, success_std)
    results['success_stat'] = success_stat

    pickle.dump(results, open(file_name, "wb"))