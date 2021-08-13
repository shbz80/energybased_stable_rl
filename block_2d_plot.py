import pickle
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import torch
from energybased_stable_rl.envs.block2D import T


base_filename = '/home/shahbaz/Software/garage36/energybased_stable_rl/data/local/experiment'
# base_filename = '/media/shahbaz/My Passport/nfdata/data/local/experiment/'
# baseline
exp_name = 'block2D_ppo_torch_garage_1'
# exp_name = 'block2D_energy_based_ppo_3'
# exp_name = 'block_2d_sac_torch'
# exp_name = 'block2D_psppo_torch_garage_2'

# exp_name = 'cem_nf_block2d_2'
# exp_name = 'block2d_nfppo_garage_e_3'
# exp_name = 'elitebook/cem_energybased_block2d'
# exp_name = 'elitebook/cem_energybased_block2d_1'
# exp_name = 'cem_energybased_block2d_24'
SUCCESS_DIST = 0.025
plot_skip = 5
plot_traj = False
traj_skip = 1
# GOAL = block2D.GOAL
epoch_start = 0
epoch_num = 100
tm = range(T)
sample_num = 15
plot_energy = False

for ep in range(epoch_start,epoch_num):
    if ((ep==0) or (not ((ep+1) % plot_skip))) and plot_traj:
        filename = base_filename + '/' + exp_name + '/' + 'itr_' + str(ep) + '.pkl'
        infile = open(filename, 'rb')
        ep_data = pickle.load(infile)
        infile.close()
        epoch = ep_data['stats'].last_episode
        obs0 = epoch[0]['observations']
        act0 = epoch[0]['actions']
        rwd_s0 = epoch[0]['env_infos']['reward_dist']
        rwd_a0 = epoch[0]['env_infos']['reward_ctrl']
        pos = obs0[:,:2].reshape(T,1,2)
        vel = obs0[:,2:4].reshape(T,1,2)
        act = act0[:,:2].reshape(T,1,2)
        rwd_s = rwd_s0.reshape(T,1)
        rwd_a = rwd_a0.reshape(T,1)


        cum_rwd_s_epoch = 0
        cum_rwd_a_epoch = 0
        # cum_rwd_t_epoch = 0
        for sp in range(0,sample_num):
            if ((sp == 0) or (not ((sp + 1) % traj_skip))):
                sample = epoch[sp]
                p = sample['observations'][:,:2].reshape(T,1,2)
                v = sample['observations'][:, 2:4].reshape(T, 1, 2)
                a = sample['actions'][:,:2].reshape(T, 1, 2)
                rs = sample['env_infos']['reward_dist'].reshape(T, 1)
                cum_rwd_s_epoch = cum_rwd_s_epoch + np.sum(rs.reshape(-1))
                ra = sample['env_infos']['reward_ctrl'].reshape(T, 1)
                cum_rwd_a_epoch = cum_rwd_a_epoch + np.sum(ra.reshape(-1))
                pos = np.concatenate((pos,p), axis=1)
                vel = np.concatenate((vel, v), axis=1)
                act = np.concatenate((act, a), axis=1)
                rwd_s = np.concatenate((rwd_s, rs), axis=1)
                rwd_a = np.concatenate((rwd_a, ra), axis=1)

        fig = plt.figure()
        plt.title('Epoch '+str(ep))
        plt.axis('off')
        ax = fig.add_subplot(3, 4, 1)
        ax.set_title('s1')
        ax.plot(tm, pos[:, :, 0], color='g')
        ax = fig.add_subplot(3, 4, 2)
        ax.set_title('s2')
        ax.plot(tm, pos[:, :, 1], color='g')
        ax = fig.add_subplot(3, 4, 3)
        ax.set_title('sdot1')
        ax.plot(tm, vel[:, :, 0], color='b')
        ax = fig.add_subplot(3, 4, 4)
        ax.set_title('sdot2')
        ax.plot(tm, vel[:, :, 1], color='b')
        ax = fig.add_subplot(3, 4, 5)
        ax.set_title('a1')
        ax.plot(tm, act[:, :, 0], color='r')
        ax = fig.add_subplot(3, 4, 6)
        ax.set_title('a2')
        ax.plot(tm, act[:, :, 1], color='r')
        ax = fig.add_subplot(3, 4, 7)
        ax.set_title('rs')
        ax.plot(tm, rwd_s, color='m')
        ax = fig.add_subplot(3, 4, 8)
        ax.set_title('ra')
        ax.plot(tm, rwd_a, color='c')

# rewards_disc_rtn = np.zeros(epoch_num)
rewards_undisc_mean = np.zeros(epoch_num)
rewards_undisc_std = np.zeros(epoch_num)
success_mat = np.zeros((epoch_num, sample_num))
state_dist_all = np.zeros((epoch_num, sample_num, T))
state_dist_last = np.zeros((epoch_num,sample_num))

for ep in range(epoch_start, epoch_num):
    filename = base_filename + '/' + exp_name + '/' + 'itr_' + str(ep) + '.pkl'
    infile = open(filename, 'rb')
    ep_data = pickle.load(infile)
    infile.close()
    epoch = ep_data['stats'].last_episode
    # rewards_disc_rtn[ep] = np.mean([epoch[s]['returns'][0] for s in range(sample_num)])
    rewards_undisc_mean[ep] = np.mean([np.sum(epoch[s]['rewards']) for s in range(sample_num)])
    rewards_undisc_std[ep] = np.std([np.sum(epoch[s]['rewards']) for s in range(sample_num)])
    for s in range(sample_num):
        pos_norm = np.linalg.norm(epoch[s]['observations'][:, :2], axis=1)
        success_mat[ep, s] = np.min(pos_norm)<SUCCESS_DIST
        state_dist_all[ep][s] = np.linalg.norm(epoch[s]['observations'][:, :2], axis=1).reshape(-1)
        state_dist_last[ep][s] = np.linalg.norm(epoch[s]['observations'][:, :2], axis=1)[-1]

itr_state_dist = 10
state_dist_all = state_dist_all[:itr_state_dist,:,:].reshape(-1)
state_dist_last = state_dist_last[:itr_state_dist,:].reshape(-1)

# fig1 = plt.figure()
# ax1 = fig1.add_subplot(1, 1, 1)
# # ax1.set_title(r'\textbf{(a)}')
# # ax1.set_xlabel(r'Iteration')
# # ax1.set_ylabel(r'Reward')
# # ax1.set_xticks([0.0,0.4,0.8])
# # ax1.set_ylim(-1.7e4,-0.2e4)
# # data = [dist_ours, dist_vices, last_dist_ours, last_dist_vices]
# data = [state_dist_all, state_dist_last]
# # ax1.boxplot(data, showfliers=False, whis=(0,100),vert=False)
# # bp = ax1.boxplot(data, patch_artist = False, showfliers=False, whis=(0,100),vert=False)
# bp = ax1.boxplot(data, patch_artist = False, showfliers=False, vert=False)
# for median in bp['medians']:
#     median.set(color ='blue',
#                linewidth = 1)
# ax1.set_yticklabels(['All pos','Final pos'])
# ax1.set_xlabel('m',labelpad=1)
# # plt.subplots_adjust(left=0.429, bottom=0.2, right=.99, top=0.98, wspace=0.5, hspace=0.7)

success_stat = np.sum(success_mat, axis=1)*(100/sample_num)

fig = plt.figure()
plt.axis('off')
ax = fig.add_subplot(1, 2, 1)
ax.set_title('Progress')
ax.set_xlabel('Epoch')
ax.plot(rewards_undisc_mean, label='undisc. reward')
ax.legend()
ax = fig.add_subplot(1, 2, 2)
ax.set_ylabel('Succes rate')
ax.set_xlabel('Epoch')
ax.plot(success_stat)
ax.legend()

if plot_energy:
    for ep in range(0,epoch_num):
        if ((ep==0) or (not ((ep+1) % plot_skip))) and plot_traj:
            filename = base_filename + '/' + exp_name + '/' + 'itr_' + str(ep) + '.pkl'
            infile = open(filename, 'rb')
            ep_data = pickle.load(infile)
            infile.close()
            phi = ep_data['algo'].policy._module.phi
            sample_num = len(epoch)
            epoch = ep_data['stats'].last_episode
            obs0 = epoch[0]['observations']
            T = obs0.shape[0]
            tm = range(T)
            pos = obs0[:, :2].astype('float32')
            pos = torch.from_numpy(pos)
            with torch.no_grad():
                eng_0 = phi(pos)
            eng_0 = eng_0.numpy()
            eng = eng_0.reshape(T,1)

            for sp in range(0,sample_num):
                if ((sp == 0) or (not ((sp + 1) % traj_skip))):
                    sample = epoch[sp]
                    p = sample['observations'][:,:2].reshape(T,1,2).astype('float32')
                    p = torch.from_numpy(p)
                    with torch.no_grad():
                        e = phi(p)
                    eng = np.concatenate((eng, e), axis=1)

            fig = plt.figure()
            plt.title('Epoch '+str(ep))
            plt.axis('off')
            ax = fig.add_subplot(1, 1, 1)
            ax.set_title('Phi')
            ax.plot(tm, eng, color='g')

plt.show()

