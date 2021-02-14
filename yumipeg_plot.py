import pickle
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import torch
from energybased_stable_rl.envs.yumipegcart import T

base_filename = '/home/shahbaz/Software/garage36/energybased_stable_rl/data/local/experiment'
exp_name = 'cem_energybased_yumi_47'

SUCCESS_DIST = 0.004
plot_skip = 1
plot_traj = True
traj_skip = 1
epoch_start = 0
epoch_num = 2
T = 200
tm = range(T)
plot_energy = False
sample_num = 15

for i in range(epoch_start,epoch_num):
    if ((i==0) or (not ((i+1) % plot_skip))) and plot_traj:
        filename = base_filename + '/' + exp_name + '/' + 'itr_' + str(i) + '.pkl'
        infile = open(filename, 'rb')
        ep_data = pickle.load(infile)
        infile.close()
        epoch = ep_data['stats'].last_episode
        # sample_num = len(epoch)
        ep = epoch[0]['observations'][:,:3].reshape(T,1,3)
        ep = np.concatenate((ep, epoch[0]['env_infos']['er'].reshape(T, 1, 3)),axis=2)
        ev = epoch[0]['observations'][:, 3:].reshape(T, 1, 3)
        ev = np.concatenate((ev, epoch[0]['env_infos']['erdot'].reshape(T, 1, 3)), axis=2)
        jp = epoch[0]['env_infos']['jx'][:,:7].reshape(T,1,7)
        jv = epoch[0]['env_infos']['jx'][:, 7:].reshape(T, 1, 7)
        ef = epoch[0]['actions'].reshape(T, 1, 3)
        ef = np.concatenate((ef, epoch[0]['env_infos']['fr'].reshape(T, 1, 3)), axis=2)
        jt = epoch[0]['env_infos']['jt'].reshape(T, 1, 7)
        rw = epoch[0]['rewards'].reshape(T, 1)

        for sp in range(0,sample_num):
            if ((sp == 0) or (not ((sp + 1) % traj_skip))):
                sample = epoch[sp]
                ep_ = sample['observations'][:, :3].reshape(T, 1, 3)
                ep_ = np.concatenate((ep_, sample['env_infos']['er'].reshape(T, 1, 3)), axis=2)
                ev_ = sample['observations'][:, 3:].reshape(T, 1, 3)
                ev_ = np.concatenate((ev_, sample['env_infos']['erdot'].reshape(T, 1, 3)), axis=2)
                jp_ = sample['env_infos']['jx'][:, :7].reshape(T, 1, 7)
                jv_ = sample['env_infos']['jx'][:, 7:].reshape(T, 1, 7)
                ef_ = sample['actions'].reshape(T, 1, 3)
                ef_ = np.concatenate((ef_, sample['env_infos']['fr'].reshape(T, 1, 3)), axis=2)
                jt_ = sample['env_infos']['jt'].reshape(T, 1, 7)
                rw_ = sample['rewards'].reshape(T, 1)

                ep = np.concatenate((ep,ep_), axis=1)
                ev = np.concatenate((ev, ev_), axis=1)
                jp = np.concatenate((jp, jp_), axis=1)
                jv = np.concatenate((jv, jv_), axis=1)
                ef = np.concatenate((ef, ef_), axis=1)
                jt = np.concatenate((jt, jt_), axis=1)
                rw = np.concatenate((rw, rw_), axis=1)

        fig = plt.figure()
        plt.title('Epoch ' + str(i))
        plt.axis('off')
        # plot Cartesian positions
        for i in range(7):
            ax = fig.add_subplot(6, 7, i + 1)
            ax.set_title(r'$j_{%d}$' % (i + 1))
            ax.plot(tm, jp[:, :, i], color='g')

            ax = fig.add_subplot(6, 7, 7 + i + 1)
            ax.set_title(r'$\dot{j}_{%d}$' % (i + 1))
            ax.plot(tm, jv[:, :, i], color='b')

            ax = fig.add_subplot(6, 7, 7*2 + i + 1)
            ax.set_title(r'$\tau_{%d}$' % (i + 1))
            ax.plot(tm, jt[:, :, i], color='r')

            if i < 6:
                ax = fig.add_subplot(6, 7, 7 * 3 + i + 1)
                ax.set_title(r'$x_{%d}$' % (i + 1))
                ax.plot(tm, ep[:, :, i], color='m')

                ax = fig.add_subplot(6, 7, 7 * 4 + i + 1)
                ax.set_title(r'$\dot{x}_{%d}$' % (i + 1))
                ax.plot(tm, ev[:, :, i], color='c')

                ax = fig.add_subplot(6, 7, 7 * 5 + i + 1)
                ax.set_title(r'$f_{%d}$' % (i + 1))
                ax.plot(tm, ef[:, :, i], color='y')

rewards_undisc_mean = np.zeros(epoch_num)
rewards_undisc_std = np.zeros(epoch_num)
success_mat = np.zeros((epoch_num, sample_num))
state_dist_all = np.zeros((epoch_num, sample_num, T))
state_dist_last = np.zeros((epoch_num,sample_num))
for i in range(epoch_start, epoch_num):
    filename = base_filename + '/' + exp_name + '/' + 'itr_' + str(i) + '.pkl'
    infile = open(filename, 'rb')
    ep_data = pickle.load(infile)
    infile.close()
    epoch = ep_data['stats'].last_episode
    rewards_undisc_mean[i] = np.mean([np.sum(epoch[s]['rewards']) for s in range(sample_num)])
    rewards_undisc_std[i] = np.std([np.sum(epoch[s]['rewards']) for s in range(sample_num)])

    for s in range(sample_num):
        pos_norm = np.linalg.norm(epoch[s]['observations'][:, :3], axis=1)
        success_mat[i, s] = np.min(pos_norm)<SUCCESS_DIST
        # state_dist_all[i][s] = np.linalg.norm(epoch[s]['observations'][:, :3], axis=1).reshape(-1)
        # state_dist_last[i][s] = np.linalg.norm(epoch[s]['observations'][:, :3], axis=1)[-1]

# itr_state_dist = 10
# state_dist_all = state_dist_all[:itr_state_dist,:,:].reshape(-1)
# state_dist_last = state_dist_last[:itr_state_dist,:].reshape(-1)

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

plt.show()
None

