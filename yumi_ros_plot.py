import pickle
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import torch
from yumikin.YumiKinematics import YumiKinematics
from energybased_stable_rl.agent_hyperparams import kin_params_yumi
base_filename = '/home/shahbaz/Software/garage36/energybased_stable_rl/data/local/experiment'
exp_name = 'yumipeg_energy_based_ros_1'
yumikin = YumiKinematics(kin_params_yumi)
SUCCESS_DIST = 0.005
plot_skip = 5
plot_traj = False
traj_skip = 1
epoch_start = 0
epoch_num = 50
plot_energy = False
sample_num = 15

for i in range(epoch_start,epoch_num):
    if ((i==0) or (not ((i+1) % plot_skip))) and plot_traj:
        filename = base_filename + '/' + exp_name + '/' + 'itr_' + str(i) + '.pkl'
        infile = open(filename, 'rb')
        ep_data = pickle.load(infile)
        infile.close()
        epoch = ep_data['stats'].last_episode

        fig = plt.figure()
        plt.title('Epoch ' + str(i))
        plt.axis('off')
        # plot Cartesian positions

        for sp in range(0,sample_num):
            if ((sp == 0) or (not ((sp + 1) % traj_skip))):
                sample = epoch[sp]
                Q_Qdots = sample['agent_infos']['jx']
                X_Xdots = yumikin.get_cart_error_frame_list(Q_Qdots)
                F = sample['agent_infos']['ef']
                Tq = sample['agent_infos']['jt']
                rw = sample['rewards']
                ep = X_Xdots[:, :6]
                ev = X_Xdots[:, 6:]
                jp = Q_Qdots[:, :7]
                jv = Q_Qdots[:, 7:]
                ef = F
                jt = Tq
                T = ep.shape[0]
                tm = range(T)

                for i in range(7):
                    ax = fig.add_subplot(6, 7, i + 1)
                    ax.set_title(r'$j_{%d}$' % (i + 1))
                    ax.plot(tm, jp[:, i], color='g')

                    ax = fig.add_subplot(6, 7, 7 + i + 1)
                    ax.set_title(r'$\dot{j}_{%d}$' % (i + 1))
                    ax.plot(tm, jv[:, i], color='b')

                    ax = fig.add_subplot(6, 7, 7*2 + i + 1)
                    ax.set_title(r'$\tau_{%d}$' % (i + 1))
                    ax.plot(tm, jt[:, i], color='r')

                    if i < 6:
                        ax = fig.add_subplot(6, 7, 7 * 3 + i + 1)
                        ax.set_title(r'$x_{%d}$' % (i + 1))
                        ax.plot(tm, ep[:, i], color='m')

                        ax = fig.add_subplot(6, 7, 7 * 4 + i + 1)
                        ax.set_title(r'$\dot{x}_{%d}$' % (i + 1))
                        ax.plot(tm, ev[:, i], color='c')

                        ax = fig.add_subplot(6, 7, 7 * 5 + i + 1)
                        ax.set_title(r'$f_{%d}$' % (i + 1))
                        ax.plot(tm, ef[:, i], color='y')

rewards_undisc_mean = np.zeros(epoch_num)
rewards_undisc_std = np.zeros(epoch_num)
success_mat = np.zeros((epoch_num, sample_num))
state_dist_all = None
for i in range(epoch_start, epoch_num):
    filename = base_filename + '/' + exp_name + '/' + 'itr_' + str(i) + '.pkl'
    infile = open(filename, 'rb')
    ep_data = pickle.load(infile)
    infile.close()
    epoch = ep_data['stats'].last_episode
    rewards_undisc_mean[i] = np.mean([np.sum(epoch[s]['rewards']) for s in range(sample_num)])
    rewards_undisc_std[i] = np.std([np.sum(epoch[s]['rewards']) for s in range(sample_num)])
    for s in range(sample_num):
        pos_norm = np.linalg.norm(epoch[s]['observations'][:, :1], axis=1)
        success_mat[i, s] = np.min(pos_norm)<SUCCESS_DIST
        sample_dist = np.linalg.norm(epoch[s]['observations'][:, :3], axis=1).reshape(-1)
        if state_dist_all is None:
            state_dist_all = sample_dist
        else:
            if i < 10:
                state_dist_all = np.concatenate((state_dist_all, sample_dist))


print('state_dist_all mean', np.mean(state_dist_all))
print('state_dist_all std', np.std(state_dist_all))
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

