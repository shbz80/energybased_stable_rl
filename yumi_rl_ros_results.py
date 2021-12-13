import pickle
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from matplotlib import rc
import torch
import copy

font_size_1 = 12
font_size_2 = 14
font_size_3 = 10
plt.rcParams.update({'font.size': font_size_1})
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

SUCCESS_DIST = 0.005
epoch_start = 0

tm = range(500)
sample_num = 15

base_filename = '/home/shahbaz/Software/garage36/energybased_stable_rl/data/local/experiment'
plt.rcParams["figure.figsize"] = (6,2)
####### rl progress ########33


exps = ['yumipeg_energy_based_ros_1']
color_list = ['b']
# legend_list = ['$ES-CEM$', '$NF-CEM-K1$', '$NF-CEM-K4$', '$ANN-PPO$']
epoch_nums = [50]


fig1 = plt.figure()
plt.axis('off')
ax1 = fig1.add_subplot(1, 2, 1)
# ax1.set_title(r'\textbf{(c)}', position=(-0.3, 0.94), fontsize=font_size_2)
ax1.set_xlabel(r'Iteration')
ax1.set_ylabel(r'Reward')
ax1.set_xlim(0,50)
ax1.set_xticks(range(0, 50, 10))
# ax1.set_ylim(-5.0e3,-1.5e3)
ax2 = fig1.add_subplot(1, 2, 2)
ax2.set_ylabel(r'Success \%')
ax2.set_xlabel('Iteration')
# ax2.set_title(r'\textbf{(d)}', position=(-0.25, 0.94), fontsize=font_size_2)
ax2.set_xlim(0,50)
ax2.set_xticks(range(0, 50, 10))
ax2.set_yticks([0,50,100])
ax2.set_ylim(0,100+5)



# yumi_eb_rl_ros = []
# for i in range(len(exps)):
#     epoch_num = epoch_nums[i]
#     rewards_undisc_mean = np.zeros(epoch_num)
#     rewards_undisc_std = np.zeros(epoch_num)
#     success_mat = np.zeros((epoch_num, sample_num))
#     for ep in range(epoch_num):
#         filename = base_filename + '/' + exps[i] + '/' + 'itr_' + str(ep) + '.pkl'
#         infile = open(filename, 'rb')
#         ep_data = pickle.load(infile)
#         infile.close()
#         epoch = ep_data['stats'].last_episode
#         rewards_undisc_mean[ep] = np.mean([np.sum(epoch[s]['rewards']) for s in range(sample_num)])
#         rewards_undisc_std[ep] = np.std([np.sum(epoch[s]['rewards']) for s in range(sample_num)])
#         for s in range(sample_num):
#             # sample = np.min(np.absolute(epoch[s]['observations'][:,:3]), axis=0)
#             sample = np.min(np.linalg.norm(epoch[s]['observations'][:, :1], axis=1), axis=0)
#             # sample = epoch[s]['observations'][:, :6]
#             # success_mat[ep, s] = np.linalg.norm(sample) < SUCCESS_DIST
#             success_mat[ep, s] = sample < SUCCESS_DIST
#
#     success_stat = np.sum(success_mat, axis=1)*(100/sample_num)
#
#     rl_progress = {'reward_mean':copy.deepcopy(rewards_undisc_mean),
#                    'reward_std': copy.deepcopy(rewards_undisc_std),
#                    'stats': copy.deepcopy(success_stat)}
#     yumi_eb_rl_ros.append(rl_progress)
#
# pickle.dump(yumi_eb_rl_ros, open("yumi_eb_rl_ros.p", "wb"))

yumi_eb_rl_ros = pickle.load( open( "yumi_eb_rl_ros.p", "rb" ) )
width = 1
offset = [0]
for i in [0]:
    epoch_num = epoch_nums[i]
    rl_progress = yumi_eb_rl_ros[i]
    rewards_undisc_mean = rl_progress['reward_mean']
    rewards_undisc_std = rl_progress['reward_std']
    success_stat = rl_progress['stats']

    interval = 5
    # idx = i*width + offset[i]

    idx = np.array(range(epoch_num-1-i, 0, -interval))[::-1]
    idx = np.concatenate((np.array([0]), idx))
    heights = rewards_undisc_mean[idx]
    yerr = rewards_undisc_std[idx]
    # idx = np.array(range(0, epoch_num)[i*width::interval])
    ax1.errorbar(idx, heights, yerr=yerr, color=color_list[i])
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(0,00))
    # ax1.legend(prop={'size': font_size_3},frameon=False)
    ax1.legend(loc='upper left', bbox_to_anchor=(-0.23, 1.4), frameon=False, ncol=4, prop={'size': 9})

    success = success_stat[idx]
    ax2.plot(idx, success, color=color_list[i])
    # ax2.legend(prop={'size': font_size_3},frameon=False)
    off = 0.21
plt.text(0.03+off, 0.05, '(a)', fontweight='bold', fontsize=font_size_2, transform=plt.gcf().transFigure)
plt.text(0.54+off, 0.05, '(b)', fontweight='bold', fontsize=font_size_2, transform=plt.gcf().transFigure)
plt.subplots_adjust(left=0.12, bottom=0.35, right=.99, top=0.85, wspace=0.4, hspace=0.7)
fig1.savefig("yumi_rl_ros_result.pdf")


# robustness test
# base_filename = '/home/shahbaz/Software/garage36/energybased_stable_rl/data/local/experiment'
# exps = ['yumipeg_energy_based_ros_2','yumipeg_energy_based_ros_3','yumipeg_energy_based_ros_6','yumipeg_energy_based_ros_5','yumipeg_energy_based_ros_4']
# # color_list = ['b']
# legend_list = ['$\sigma_{q_0}=0.001$', '$\sigma_{q_0}=0.01$', '$\sigma_{q_0}=0.02$', '$\sigma_{q_0}=0.04$','$\sigma_{q_0}=0.1$']
# epoch_num = 1
#
# for i in range(len(exps)):
#     success_mat = np.zeros((epoch_num, sample_num))
#     for ep in range(epoch_num):
#         filename = base_filename + '/' + exps[i] + '/' + 'itr_' + str(50) + '.pkl'
#         infile = open(filename, 'rb')
#         ep_data = pickle.load(infile)
#         infile.close()
#         epoch = ep_data['stats'].last_episode
#         for s in range(sample_num):
#             # sample = np.min(np.absolute(epoch[s]['observations'][:,:3]), axis=0)
#             sample = np.min(np.linalg.norm(epoch[s]['observations'][:, :1], axis=1), axis=0)
#             # sample = epoch[s]['observations'][:, :6]
#             # success_mat[ep, s] = np.linalg.norm(sample) < SUCCESS_DIST
#             success_mat[ep, s] = sample < SUCCESS_DIST
#
#     success_stat = np.sum(success_mat, axis=1)*(100/sample_num)
#     print(legend_list[i], success_stat)
