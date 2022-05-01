import pickle
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from matplotlib import rc
import torch
from envs.block2D import T
import copy
import random

base_filename = '/home/shahbaz/Software/garage36/energybased_stable_rl/data/local/experiment'
font_size_1 = 12
font_size_2 = 14
font_size_3 = 10
plt.rcParams.update({'font.size': font_size_1})
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
tm = range(T)
epoch_num = 100
sample_num = 15

############################## energy level blocks ###########################################
# plt.rcParams["figure.figsize"] = (6,2.1)
#
# X1_n = 50
# X2_n = 50
# X1 = np.linspace(-0.5, 0.5, X1_n)
# X2 = np.linspace(-0.5, 0.5, X2_n)
# X1, X2 = np.meshgrid(X1, X2)
# V = np.zeros((X1_n,X2_n))
#
# exp_name = 'cem_energybased_block2d_17'
# filename = base_filename + '/' + exp_name + '/' + 'itr_' + str(0) + '.pkl'
# infile = open(filename, 'rb')
# ep_data = pickle.load(infile)
# infile.close()
# epoch = ep_data['stats'].last_episode
# Psi = ep_data['algo'].policy._module._icnn_module
#
# for i in range(X1_n):
#     for j in range(X2_n):
#         X = np.array([X1[i, j], X2[i, j]])
#         X = X.astype('float32')
#         X = torch.from_numpy(X).view(1,2)
#         V[i,j] = Psi(X)
#
# levels = np.array(range(20))*0.15
# fig = plt.figure()
# ax1 = fig.add_subplot(1, 3, 1)
# ax1.set_xlim(-0.5, 0.5)
# ax1.set_xticks([-0.5,0,0.5])
# ax1.set_ylim(-0.5, 0.5)
# ax1.set_yticks([-0.5,0,0.5])
# ax1.contour(X1, X2, V, levels, label='Potential function', colors='b', alpha=0.5)
# pos = epoch[14]['observations'][:, :2]
# ax1.plot(pos[:,0], pos[:,1], color='b',linewidth=2)
# ax1.scatter(0,0,color='r',marker='o',s=10)
# ax1.set_xlabel(r'$x_1$',fontsize=font_size_2,labelpad=1)
# ax1.set_ylabel(r'$x_2$',fontsize=font_size_2, labelpad=0.2)
#
# exp_name = 'cem_energybased_block2d_17'
# filename = base_filename + '/' + exp_name + '/' + 'itr_' + str(99) + '.pkl'
# infile = open(filename, 'rb')
# ep_data = pickle.load(infile)
# infile.close()
# epoch = ep_data['stats'].last_episode
# Psi = ep_data['algo'].policy._module._icnn_module
#
# for i in range(X1_n):
#     for j in range(X2_n):
#         X = np.array([X1[i, j], X2[i, j]])
#         X = X.astype('float32')
#         X = torch.from_numpy(X).view(1,2)
#         V[i,j] = Psi(X)
#
# levels = np.array(range(20))*0.15
# ax1.contour(X1, X2, V, levels, label='Potential function', colors='g', alpha=0.5)
# pos = epoch[14]['observations'][:, :2]
# ax1.plot(pos[:,0], pos[:,1], color='g',linewidth=2)
# ax1.scatter(0,0,color='r',marker='o',s=20)
# ax1.set_xlabel(r'$x_1$',fontsize=font_size_2,labelpad=1)
# ax1.set_ylabel(r'$x_2$',fontsize=font_size_2, labelpad=0.2)
#
# exp_name = 'cem_energybased_block2d_18'
# filename = base_filename + '/' + exp_name + '/' + 'itr_' + str(0) + '.pkl'
# infile = open(filename, 'rb')
# ep_data = pickle.load(infile)
# infile.close()
# epoch = ep_data['stats'].last_episode
# S = ep_data['algo'].policy._module._quad_pot_param
#
# for i in range(X1_n):
#     for j in range(X2_n):
#         X = np.array([X1[i, j], X2[i, j]])
#         X = X.astype('float32')
#         X = torch.from_numpy(X).view(1,2)
#         V[i,j] = X@torch.diag(S)@X.t()
#
# levels = np.array(range(20))*0.008
# ax1 = fig.add_subplot(1, 3, 2)
# ax1.set_xlim(-0.5, 0.5)
# ax1.set_xticks([-0.5,0,0.5])
# ax1.set_ylim(-0.5, 0.5)
# ax1.set_yticks([-0.5,0,0.5])
# ax1.contour(X1, X2, V, levels, label='Potential function', colors='b', alpha=0.5)
# pos = epoch[14]['observations'][:, :2]
# ax1.plot(pos[:,0], pos[:,1], color='b',linewidth=2)
# ax1.scatter(0,0,color='r',marker='o',s=20)
# ax1.set_xlabel(r'$x_1$',fontsize=font_size_2,labelpad=1)
# ax1.set_ylabel(r'$x_2$',fontsize=font_size_2, labelpad=0.2)
#
# exp_name = 'cem_energybased_block2d_18'
# filename = base_filename + '/' + exp_name + '/' + 'itr_' + str(99) + '.pkl'
# infile = open(filename, 'rb')
# ep_data = pickle.load(infile)
# infile.close()
# epoch = ep_data['stats'].last_episode
# S = ep_data['algo'].policy._module._quad_pot_param
#
# for i in range(X1_n):
#     for j in range(X2_n):
#         X = np.array([X1[i, j], X2[i, j]])
#         X = X.astype('float32')
#         X = torch.from_numpy(X).view(1,2)
#         V[i,j] = X@torch.diag(S)@X.t()
#
# levels = np.array(range(20))*0.01
# ax1.contour(X1, X2, V, levels, label='Potential function', colors='g', alpha=0.5)
# pos = epoch[14]['observations'][:, :2]
# ax1.plot(pos[:,0], pos[:,1], color='g',linewidth=2)
# ax1.scatter(0,0,color='r',marker='o',s=20)
# ax1.set_xlabel(r'$x_1$',fontsize=font_size_2,labelpad=1)
# ax1.set_ylabel(r'$x_2$',fontsize=font_size_2, labelpad=0.2)
#
# # icnn + quad itr0
# exp_name = 'cem_energybased_block2d_12'
# filename = base_filename + '/' + exp_name + '/' + 'itr_' + str(0) + '.pkl'
# infile = open(filename, 'rb')
# ep_data = pickle.load(infile)
# infile.close()
# epoch = ep_data['stats'].last_episode
# S = ep_data['algo'].policy._module._quad_pot_param
# Psi = ep_data['algo'].policy._module._icnn_module
#
# for i in range(X1_n):
#     for j in range(X2_n):
#         X = np.array([X1[i, j], X2[i, j]])
#         X = X.astype('float32')
#         X = torch.from_numpy(X).view(1,2)
#         V[i,j] = X@torch.diag(S)@X.t() + Psi(X)
#
# levels = np.array(range(20))*0.15
# ax1 = fig.add_subplot(1, 3, 3)
# ax1.set_xlim(-0.5, 0.5)
# ax1.set_xticks([-0.5,0,0.5])
# ax1.set_ylim(-0.5, 0.5)
# ax1.set_yticks([-0.5,0,0.5])
# ax1.contour(X1, X2, V, levels, label='Potential function', colors='b', alpha=0.5)
# pos = epoch[14]['observations'][:, :2]
# ax1.plot(pos[:,0], pos[:,1], color='b',linewidth=2,label='Itr 0')
# ax1.scatter(0,0,color='r',marker='o',s=20)
# ax1.set_xlabel(r'$x_1$',fontsize=font_size_2,labelpad=1)
# ax1.set_ylabel(r'$x_2$',fontsize=font_size_2, labelpad=0.2)
#
# # icnn + quad itr 99
# exp_name = 'cem_energybased_block2d_12'
# filename = base_filename + '/' + exp_name + '/' + 'itr_' + str(99) + '.pkl'
# infile = open(filename, 'rb')
# ep_data = pickle.load(infile)
# infile.close()
# epoch = ep_data['stats'].last_episode
# S = ep_data['algo'].policy._module._quad_pot_param
# Psi = ep_data['algo'].policy._module._icnn_module
#
# for i in range(X1_n):
#     for j in range(X2_n):
#         X = np.array([X1[i, j], X2[i, j]])
#         X = X.astype('float32')
#         X = torch.from_numpy(X).view(1,2)
#         V[i,j] = X@torch.diag(S)@X.t() + Psi(X)
#
# levels = np.array(range(20))*0.15
# ax1.contour(X1, X2, V, levels, label='Potential function', colors='g', alpha=0.5)
# pos = epoch[14]['observations'][:, :2]
# ax1.plot(pos[:,0], pos[:,1], color='g',linewidth=2,label='Itr 99')
# ax1.scatter(0,0,color='r',marker='o',s=20, label='Goal')
# ax1.set_xlabel(r'$x_1$',fontsize=font_size_2,labelpad=1)
# ax1.set_ylabel(r'$x_2$',fontsize=font_size_2, labelpad=0.2)
#
# off = 0.15
# plt.text(0.03+off, 0.05, '(c)', fontweight='bold', fontsize=font_size_2, transform=plt.gcf().transFigure)
# plt.text(0.36+off, 0.05, '(d)', fontweight='bold', fontsize=font_size_2, transform=plt.gcf().transFigure)
# plt.text(0.7+off, 0.05, '(e)', fontweight='bold', fontsize=font_size_2, transform=plt.gcf().transFigure)
# ax1.legend(loc='upper left', bbox_to_anchor=(-2.3, 1.35),frameon=False,ncol=3,prop={'size': 10})
# plt.subplots_adjust(left=0.1, bottom=0.36, right=0.98, top=0.85, wspace=0.6, hspace=.1)
#
# fig.savefig("blocks2d_icnn_potential_function.pdf")

############################## energy level blocks ###########################################

############################## ICNN QUAD comparison ###########################################
# color_list = ['b', 'g', 'm']
# legend_list = ['$ICNN$', '$QUAD$', '$ICNN+QUAD$']
# blocks_es = pickle.load( open( "blocks_es", "rb" ) )
# blocks_es_icnn = pickle.load( open( "blocks_es_icnn", "rb" ) )
# blocks_es_quad = pickle.load( open( "blocks_es_quad", "rb" ) )
# blocks_results = [blocks_es_icnn, blocks_es_quad, blocks_es]
#
# plt.rcParams["figure.figsize"] = (6,2.2)
#
# fig1 = plt.figure()
# plt.axis('off')
# ax1 = fig1.add_subplot(1, 2, 1)
# # ax1.set_title(r'\textbf{(c)}', position=(-0.3, 0.94), fontsize=font_size_2)
# ax1.set_xlabel(r'Iteration')
# ax1.set_ylabel(r'Reward')
# ax1.set_xlim(0,100)
# ax1.set_xticks(range(0, 100, 20))
# # ax1.set_ylim(-5.0e3,-1.5e3)
# ax2 = fig1.add_subplot(1, 2, 2)
# ax2.set_ylabel(r'Success \%')
# ax2.set_xlabel('Iteration')
# # ax2.set_title(r'\textbf{(d)}', position=(-0.25, 0.94), fontsize=font_size_2)
# ax2.set_xlim(0,100)
# ax2.set_xticks(range(0, 100, 20))
# ax2.set_yticks([0,50,100])
# ax2.set_ylim(0,100+5)
#
# width = 1
# interval = 10
# # offset = [3, 0, 2]
# offset = [0, 0, 0]
# # for i in [0, 1, 2]:
# for i in [2, 1, 0]:
#     rl_progress = blocks_results[i]
#     rewards_mean = rl_progress['reward_stat'][0]
#     rewards_std = rl_progress['reward_stat'][1]
#     success_mean = rl_progress['success_stat'][0]
#     success_std = rl_progress['success_stat'][1]
#
#     idx = np.array(range(epoch_num-1-i, 0, -interval))[::-1]
#     idx = np.concatenate((np.array([0]),idx))
#
#     heights = rewards_mean[idx]
#     yerr = rewards_std[idx]
#     # yerr[0] = 0
#     ax1.errorbar(idx, heights, yerr=yerr, label=legend_list[i], color=color_list[i])
#     ax1.ticklabel_format(axis="y", style="sci", scilimits=(0,00))
#     # ax1.legend(prop={'size': font_size_3},frameon=False)
#     ax1.legend(loc='upper left', bbox_to_anchor=(.5, 1.4), frameon=False, ncol=4, prop={'size': font_size_3})
#
#     heights = success_mean[idx]
#     yerr = success_std[idx]
#     ax2.errorbar(idx, heights, yerr=yerr, label=legend_list[i], color=color_list[i])
#
# off = 0.2
# plt.text(0.03+off, 0.05, '(a)', fontweight='bold', fontsize=font_size_2, transform=plt.gcf().transFigure)
# plt.text(0.54+off, 0.05, '(b)', fontweight='bold', fontsize=font_size_2, transform=plt.gcf().transFigure)
# plt.subplots_adjust(left=0.1, bottom=0.35, right=.99, top=0.8, wspace=0.4, hspace=0.7)
# fig1.savefig("blocks2d_icnn_vs_quad_rl_progress.pdf")
############################## ICNN QUAD comparison ###########################################

############################## RL comparison ###########################################
# color_list = ['b', 'g', 'm', 'c']
# legend_list = ['$ES-CEM$', '$NF-CEM-K1$', '$NF-CEM-K2$', '$ANN-PPO$']
# blocks_es = pickle.load( open( "blocks_es", "rb" ) )
# blocks_nf_k1 = pickle.load( open( "blocks_nf_k1", "rb" ) )
# blocks_nf_k2 = pickle.load( open( "blocks_nf_k2", "rb" ) )
# blocks_ann = pickle.load( open( "blocks_ann", "rb" ) )
# blocks_results = [blocks_es, blocks_nf_k1, blocks_nf_k2, blocks_ann]
#
# plt.rcParams["figure.figsize"] = (6,2.2)
#
# fig1 = plt.figure()
# plt.axis('off')
# ax1 = fig1.add_subplot(1, 2, 1)
# # ax1.set_title(r'\textbf{(c)}', position=(-0.3, 0.94), fontsize=font_size_2)
# ax1.set_xlabel(r'Iteration')
# ax1.set_ylabel(r'Reward')
# ax1.set_xlim(0,100)
# ax1.set_xticks(range(0, 100, 20))
# # ax1.set_ylim(-5.0e3,-1.5e3)
# ax2 = fig1.add_subplot(1, 2, 2)
# ax2.set_ylabel(r'Success \%')
# ax2.set_xlabel('Iteration')
# # ax2.set_title(r'\textbf{(d)}', position=(-0.25, 0.94), fontsize=font_size_2)
# ax2.set_xlim(0,100)
# ax2.set_xticks(range(0, 100, 20))
# ax2.set_yticks([0,50,100])
# ax2.set_ylim(0,100+5)
#
# width = 1
# interval = 10
# order = [0,1,2,3]
# # random.shuffle(order)
# print('order:',order)
# for i in order:
#     rl_progress = blocks_results[i]
#     rewards_mean = rl_progress['reward_stat'][0]
#     rewards_std = rl_progress['reward_stat'][1]
#     success_mean = rl_progress['success_stat'][0]
#     success_std = rl_progress['success_stat'][1]
#
#     idx = np.array(range(epoch_num-1-i, 0, -interval))[::-1]
#     idx = np.concatenate((np.array([0]),idx))
#
#     heights = rewards_mean[idx]
#     yerr = rewards_std[idx]
#     # yerr[0] = 0
#     ax1.errorbar(idx, heights, yerr=yerr, label=legend_list[i], color=color_list[i])
#     ax1.ticklabel_format(axis="y", style="sci", scilimits=(0,00))
#     # ax1.legend(prop={'size': font_size_3},frameon=False)
#     ax1.legend(loc='upper left', bbox_to_anchor=(-0.23, 1.5), frameon=False, ncol=4, prop={'size': 9})
#
#     heights = success_mean[idx]
#     yerr = success_std[idx]
#     ax2.errorbar(idx, heights, yerr=yerr, label=legend_list[i], color=color_list[i])
#
# off = 0.2
# plt.text(0.03+off, 0.05, '(a)', fontweight='bold', fontsize=font_size_2, transform=plt.gcf().transFigure)
# plt.text(0.54+off, 0.05, '(b)', fontweight='bold', fontsize=font_size_2, transform=plt.gcf().transFigure)
# plt.subplots_adjust(left=0.1, bottom=0.35, right=.99, top=0.8, wspace=0.4, hspace=0.7)
# fig1.savefig("blocks2d_rl_result.pdf")
############################## RL comparison ###########################################

############################## Init traj comparison ###########################################
plt.rcParams["figure.figsize"] = (6,2.2)
fig1 = plt.figure()
plt.axis('off')
ax1 = fig1.add_subplot(1, 2, 1)
ax1.set_xlabel(r'$x_1$',fontsize=font_size_2,labelpad=1)
ax1.set_ylabel(r'$x_2$',fontsize=font_size_2, labelpad=0.2)
ax1.set_xlim(-0.4,0.6)
ax1.set_ylim(-0.9,0.2)

exp = 'cem_energybased_block2d_12'
filename = base_filename + '/' + exp + '/' + 'itr_' + str(0) + '.pkl'
infile = open(filename, 'rb')
ep_data = pickle.load(infile)
infile.close()
epoch = ep_data['stats'].last_episode
for s in range(sample_num):
    pos = epoch[s]['observations'][:, :2]
    ax1.plot(pos[:, 0], pos[:, 1], color='b', linewidth=1)
ax1.scatter(0,0,color='r',marker='o',s=20, label=r'Goal')

ax2 = fig1.add_subplot(1, 2, 2)
ax2.set_xlabel(r'$x_1$',fontsize=font_size_2,labelpad=1)
ax2.set_ylabel(r'$x_2$',fontsize=font_size_2, labelpad=0.2)
ax2.set_xlim(-0.4,0.6)
ax2.set_ylim(-0.9,0.2)
ax1.legend(loc='upper left', bbox_to_anchor=(0.9, 1.5), frameon=False, prop={'size': 10})

exp = 'block2D_ppo_torch_garage'
filename = base_filename + '/' + exp + '/' + 'itr_' + str(0) + '.pkl'
infile = open(filename, 'rb')
ep_data = pickle.load(infile)
infile.close()
epoch = ep_data['stats'].last_episode
for s in range(sample_num):
    pos = epoch[s]['observations'][:, :2]
    ax2.plot(pos[:, 0], pos[:, 1], color='b', linewidth=1)
off = 0.2
ax2.scatter(0,0,color='r',marker='o',s=20)
plt.subplots_adjust(left=0.1, bottom=0.35, right=.99, top=0.8, wspace=0.4, hspace=0.7)
plt.text(0.03+off, 0.05, '(c)', fontweight='bold', fontsize=font_size_2, transform=plt.gcf().transFigure)
plt.text(0.54+off, 0.05, '(d)', fontweight='bold', fontsize=font_size_2, transform=plt.gcf().transFigure)
fig1.savefig("blocks2d_traj.pdf")
############################## Init traj comparison ###########################################