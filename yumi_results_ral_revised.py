import pickle
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from matplotlib import rc
import torch
from energybased_stable_rl.envs.yumipegcart import T
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

############################## RL comparison ###########################################
color_list = ['b', 'g', 'm', 'c']
legend_list = ['$ES-CEM$', '$NF-CEM-K1$', '$NF-CEM-K4$', '$ANN-PPO$']
yumi_es = pickle.load( open( "yumi_es", "rb" ) )
yumi_nf_k1 = pickle.load( open( "yumi_nf_k1", "rb" ) )
yumi_nf_k4 = pickle.load( open( "yumi_nf_k4", "rb" ) )
yumi_ann = pickle.load( open( "yumi_ann", "rb" ) )
yumi_results = [yumi_es, yumi_nf_k1, yumi_nf_k4, yumi_ann]

plt.rcParams["figure.figsize"] = (6,2.2)

fig1 = plt.figure()
plt.axis('off')
ax1 = fig1.add_subplot(1, 2, 1)
# ax1.set_title(r'\textbf{(c)}', position=(-0.3, 0.94), fontsize=font_size_2)
ax1.set_xlabel(r'Iteration')
ax1.set_ylabel(r'Reward')
ax1.set_xlim(0,100)
ax1.set_xticks(range(0, 100, 20))
# ax1.set_ylim(-5.0e3,-1.5e3)
ax2 = fig1.add_subplot(1, 2, 2)
ax2.set_ylabel(r'Success \%')
ax2.set_xlabel('Iteration')
# ax2.set_title(r'\textbf{(d)}', position=(-0.25, 0.94), fontsize=font_size_2)
ax2.set_xlim(0,100)
ax2.set_xticks(range(0, 100, 20))
ax2.set_yticks([0,50,100])
ax2.set_ylim(0,100+5)

width = 1
interval = 10
order = [0,1,2,3]
# random.shuffle(order)
print('order:',order)
for i in order:
    rl_progress = yumi_results[i]
    rewards_mean = rl_progress['reward_stat'][0]
    rewards_std = rl_progress['reward_stat'][1]
    success_mean = rl_progress['success_stat'][0]
    success_std = rl_progress['success_stat'][1]

    idx = np.array(range(epoch_num-1-i, 0, -interval))[::-1]
    idx = np.concatenate((np.array([0]),idx))

    heights = rewards_mean[idx]
    yerr = rewards_std[idx]
    # yerr[0] = 0
    ax1.errorbar(idx, heights, yerr=yerr, label=legend_list[i], color=color_list[i])
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(0,00))
    # ax1.legend(prop={'size': font_size_3},frameon=False)
    ax1.legend(loc='upper left', bbox_to_anchor=(-0.23, 1.5), frameon=False, ncol=4, prop={'size': 9})

    heights = success_mean[idx]
    yerr = success_std[idx]
    ax2.errorbar(idx, heights, yerr=yerr, label=legend_list[i], color=color_list[i])

off = 0.2
plt.text(0.03+off, 0.05, '(a)', fontweight='bold', fontsize=font_size_2, transform=plt.gcf().transFigure)
plt.text(0.54+off, 0.05, '(b)', fontweight='bold', fontsize=font_size_2, transform=plt.gcf().transFigure)
plt.subplots_adjust(left=0.11, bottom=0.35, right=.99, top=0.8, wspace=0.4, hspace=0.7)
fig1.savefig("yumi_rl_result.pdf")
############################## RL comparison ###########################################