#!/usr/bin/env python3
"""This is an example to train a task with SAC algorithm written in PyTorch."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from garage import wrap_experiment
from garage.envs import GymEnv, normalize
from garage.experiment import deterministic
from garage.replay_buffer import PathBuffer
from garage.sampler import LocalSampler
from garage.sampler.default_worker import DefaultWorker
from garage.sampler import RaySampler
from garage.torch import set_gpu_mode
from garage.torch.algos import SAC
from garage.torch.policies import TanhGaussianMLPPolicy
from garage.torch.q_functions import ContinuousMLPQFunction
from garage.trainer import Trainer
from energybased_stable_rl.envs.block2D import T

@wrap_experiment(snapshot_mode='all')
def block_2d_sac_torch(ctxt=None, seed=1):
    """Set up environment and algorithm and run the task.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.

    """
    deterministic.set_seed(seed)
    trainer = Trainer(snapshot_config=ctxt)
    # env = normalize(GymEnv('HalfCheetah-v2'))
    env = normalize(GymEnv('Block2D-v1', max_episode_length=T))

    policy = TanhGaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=[32, 32],
        hidden_nonlinearity=nn.ReLU,
        output_nonlinearity=None,
        min_std=np.exp(-20.),
        max_std=np.exp(2.),
    )

    qf1 = ContinuousMLPQFunction(env_spec=env.spec,
                                 hidden_sizes=[32, 32],
                                 hidden_nonlinearity=F.relu)

    qf2 = ContinuousMLPQFunction(env_spec=env.spec,
                                 hidden_sizes=[32, 32],
                                 hidden_nonlinearity=F.relu)

    replay_buffer = PathBuffer(capacity_in_transitions=int(1e6))

    N = 200  # number of epochs
    S = 20  # number of episodes in an epoch

    sac = SAC(env_spec=env.spec,
              policy=policy,
              qf1=qf1,
              qf2=qf2,
              gradient_steps_per_itr=500,
              fixed_alpha=None,
              max_episode_length_eval=T,
              replay_buffer=replay_buffer,
              min_buffer_size=3*T*S,
              target_update_tau=5e-3,
              discount=1.0,
              buffer_batch_size=256,
              reward_scale=1.,
              steps_per_epoch=1)

    if torch.cuda.is_available():
        set_gpu_mode(True)
    else:
        set_gpu_mode(False)
    sac.to()
    trainer.setup(algo=sac, env=env, n_workers=4, sampler_cls=RaySampler, worker_class=DefaultWorker)
    # trainer.setup(algo=sac, env=env, sampler_cls=LocalSampler)
    trainer.train(n_epochs=N, batch_size=T*S, plot=True, store_episodes=True)


s = np.random.randint(0, 1000)
block_2d_sac_torch(seed=3)

# block_2d_sac_torch_2(seed=1)
# policy = TanhGaussianMLPPolicy(
#         env_spec=env.spec,
#         hidden_sizes=[256, 256],
#         hidden_nonlinearity=nn.ReLU,
#         output_nonlinearity=None,
#         min_std=np.exp(-20.),
#         max_std=np.exp(2.),
#     )
#
# qf1 = ContinuousMLPQFunction(env_spec=env.spec,
#                              hidden_sizes=[256, 256],
#                              hidden_nonlinearity=F.relu)
#
# qf2 = ContinuousMLPQFunction(env_spec=env.spec,
#                              hidden_sizes=[256, 256],
#                              hidden_nonlinearity=F.relu)
#
# replay_buffer = PathBuffer(capacity_in_transitions=int(1e6))
#
# N = 200  # number of epochs
# S = 20  # number of episodes in an epoch
#
# sac = SAC(env_spec=env.spec,
#           policy=policy,
#           qf1=qf1,
#           qf2=qf2,
#           gradient_steps_per_itr=100,
#           fixed_alpha=0.5,
#           max_episode_length_eval=T,
#           replay_buffer=replay_buffer,
#           min_buffer_size=T*S,
#           target_update_tau=5e-3,
#           discount=0.99,
#           buffer_batch_size=256,
#           reward_scale=1.,
#           steps_per_epoch=1)
# trainer.setup(algo=sac, env=env, n_workers=4, sampler_cls=RaySampler, worker_class=DefaultWorker)
# trainer.train(n_epochs=N, batch_size=T*S, plot=True, store_episodes=True)
# size="0.05 0.048 0.05"
# OFFSET = np.array([0.4, -0.6])
# OFFSET_1 = np.array([0, 0])         # NFPPPO fixed init
# GOAL = np.array([0, 0.5])+OFFSET
# INIT = np.array([-0.3, 0.8])+OFFSET+OFFSET_1
# T = 200
# POS_SCALE = 1
# VEL_SCALE = 0.1
# ACTION_SCALE = 1e-3
# v = 2
# w = 1
# TERMINAL_STATE_SCALE = 10

# block_2d_sac_torch(seed=1)
#           gradient_steps_per_itr=1000,
#           fixed_alpha=None,
