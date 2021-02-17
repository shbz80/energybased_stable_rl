#!/usr/bin/env python3
"""This is an example to train a task with Cross Entropy Method.

Results:
    AverageReturn: 100
    RiseTime: epoch 8
"""
import torch
import torch.nn as nn
from garage import wrap_experiment
from garage.envs import GymEnv
from garage.experiment.deterministic import set_seed
from energybased_stable_rl.algos.cem import CEM
from energybased_stable_rl.policies.nf_deterministic_policy import DeterministicNormFlowPolicy
from normflow_policy.normflow_policy_garage import GaussianNormFlowPolicy
from garage.np.baselines import LinearFeatureBaseline
from garage.trainer import Trainer
from energybased_stable_rl.envs.block2D import T
from garage.sampler import LocalSampler, RaySampler
from garage.sampler.default_worker import DefaultWorker
import traceback

@wrap_experiment(snapshot_mode='all')
def cem_nf_block2d(ctxt=None, seed=1):
    """Train CEM with Cartpole-v1 environment.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.

    """
    set_seed(seed)
    # env = GymEnv('CartPole-v1')
    env = GymEnv('Block2D-v1', max_episode_length=T)
    trainer = Trainer(ctxt)


    init_std = 0.5
    init_log_std = 0.01

    init_policy = None
    policy = DeterministicNormFlowPolicy(env.spec,
                                    n_flows=1,
                                    hidden_dim=16,
                                    init_std=2.,
                                    K=1.0,
                                    D=1.0,
                                     init_func=nn.init.xavier_uniform_,
                                     init_const=None,
                                     # init_func=nn.init.constant_,
                                     # init_const=0.1,
                                    jac_damping=True)

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    n_samples = 15

    algo = CEM(env_spec=env.spec,
               policy=policy,
               init_std=init_std,
               init_log_std = init_log_std,
               baseline=baseline,
               best_frac=0.2,
               action_lt=5.0,
               n_samples=n_samples,
               init_policy=init_policy,     # pass None if policy init is not required
               min_icnn=False)

    # n_workers should be 1
    trainer.setup(algo, env, n_workers=1, sampler_cls=LocalSampler, worker_class=DefaultWorker)

    trainer.train(n_epochs=50, batch_size=T, plot=True, store_episodes=True)

try:
    cem_nf_block2d(seed=1)
except Exception:
    traceback.print_exc()

# cem_nf_block2d(seed=1)
# init_std = 0.05
# init_log_std = 0.01
# init_policy = None
# policy = DeterministicNormFlowPolicy(env.spec,
#                                 n_flows=2,
#                                 hidden_dim=8,
#                                 init_std=2.,
#                                 K=1.0,
#                                 D=1.0,
#                                 jac_damping=True)
#
# baseline = LinearFeatureBaseline(env_spec=env.spec)
#
# n_samples = 15
#
# algo = CEM(env_spec=env.spec,
#            policy=policy,
#            init_std=init_std,
#            init_log_std = init_log_std,
#            baseline=baseline,
#            best_frac=0.2,
#            action_lt=5.0,
#            n_samples=n_samples,
#            init_policy=init_policy,     # pass None if policy init is not required
#            min_icnn=False)
#
# # n_workers should be 1
# trainer.setup(algo, env, n_workers=1, sampler_cls=LocalSampler, worker_class=DefaultWorker)
#
# trainer.train(n_epochs=50, batch_size=T, plot=True, store_episodes=True)
# OFFSET = np.array([0.4, -0.6])
# OFFSET_1 = np.array([0, 0])         # NFPPPO fixed init
# GOAL = np.array([0, 0.5])+OFFSET
# INIT = np.array([-0.3, 0.8])+OFFSET+OFFSET_1      pos 1
# T = 200
# POS_SCALE = 1
# VEL_SCALE = 0.1
# ACTION_SCALE = 1e-3
# v = 2
# w = 1
# TERMINAL_STATE_SCALE = 10
# size="0.05 0.048 0.05"

# cem_nf_block2d_1(seed=1)
# K=4.0,
# D=2.0,