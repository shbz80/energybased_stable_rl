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
from garage.torch.policies import GaussianMLPPolicy
from energybased_stable_rl.policies.energy_based_control_policy import GaussianEnergyBasedPolicy
from garage.np.baselines import LinearFeatureBaseline
from garage.trainer import Trainer
from energybased_stable_rl.envs.block2D import T
from garage.sampler import LocalSampler, RaySampler
from garage.sampler.default_worker import DefaultWorker

@wrap_experiment(snapshot_mode='all')
def cem_block2d(ctxt=None, seed=1):
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
    # policy = GaussianMLPPolicy(env.spec,
    #                            hidden_sizes=[16, 16],
    #                            hidden_nonlinearity=torch.tanh,
    #                            output_nonlinearity=None,
    #                            learn_std=False,
    #                            init_std=1.)
    init_std = 1.0
    policy = GaussianEnergyBasedPolicy(env.spec,
                                       icnn_hidden_sizes=(16, 16),
                                       damper_hidden_sizes=(8, 8),
                                       w_init_icnn=nn.init.xavier_uniform_,
                                       # w_init_damper=nn.init.zeros_,  # todo have to modfy damper code to change this
                                       w_init_damper=nn.init.constant_, # all network init settings moved to the module file todo
                                       w_init_damper_const = .5,
                                       b_init=nn.init.zeros_,
                                       init_std=init_std,
                                       damper_full_mat=True,
                                       init_quad_pot=1.0,
                                       min_quad_pot=1e-3,
                                       max_quad_pot=1e1,
                                       icnn_min_lr=1e-1,
                                       action_limit = 5.)       # action limit stability not valid todo

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    n_samples = 15

    algo = CEM(env_spec=env.spec,
               policy=policy,
               init_std=init_std,
               baseline=baseline,
               best_frac=0.2,
               n_samples=n_samples)

    trainer.setup(algo, env, n_workers=1, sampler_cls=LocalSampler, worker_class=DefaultWorker)

    trainer.train(n_epochs=50, batch_size=T, plot=False, store_episodes=True)


cem_block2d(seed=1)

# cem_block2d_20(seed=1)
# policy = GaussianMLPPolicy(env.spec,
#                                    hidden_sizes=[16, 16],
#                                    hidden_nonlinearity=tf.nn.tanh,
#                                    output_nonlinearity=None,
#                                    learn_std=False,
#                                    init_std=1.)
#
# baseline = LinearFeatureBaseline(env_spec=env.spec)
#
# n_samples = 15
#
# algo = CEM(env_spec=env.spec,
#            policy=policy,
#            init_std=3.,
#            baseline=baseline,
#            best_frac=0.2,
#            n_samples=n_samples)
#
# trainer.setup(algo, env, n_workers=4)
#
# trainer.train(n_epochs=100, batch_size=T, plot=True, store_episodes=True)
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
# size="0.05 0.048 0.05"