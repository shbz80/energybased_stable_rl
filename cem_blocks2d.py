#!/usr/bin/env python3
"""This is an example to train a task with Cross Entropy Method.

Results:
    AverageReturn: 100
    RiseTime: epoch 8
"""
import tensorflow as tf
from garage import wrap_experiment
from garage.envs import GymEnv
from garage.experiment.deterministic import set_seed
from garage.np.algos import CEM
from garage.tf.policies import GaussianMLPPolicy
from garage.np.baselines import LinearFeatureBaseline
from garage.trainer import TFTrainer
from energybased_stable_rl.envs.block2D import T


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
    with TFTrainer(snapshot_config=ctxt) as trainer:
        # env = GymEnv('CartPole-v1')
        env = GymEnv('Block2D-v1', max_episode_length=T)

        policy = GaussianMLPPolicy(env.spec,
                                   hidden_sizes=[16, 16],
                                   hidden_nonlinearity=tf.nn.tanh,
                                   output_nonlinearity=None,
                                   learn_std=False,
                                   init_std=1.)

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        n_samples = 15

        algo = CEM(env_spec=env.spec,
                   policy=policy,
                   init_std=3.,
                   baseline=baseline,
                   best_frac=0.2,
                   n_samples=n_samples)

        trainer.setup(algo, env, n_workers=4)

        trainer.train(n_epochs=100, batch_size=T, plot=True, store_episodes=True)


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