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
from garage.np.baselines import LinearFeatureBaseline
from garage.trainer import Trainer
from energybased_stable_rl.envs.yumipegcart import T, dA, dO
from garage.sampler import LocalSampler, RaySampler
from garage.sampler.default_worker import DefaultWorker
import traceback
from akro.box import Box
from garage import EnvSpec

@wrap_experiment(snapshot_mode='all')
def cem_nf_yumi(ctxt=None, seed=1):
    """Train CEM with Cartpole-v1 environment.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.

    """
    set_seed(seed)
    # env = GymEnv('CartPole-v1')
    env = GymEnv('YumiPegCart-v1', max_episode_length=T)
    env._action_space = Box(low=-10, high=10, shape=(dA,))
    env._observation_space = Box(low=-2, high=2.0, shape=(dO,))
    env._spec = EnvSpec(action_space=env.action_space,
                        observation_space=env.observation_space,
                        max_episode_length=T)

    trainer = Trainer(ctxt)


    init_std = 0.05
    init_log_std = 0.01

    init_policy = None
    policy = DeterministicNormFlowPolicy(env.spec,
                                    n_flows=2,
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
    cem_nf_yumi(seed=1)
except Exception:
    traceback.print_exc()
