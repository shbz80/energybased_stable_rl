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
from energybased_stable_rl.policies.energy_based_control_policy import EnergyBasedPolicy
from energybased_stable_rl.policies.energy_based_init_policy import EnergyBasedInitPolicy
from garage.np.baselines import LinearFeatureBaseline
from garage.trainer import Trainer
from energybased_stable_rl.envs.yumipegcart import T, dA, dO
from garage.sampler import LocalSampler, RaySampler
from garage.sampler.default_worker import DefaultWorker
import traceback
from akro.box import Box
from garage import EnvSpec

@wrap_experiment(snapshot_mode='all')
def cem_energybased_yumi(ctxt=None, seed=1):
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

    # log normal stuff
    desired_lognormal_mean = torch.tensor(0.3)
    desired_mean = desired_lognormal_mean
    desired_std = 0.05
    log_mean = torch.log((desired_mean ** 2) / torch.sqrt((desired_mean ** 2 + desired_std ** 2)))
    log_std = torch.sqrt(torch.log(1 + (desired_std ** 2) / (desired_mean ** 2)))
    init_log_std = log_std

    init_std = 0.05

    damp_min = torch.ones(3)*1e-2

    coord_dim = env.spec.observation_space.flat_dim//2

    init_policy = EnergyBasedInitPolicy(env.spec,
                                        S_param=torch.ones(coord_dim)*4.0,
                                        D_param=torch.ones(coord_dim)*2.0,
                                        std=2.0
                                        )

    policy = EnergyBasedPolicy(env.spec,
                                       icnn_hidden_sizes=(16,16),
                                       w_init_icnn_y=nn.init.xavier_uniform_,
                                       b_init_icnn_y=nn.init.zeros_,
                                       w_init_icnn_z=nn.init.constant_,
                                       w_init_icnn_z_param=log_mean,
                                       nonlinearity_icnn=torch.relu,
                                       damper_hidden_sizes=(8, 8),
                                       w_init_damper_offdiag=nn.init.xavier_uniform_,
                                       b_init_damper_offdiag=nn.init.zeros_,
                                       w_init_damper_diag=nn.init.xavier_uniform_,
                                       w_init_damper_diag_param=log_mean,
                                       b_init_damper_diag=nn.init.zeros_,
                                       hidden_nonlinearity_damper=torch.tanh,
                                       full_mat_damper=True,
                                       damp_min = damp_min,
                                       init_quad_pot=1.0,
                                       min_quad_pot=1e-3,
                                       max_quad_pot=0.5e1,
                                       icnn_min_lr=1e-4,)

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    n_samples = 15

    algo = CEM(env_spec=env.spec,
               policy=policy,
               init_std=init_std,
               init_log_std = init_log_std,
               baseline=baseline,
               best_frac=0.2,
               action_lt=10.0,
               n_samples=n_samples,
               init_policy=init_policy)

    trainer.setup(algo, env, n_workers=1, sampler_cls=LocalSampler, worker_class=DefaultWorker)

    trainer.train(n_epochs=50, batch_size=T, plot=True, store_episodes=True)

try:
    cem_energybased_yumi(seed=3)
except Exception:
    traceback.print_exc()

