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
from energybased_stable_rl.envs.block2D import T
from garage.sampler import LocalSampler, RaySampler
from garage.sampler.default_worker import DefaultWorker
import traceback

def smooth_relu(x):
    d = torch.tensor(.01)
    z = torch.zeros_like(x)
    if torch.any(x <= 0.0):
        z[(x <= 0.0)] = 0.0

    if torch.any(torch.logical_and(x > 0.0, x < d)):
        z[torch.logical_and(x > 0.0, x < d)] = x[torch.logical_and(x > 0.0, x < d)]**2/(2.0*d)

    if torch.any(x >= d):
        z[(x >= d)] = x[(x >= d)]-(d/2.0)

    return z


@wrap_experiment(snapshot_mode='all')
def cem_energybased_block2d(ctxt=None, seed=1):
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

    # log normal stuff
    desired_lognormal_mean = torch.tensor(.3)
    desired_mean = desired_lognormal_mean
    desired_std = 0.001
    log_mean = torch.log((desired_mean ** 2) / torch.sqrt((desired_mean ** 2 + desired_std ** 2)))
    log_std = torch.sqrt(torch.log(1 + (desired_std ** 2) / (desired_mean ** 2)))
    init_log_std = log_std


    # init_std = 0.05
    # init_log_std = 1.0

    init_std = 0.2
    init_log_std = 0.2

    coord_dim = env.spec.observation_space.flat_dim // 2

    damp_min = torch.ones(coord_dim) * 1e-3

    icnn_bias = False

    init_policy = EnergyBasedInitPolicy(env.spec,
                                        S_param=torch.ones(coord_dim) * 4.0,
                                        D_param=torch.ones(coord_dim) * 2.0,
                                        std=2.0
                                        )
    init_policy = None
    policy = EnergyBasedPolicy(env.spec,
                                       icnn_hidden_sizes=(16,16),
                                       w_init_icnn_y=nn.init.xavier_uniform_,
                                       b_init_icnn_y=nn.init.zeros_,
                                        w_init_icnn_y_param=None,  # pass None if...
                                       # w_init_icnn_z=nn.init.constant_,
                                        w_init_icnn_z=nn.init.xavier_uniform_,
                                       w_init_icnn_z_param=None,        # pass None if...
                                       icnn_bias = icnn_bias,
                                       positive_type = 'relu',
                                       # nonlinearity_icnn=torch.relu,
                                        nonlinearity_icnn=smooth_relu,  # if using this change relu_grad in class ICNN
                                       damper_hidden_sizes=(8, 8),
                                       w_init_damper_offdiag=nn.init.xavier_uniform_,
                                       b_init_damper_offdiag=nn.init.zeros_,
                                       w_init_damper_diag=nn.init.xavier_uniform_,
                                       b_init_damper_diag=nn.init.zeros_,
                                       hidden_nonlinearity_damper=torch.tanh,
                                       full_mat_damper=True,
                                       damp_min = damp_min,
                                       init_quad_pot=0.1,
                                       min_quad_pot=1e-3,
                                       max_quad_pot=0.5e1,
                                       # init_quad_pot=1e-3,
                                       # min_quad_pot=1e-3,
                                       # max_quad_pot=1e-3,
                                       icnn_min_lr=1e-4,)

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
               min_icnn=False,
               sensitivity=False,       # do not use this
               extr_std_scale=0.5,
               std_scale=1.0)   # 1.0: standard cem, 0.0: sensitivity scaled cem

    # n_workers should be 1
    trainer.setup(algo, env, n_workers=1, sampler_cls=LocalSampler, worker_class=DefaultWorker)

    trainer.train(n_epochs=50, batch_size=T, plot=True, store_episodes=True)

try:
    cem_energybased_block2d(seed=1)
except Exception:
    traceback.print_exc()

# cem_energybased_block2d(seed=1)
# only icnn no quad
# init_std = 0.05
# init_log_std = 1.0
# coord_dim = env.spec.observation_space.flat_dim // 2
# damp_min = torch.ones(coord_dim) * 1e-3
# icnn_bias = False
# init_policy = None
# policy = EnergyBasedPolicy(env.spec,
#                                    icnn_hidden_sizes=(16,16),
#                                    w_init_icnn_y=nn.init.xavier_uniform_,
#                                    b_init_icnn_y=nn.init.zeros_,
#                                     w_init_icnn_y_param=None,  # pass None if...
#                                    # w_init_icnn_z=nn.init.constant_,
#                                     w_init_icnn_z=nn.init.xavier_uniform_,
#                                    w_init_icnn_z_param=None,        # pass None if...
#                                    icnn_bias = icnn_bias,
#                                    positive_type = 'relu',
#                                    nonlinearity_icnn=torch.relu,
#                                    damper_hidden_sizes=(8, 8),
#                                    w_init_damper_offdiag=nn.init.xavier_uniform_,
#                                    b_init_damper_offdiag=nn.init.zeros_,
#                                    w_init_damper_diag=nn.init.xavier_uniform_,
#                                    b_init_damper_diag=nn.init.zeros_,
#                                    hidden_nonlinearity_damper=torch.tanh,
#                                    full_mat_damper=True,
#                                    damp_min = damp_min,
#                                    init_quad_pot=1.,
#                                    min_quad_pot=1e-3,
#                                    max_quad_pot=0.5e1,
#                                    # init_quad_pot=1e-3,
#                                    # min_quad_pot=1e-3,
#                                    # max_quad_pot=1e-3,
#                                    icnn_min_lr=1e-4,)
# baseline = LinearFeatureBaseline(env_spec=env.spec)
# n_samples = 15
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
# # n_workers should be 1
# trainer.setup(algo, env, n_workers=1, sampler_cls=LocalSampler, worker_class=DefaultWorker)
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

# cem_energybased_block2d_1(seed=1)
# only quad no icnn
# init_quad_pot=4.,

# cem_energybased_block2d_2(seed=1)
# full, pos 1
# init_quad_pot=0.1,
# min_quad_pot=1e-3,
# max_quad_pot=0.5e1,

# cem_energybased_block2d_3(seed=1)
# pos 2

# cem_energybased_block2d_4(seed=1)
# pos 1 pure cem
# init_std = 0.5
# init_log_std = 0.5
# coord_dim = env.spec.observation_space.flat_dim // 2
# damp_min = torch.ones(coord_dim) * 1e-3
# icnn_bias = False
# init_policy = None
# policy = EnergyBasedPolicy(env.spec,
#                                    icnn_hidden_sizes=(16,16),
#                                    w_init_icnn_y=nn.init.xavier_uniform_,
#                                    b_init_icnn_y=nn.init.zeros_,
#                                     w_init_icnn_y_param=None,  # pass None if...
#                                    # w_init_icnn_z=nn.init.constant_,
#                                     w_init_icnn_z=nn.init.xavier_uniform_,
#                                    w_init_icnn_z_param=None,        # pass None if...
#                                    icnn_bias = icnn_bias,
#                                    positive_type = 'relu',
#                                    nonlinearity_icnn=torch.relu,
#                                    damper_hidden_sizes=(8, 8),
#                                    w_init_damper_offdiag=nn.init.xavier_uniform_,
#                                    b_init_damper_offdiag=nn.init.zeros_,
#                                    w_init_damper_diag=nn.init.xavier_uniform_,
#                                    b_init_damper_diag=nn.init.zeros_,
#                                    hidden_nonlinearity_damper=torch.tanh,
#                                    full_mat_damper=True,
#                                    damp_min = damp_min,
#                                    init_quad_pot=0.1,
#                                    min_quad_pot=1e-3,
#                                    max_quad_pot=0.5e1,
#                                    # init_quad_pot=1e-3,
#                                    # min_quad_pot=1e-3,
#                                    # max_quad_pot=1e-3,
#                                    icnn_min_lr=1e-4,)
# n_samples = 15
# algo = CEM(env_spec=env.spec,
#            policy=policy,
#            init_std=init_std,
#            init_log_std = init_log_std,
#            baseline=baseline,
#            best_frac=0.2,
#            action_lt=5.0,
#            n_samples=n_samples,
#            init_policy=init_policy,     # pass None if policy init is not required
#            min_icnn=False,
#            sensitivity=False,
#            extr_std_scale=0.5,
#            std_scale=1)   # 1.0: standard cem, 0.0: sensitivity scaled cem
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

# cem_energybased_block2d_5(seed=1)
# pos 1 mixed cem
# std_scale=0.5

# cem_energybased_block2d_6(seed=1)
# pos 1 mixed cem
# std_scale=0

# cem_energybased_block2d_7(seed=1)
# pos 2 pure cem
# std_scale=0

# cem_energybased_block2d_8(seed=3)
# pos 2 mixed cem
# std_scale=0.5

# cem_energybased_block2d_9(seed=1)
# init_std = 0.2
# init_log_std = 0.2
# coord_dim = env.spec.observation_space.flat_dim // 2
# damp_min = torch.ones(coord_dim) * 1e-3
# icnn_bias = False
# init_policy = None
# policy = EnergyBasedPolicy(env.spec,
#                                    icnn_hidden_sizes=(16,16),
#                                    w_init_icnn_y=nn.init.xavier_uniform_,
#                                    b_init_icnn_y=nn.init.zeros_,
#                                     w_init_icnn_y_param=None,  # pass None if...
#                                    # w_init_icnn_z=nn.init.constant_,
#                                     w_init_icnn_z=nn.init.xavier_uniform_,
#                                    w_init_icnn_z_param=None,        # pass None if...
#                                    icnn_bias = icnn_bias,
#                                    positive_type = 'relu',
#                                    nonlinearity_icnn=torch.relu,
#                                    damper_hidden_sizes=(8, 8),
#                                    w_init_damper_offdiag=nn.init.xavier_uniform_,
#                                    b_init_damper_offdiag=nn.init.zeros_,
#                                    w_init_damper_diag=nn.init.xavier_uniform_,
#                                    b_init_damper_diag=nn.init.zeros_,
#                                    hidden_nonlinearity_damper=torch.tanh,
#                                    full_mat_damper=True,
#                                    damp_min = damp_min,
#                                    init_quad_pot=0.1,
#                                    min_quad_pot=1e-3,
#                                    max_quad_pot=0.5e1,
#                                    # init_quad_pot=1e-3,
#                                    # min_quad_pot=1e-3,
#                                    # max_quad_pot=1e-3,
#                                    icnn_min_lr=1e-4,)
# baseline = LinearFeatureBaseline(env_spec=env.spec)
# n_samples = 15
# algo = CEM(env_spec=env.spec,
#            policy=policy,
#            init_std=init_std,
#            init_log_std = init_log_std,
#            baseline=baseline,
#            best_frac=0.2,
#            action_lt=5.0,
#            n_samples=n_samples,
#            init_policy=init_policy,     # pass None if policy init is not required
#            min_icnn=False,
#            sensitivity=False,
#            extr_std_scale=0.2,
#            std_scale=1.0)   # 1.0: standard cem, 0.0: sensitivity scaled cem
# # n_workers should be 1
# trainer.setup(algo, env, n_workers=1, sampler_cls=LocalSampler, worker_class=DefaultWorker)
# trainer.train(n_epochs=50, batch_size=T, plot=True, store_episodes=True)
# OFFSET = np.array([0.4, -0.6])
# OFFSET_1 = np.array([0, 0])         # NFPPPO fixed init
# INIT = np.array([0.0, -0.1])+OFFSET # pos2
# INIT = np.array([0.0, -0.1])+OFFSET+OFFSET_1
# T = 200
# POS_SCALE = 1
# VEL_SCALE = 0.1
# ACTION_SCALE = 1e-3
# v = 2
# w = 1
# TERMINAL_STATE_SCALE = 10
# size="0.05 0.048 0.05"

# cem_energybased_block2d_10(seed=1)
# smooth_relu, pos 2

# cem_energybased_block2d_11(seed=1)
# smooth_relu, pos 2, pure icnn