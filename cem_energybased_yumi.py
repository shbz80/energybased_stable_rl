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

    init_std = 0.2
    init_log_std = 0.2

    coord_dim = env.spec.observation_space.flat_dim // 2

    damp_min = torch.ones(coord_dim)*1e-3

    icnn_bias = False

    #
    # init_policy = EnergyBasedInitPolicy(env.spec,
    #                                     S_param=torch.ones(coord_dim)*4.0,
    #                                     D_param=torch.ones(coord_dim)*2.0,
    #                                     std=2.0
    #                                     )
    init_policy = None
    policy = EnergyBasedPolicy(env.spec,
                               icnn_hidden_sizes=(24, 24),
                               w_init_icnn_y=nn.init.xavier_uniform_,
                               b_init_icnn_y=nn.init.zeros_,
                               w_init_icnn_y_param=None,  # pass None if...
                               # w_init_icnn_z=nn.init.constant_,
                               w_init_icnn_z=nn.init.xavier_uniform_,
                               w_init_icnn_z_param=None,  # pass None if...
                               icnn_bias=icnn_bias,
                               positive_type='relu',
                               # nonlinearity_icnn=torch.relu,
                               nonlinearity_icnn=smooth_relu,  # if using this change relu_grad in class ICNN
                               damper_hidden_sizes=(12, 12),
                               w_init_damper_offdiag=nn.init.xavier_uniform_,
                               b_init_damper_offdiag=nn.init.zeros_,
                               w_init_damper_diag=nn.init.xavier_uniform_,
                               b_init_damper_diag=nn.init.zeros_,
                               hidden_nonlinearity_damper=torch.tanh,
                               full_mat_damper=True,
                               damp_min=damp_min,
                               init_quad_pot=0.1,
                               min_quad_pot=1e-3,
                               max_quad_pot=0.5e1,
                               # init_quad_pot=1e-3,
                               # min_quad_pot=1e-3,
                               # max_quad_pot=1e-3,
                               icnn_min_lr=1e-4, )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    n_samples = 15

    algo = CEM(env_spec=env.spec,
               policy=policy,
               init_std=init_std,
               init_log_std=init_log_std,
               baseline=baseline,
               best_frac=0.2,
               action_lt=5.0,
               n_samples=n_samples,
               init_policy=init_policy,  # pass None if policy init is not required
               min_icnn=False,
               sensitivity=False,
               extr_std_scale=0.2,
               std_scale=1)  # 1.0: standard cem, 0.

    # n_workers should be 1
    trainer.setup(algo, env, n_workers=1, sampler_cls=LocalSampler, worker_class=DefaultWorker)

    trainer.train(n_epochs=100, batch_size=T, plot=False, store_episodes=True)

try:
    cem_energybased_yumi(seed=9)
except Exception:
    traceback.print_exc()

# cem_energybased_yumi(seed=1)1_5
# init_std = 0.2
# init_log_std = 0.2
# coord_dim = env.spec.observation_space.flat_dim // 2
# damp_min = torch.ones(coord_dim)*1e-3
# icnn_bias = False
# init_policy = None
# policy = EnergyBasedPolicy(env.spec,
#                            icnn_hidden_sizes=(24, 24),
#                            w_init_icnn_y=nn.init.xavier_uniform_,
#                            b_init_icnn_y=nn.init.zeros_,
#                            w_init_icnn_y_param=None,  # pass None if...
#                            # w_init_icnn_z=nn.init.constant_,
#                            w_init_icnn_z=nn.init.xavier_uniform_,
#                            w_init_icnn_z_param=None,  # pass None if...
#                            icnn_bias=icnn_bias,
#                            positive_type='relu',
#                            # nonlinearity_icnn=torch.relu,
#                            nonlinearity_icnn=smooth_relu,  # if using this change relu_grad in class ICNN
#                            damper_hidden_sizes=(12, 12),
#                            w_init_damper_offdiag=nn.init.xavier_uniform_,
#                            b_init_damper_offdiag=nn.init.zeros_,
#                            w_init_damper_diag=nn.init.xavier_uniform_,
#                            b_init_damper_diag=nn.init.zeros_,
#                            hidden_nonlinearity_damper=torch.tanh,
#                            full_mat_damper=True,
#                            damp_min=damp_min,
#                            init_quad_pot=0.1,
#                            min_quad_pot=1e-3,
#                            max_quad_pot=0.5e1,
#                            # init_quad_pot=1e-3,
#                            # min_quad_pot=1e-3,
#                            # max_quad_pot=1e-3,
#                            icnn_min_lr=1e-4, )
# baseline = LinearFeatureBaseline(env_spec=env.spec)
# n_samples = 15
# algo = CEM(env_spec=env.spec,
#            policy=policy,
#            init_std=init_std,
#            init_log_std=init_log_std,
#            baseline=baseline,
#            best_frac=0.2,
#            action_lt=5.0,
#            n_samples=n_samples,
#            init_policy=init_policy,  # pass None if policy init is not required
#            min_icnn=False,
#            sensitivity=False,
#            extr_std_scale=0.2,
#            std_scale=1)  # 1.0: standard cem, 0.
#
# # n_workers should be 1
# trainer.setup(algo, env, n_workers=1, sampler_cls=LocalSampler, worker_class=DefaultWorker)
# trainer.train(n_epochs=100, batch_size=T, plot=True, store_episodes=True)
# GOAL = np.array([-1.50337106, -1.24545874,  1.21963181,  0.46298941,  2.18633697,  1.51383283,
#   0.57184653])
# # obs in operational space
# GOAL_CART = [ 0.46473501,  0.10293446,  0.10217953, -0.00858317,  0.69395054,  0.71995417,
#               0.00499788,  0.,          0.,          0.,          0.,          0.,      0.]
# INIT = np.array([-0.91912945, -0.93873615,  1.03494441,  0.56895099,  1.69821677,  1.67984028, -0.06353955]) # nf rnd init 3 of 3
# T = 200
# dA = 3
# dO = 6
# dJ = 7
# D_rot = np.eye(3)*4
# SIGMA = np.array([0.05,0.05,0.01])
# SIGMA_JT = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])*2. #todo
# rand_init = False
# rand_joint_space = False
# kin_params_yumi = {}
# kin_params_yumi['urdf'] = '/home/shahbaz/Software/yumi_kinematics/yumikin/models/yumi_ABB_left.urdf'
# kin_params_yumi['base_link'] = 'world'
# # kin_params_yumi['end_link'] = 'left_tool0'
# kin_params_yumi['end_link'] = 'left_contact_point'
# kin_params_yumi['euler_string'] = 'sxyz'
# kin_params_yumi['goal'] = GOAL
# size="0.0235"

# cem_energybased_yumi_1(seed=2)2_5

# cem_energybased_yumi_2(seed=3)*_5

# cem_energybased_yumi_3(seed=4)*_5

# cem_energybased_yumi_4(seed=5)*_5

# cem_energybased_yumi_5(seed=6)3_5

# cem_energybased_yumi_6(seed=7)4_5

# cem_energybased_yumi_7(seed=8)*_5

# cem_energybased_yumi_8(seed=9)5_5