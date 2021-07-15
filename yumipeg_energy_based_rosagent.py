import numpy as np
import torch
import gym
import torch
import torch.nn as nn
from garage import wrap_experiment
from garage.envs import GymEnv
from garage.experiment.deterministic import set_seed
from energybased_stable_rl.algos.cem import CEM
from garage.trainer import Trainer
from energybased_stable_rl.envs.yumipegcart import T, dA, dO
from garage.np.baselines import LinearFeatureBaseline
from energybased_stable_rl.policies.energy_based_control_policy import EnergyBasedPolicy
from akro.box import Box
from garage import EnvSpec
from garage.sampler import RaySampler, LocalSampler
from gps.agent.ros.agent_ros import AgentROS
from energybased_stable_rl.agent_hyperparams import agent as agent_params
import traceback

T = agent_params['T']
S = 15
N = agent_params['epoch_num']
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

# @wrap_experiment
@wrap_experiment(snapshot_mode='all')
def yumipeg_energy_based_ros(ctxt=None, seed=1):
    """Train PPO with InvertedDoublePendulum-v2 environment.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.

    """
    set_seed(seed)
    env = GymEnv('YumiPegCart-v1', max_episode_length=T)
    env._action_space = Box(low=-10, high=10, shape=(dA,))
    env._observation_space = Box(low=-2, high=2.0, shape=(dO,))
    env._spec = EnvSpec(action_space=env.action_space,
                             observation_space=env.observation_space,
                             max_episode_length=T)
    trainer = Trainer(ctxt)
    init_std = 0.3
    init_log_std = 0.3
    coord_dim = env.spec.observation_space.flat_dim // 2
    damp_min = torch.ones(coord_dim) * 1e-3
    icnn_bias = False
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
                               nonlinearity_icnn=smooth_relu,
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

    algo = CEM(env_spec=env.spec,
               policy=policy,
               init_std=init_std,
               init_log_std=init_log_std,
               baseline=baseline,
               best_frac=0.2,
               action_lt=5.0,
               n_samples=S,
               init_policy=init_policy,  # pass None if policy init is not required
               min_icnn=False,
               sensitivity=False,
               extr_std_scale=0.4,
               std_scale=1)  # 1.0: standard cem, 0.

    # resume_dir = '/home/shahbaz/Software/garage36/energybased_stable_rl/data/local/experiment/yumipeg_energy_based_ros_1'
    # trainer.restore(resume_dir, from_epoch=49)
    # trainer.resume(n_epochs=51)
    trainer.setup(algo, env, sampler_cls=AgentROS, sampler_args= agent_params)
    trainer.train(n_epochs=N, batch_size=T, plot=False, store_episodes=True)
try:
    yumipeg_energy_based_ros(seed=1)
except Exception:
    traceback.print_exc()

# yumipeg_energy_based_ros_1(seed=1)
# init_std = 0.3
# init_log_std = 0.3
# coord_dim = env.spec.observation_space.flat_dim // 2
# damp_min = torch.ones(coord_dim) * 1e-3
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
#                            nonlinearity_icnn=smooth_relu,
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
#
# baseline = LinearFeatureBaseline(env_spec=env.spec)
#
# algo = CEM(env_spec=env.spec,
#            policy=policy,
#            init_std=init_std,
#            init_log_std=init_log_std,
#            baseline=baseline,
#            best_frac=0.2,
#            action_lt=5.0,
#            n_samples=S,
#            init_policy=init_policy,  # pass None if policy init is not required
#            min_icnn=False,
#            sensitivity=False,
#            extr_std_scale=0.4,
#            std_scale=1)  # 1.0: standard cem, 0.
# GOAL = np.array([-1.5924, -1.0417,  1.4975,  0.1163,  2.0738,  1.1892, -2.4352])
# ja_x0 = np.array([-1.3687, -0.8996, 1.2355, 0.5462, 1.9021, 1.3573, -2.6171])
# T =500
# reward_params = {}
# reward_params['LIN_SCALE'] = 1
# reward_params['ROT_SCALE'] = 0
# reward_params['POS_SCALE'] = 1
# reward_params['VEL_SCALE'] = 1e-3
# reward_params['STATE_SCALE'] = 1
# reward_params['ACTION_SCALE'] = 1e-2
# reward_params['v'] = 2
# reward_params['w'] = 1
# reward_params['TERMINAL_STATE_SCALE'] = 500
# reward_params['T'] = T
#
# kin_params_yumi = {}
# kin_params_yumi['urdf'] = '/home/shahbaz/Software/yumi_kinematics/yumikin/models/yumi_ABB_left.urdf'
# kin_params_yumi['base_link'] = 'world'
# kin_params_yumi['end_link'] = 'left_tool0'
# # kin_params_yumi['end_link'] = 'left_contact_point'
# kin_params_yumi['euler_string'] = 'sxyz'
# kin_params_yumi['goal'] = GOAL
#
# agent = {
#     'dt': 0.01,
#     'conditions': common['conditions'],
#     'T': T,
#     'trial_timeout': 6,
#     'reset_timeout': 20,
#     'episode_num': 1,       # this should be 1 for CEM
#     'epoch_num': 50,
#     'x0': x0s,
#     'ee_points_tgt': ee_tgts,
#     'reset_conditions': reset_conditions,
#     'sensor_dims': SENSOR_DIMS,
#     'state_include': [JOINT_ANGLES, JOINT_VELOCITIES],
#     'end_effector_points': EE_POINTS,
#     'obs_include': [],
#     'reward': reward_params,
#     'kin_params': kin_params_yumi,
#     'K_tra': 100*np.eye(3),
#     'K_rot': 3.*np.eye(3),
# }

# _7 std 0.04
# _6 std 0.02
# _5 std 0.05
# _4 std 0.1
# _3 std 0.01
# _2 std 0.001