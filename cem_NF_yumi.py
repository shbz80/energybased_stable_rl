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


    init_std = 0.15
    init_log_std = 0.15

    init_policy = None
    policy = DeterministicNormFlowPolicy(env.spec,
                                    n_flows=2,
                                    hidden_dim=16,
                                    init_std=2.,
                                    K=4.0,
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
               init_log_std=init_log_std,
               baseline=baseline,
               best_frac=0.2,
               action_lt=5.0,
               n_samples=n_samples,
               init_policy=init_policy,  # pass None if policy init is not required
               min_icnn=False,
               sensitivity=False,
               extr_std_scale=0.15,
               std_scale=1.0)  # 1.0: standard cem, 0.0: sensitivity scaled cem

    # n_workers should be 1
    # resume_dir = '/home/shahbaz/Software/garage36/energybased_stable_rl/data/local/experiment/cem_nf_yumi'
    # trainer.restore(resume_dir, from_epoch=17)
    # trainer.resume(n_epochs=50,plot=True)
    trainer.setup(algo, env, n_workers=1, sampler_cls=LocalSampler, worker_class=DefaultWorker)
    trainer.train(n_epochs=100, batch_size=T, plot=False, store_episodes=True)

try:
    cem_nf_yumi(seed=6)
except Exception:
    traceback.print_exc()

# cem_nf_yumi(seed=1)
# init_std = 0.15
# init_log_std = 0.15
# init_policy = None
# policy = DeterministicNormFlowPolicy(env.spec,
#                                 n_flows=2,
#                                 hidden_dim=16,
#                                 init_std=2.,
#                                 K=4.0,
#                                 D=1.0,
#                                  init_func=nn.init.xavier_uniform_,
#                                  init_const=None,
#                                  # init_func=nn.init.constant_,
#                                  # init_const=0.1,
#                                 jac_damping=True)
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
#            extr_std_scale=0.15,
#            std_scale=1.0)  # 1.0: standard cem, 0.0: sensitivity scaled cem
# trainer.setup(algo, env, n_workers=1, sampler_cls=LocalSampler, worker_class=DefaultWorker)
# trainer.train(n_epochs=50, batch_size=T, plot=True, store_episodes=True)
# size="0.023"
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

# cem_nf_yumi(seed=1)1_5
# trainer.train(n_epochs=100, batch_size=T, plot=False, store_episodes=True)

# cem_nf_yumi_1(seed=2)2_5

# cem_nf_yumi_2(seed=3)3_5

# cem_nf_yumi_3(seed=4)4_5

# cem_nf_yumi_4(seed=5)5_5

# cem_nf_yumi_5(seed=6)6_5