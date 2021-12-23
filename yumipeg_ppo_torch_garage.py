import torch
import gym
from garage import wrap_experiment
from garage.envs import GymEnv
from garage.experiment.deterministic import set_seed
from garage.torch.algos import PPO
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.trainer import Trainer
from energybased_stable_rl.envs.yumipegcart import T, dA, dO
from garage.np.baselines import LinearFeatureBaseline
from akro.box import Box
from garage import EnvSpec
import numpy as np

# @wrap_experiment
@wrap_experiment(snapshot_mode='all')
def yumipeg_ppo_garage(ctxt=None, seed=1):
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

    policy = GaussianMLPPolicy(env.spec,
                               hidden_sizes=[32, 32],
                               hidden_nonlinearity=torch.tanh,
                               output_nonlinearity=None,
                               init_std=2.)

    value_function = LinearFeatureBaseline(env_spec=env.spec)
    # value_function = GaussianMLPValueFunction(env_spec=env.spec,
    #                                           hidden_sizes=(32, 32),
    #                                           hidden_nonlinearity=torch.tanh,
    #                                           output_nonlinearity=None)
    N = 100  # number of epochs
    S = 15  # number of episodes in an epoch
    algo = PPO(env_spec=env.spec,
               policy=policy,
               value_function=value_function,
               discount=0.99,
               lr_clip_range=0.05,
               # center_adv=False,
               )

    # resume_dir = '/home/shahbaz/Software/garage36/normflow_policy/data/local/experiment/yumipeg_ppo_garage_31'
    # trainer.restore(resume_dir, from_epoch=98)
    # trainer.resume(n_epochs=100)
    trainer.setup(algo, env, n_workers=4)
    trainer.train(n_epochs=N, batch_size=T*S, plot=True, store_episodes=True)

yumipeg_ppo_garage(seed=5)


# yumipeg_ppo_garage(seed=1)1_5
# policy = GaussianMLPPolicy(env.spec,
#                                hidden_sizes=[32, 32],
#                                hidden_nonlinearity=torch.tanh,
#                                output_nonlinearity=None,
#                                init_std=2.)
#
# value_function = LinearFeatureBaseline(env_spec=env.spec)
# N = 100  # number of epochs
# S = 15  # number of episodes in an epoch
# algo = PPO(env_spec=env.spec,
#            policy=policy,
#            value_function=value_function,
#            discount=0.99,
#            lr_clip_range=0.05,
#            # center_adv=False,
#            )
# trainer.setup(algo, env, n_workers=4)
# trainer.train(n_epochs=N, batch_size=T*S, plot=True, store_episodes=True)
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

# yumipeg_ppo_garage_1(seed=2)2_5
# yumipeg_ppo_garage_2(seed=3)3_5
# yumipeg_ppo_garage_3(seed=4)4_5
# yumipeg_ppo_garage_4(seed=5)5_5