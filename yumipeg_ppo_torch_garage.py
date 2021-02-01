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

    # value_function = LinearFeatureBaseline(env_spec=env.spec)
    value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                              hidden_sizes=(32, 32),
                                              hidden_nonlinearity=torch.tanh,
                                              output_nonlinearity=None)
    N = 100  # number of epochs
    S = 15  # number of episodes in an epoch
    algo = PPO(env_spec=env.spec,
               policy=policy,
               value_function=value_function,
               discount=0.99,
               lr_clip_range=0.1,
               # center_adv=False,
               )

    # resume_dir = '/home/shahbaz/Software/garage36/normflow_policy/data/local/experiment/yumipeg_ppo_garage_31'
    # trainer.restore(resume_dir, from_epoch=98)
    # trainer.resume(n_epochs=100)
    trainer.setup(algo, env, n_workers=4)
    trainer.train(n_epochs=N, batch_size=T*S, plot=True, store_episodes=True)

yumipeg_ppo_garage(seed=2)

# yumipeg_ppo_garage(seed=1)
# yumipeg_ppo_garage_1(seed=2)
# policy = GaussianMLPPolicy(env.spec,
#                            hidden_sizes=[32, 32],
#                            hidden_nonlinearity=torch.tanh,
#                            output_nonlinearity=None,
#                            init_std=2.)
# value_function = LinearFeatureBaseline(env_spec=env.spec)
# N = 100  # number of epochs
# S = 15  # number of episodes in an epoch
# algo = PPO(env_spec=env.spec,
#            policy=policy,
#            value_function=value_function,
#            discount=0.99,
#            lr_clip_range=0.1,
#            # center_adv=False,
#            )
# trainer.setup(algo, env, n_workers=4)
# GOAL = np.array([-1.50337106, -1.24545874,  1.21963181,  0.46298941,  2.18633697,  1.51383283,
#   0.57184653])
# INIT = np.array([-1.14, -1.21, 0.965, 0.728, 1.97, 1.49, 0.])
# T = 200
# rand_init = False
# kin_params_yumi['end_link'] = 'left_contact_point'
# size="0.023"

# yumipeg_ppo_garage_2(seed=1)
# yumipeg_ppo_garage_3(seed=2)
# value_function = GaussianMLPValueFunction(env_spec=env.spec,
#                                               hidden_sizes=(32, 32),
#                                               hidden_nonlinearity=torch.tanh,
#                                               output_nonlinearity=None)