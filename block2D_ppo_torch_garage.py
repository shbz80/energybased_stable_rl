import torch
import gym
from garage import wrap_experiment
from garage.envs import GymEnv
from garage.experiment.deterministic import set_seed
from garage.torch.algos import PPO
from garage.torch.policies import GaussianMLPPolicy
from garage.sampler import LocalSampler, RaySampler
from garage.sampler.default_worker import DefaultWorker
from garage.np.baselines import LinearFeatureBaseline
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.trainer import Trainer
from energybased_stable_rl.envs.block2D import T

# @wrap_experiment
@wrap_experiment(snapshot_mode='all')
def block2D_ppo_torch_garage(ctxt=None, seed=1):
    """Train PPO with InvertedDoublePendulum-v2 environment.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.

    """
    set_seed(seed)
    # env = GymEnv(gym.make('Block2D-v0'))
    env = GymEnv('Block2D-v1',max_episode_length=T)
    trainer = Trainer(ctxt)

    policy = GaussianMLPPolicy(env.spec,
                               hidden_sizes=[16, 16],
                               hidden_nonlinearity=torch.tanh,
                               output_nonlinearity=None,
                               init_std=2)

    # value_function = GaussianMLPValueFunction(env_spec=env.spec,
    #                                           hidden_sizes=(32, 32),
    #                                           hidden_nonlinearity=torch.tanh,
    #                                           output_nonlinearity=None)

    value_function = LinearFeatureBaseline(env_spec=env.spec)

    N = 100  # number of epochs
    S = 15  # number of episodes in an epoch
    algo = PPO(env_spec=env.spec,
               policy=policy,
               value_function=value_function,
               discount=0.99,
               lr_clip_range=0.2,)

    trainer.setup(algo, env, n_workers=15, sampler_cls=LocalSampler, worker_class=DefaultWorker)
    # trainer.setup(algo, env, n_workers=4)
    trainer.train(n_epochs=N, batch_size=T*S, plot=True, store_episodes=False)

block2D_ppo_torch_garage(seed=2)

# block2D_ppo_torch_garage(seed=1)
# block2D_ppo_torch_garage_1(seed=2)
# block2D_ppo_torch_garage_2(seed=3)
# policy = GaussianMLPPolicy(env.spec,
#                                hidden_sizes=[16, 16],
#                                hidden_nonlinearity=torch.tanh,
#                                output_nonlinearity=None,
#                                init_std=2)
# value_function = LinearFeatureBaseline(env_spec=env.spec)
# N = 100  # number of epochs
# S = 15  # number of episodes in an epoch
# algo = PPO(env_spec=env.spec,
#            policy=policy,
#            value_function=value_function,
#            discount=0.99,
#            lr_clip_range=0.2,)
#
# trainer.setup(algo, env, n_workers=6)
# trainer.train(n_epochs=N, batch_size=T*S, plot=True, store_episodes=True)
# OFFSET = np.array([0.4, -0.6])
# OFFSET_1 = np.array([0, 0])         # fixed init
# GOAL = np.array([0, 0.5])+OFFSET
# INIT = np.array([-0.3, 0.8])+OFFSET+OFFSET_1
# T = 200
# POS_SCALE = 1
# VEL_SCALE = 0.1
# ACTION_SCALE = 1e-3
# v = 2
# w = 1
# TERMINAL_STATE_SCALE = 10
# SIGMA = np.array([0.05, 0.1])
# size="0.05 0.048 0.05"

# block2D_ppo_torch_garage_3(seed=1)
# block2D_ppo_torch_garage_4(seed=2)
# block2D_ppo_torch_garage_5(seed=3) *best*
# value_function = GaussianMLPValueFunction(env_spec=env.spec,
#                                               hidden_sizes=(16, 16),
#                                               hidden_nonlinearity=torch.tanh,
#                                               output_nonlinearity=None)

# block2D_ppo_torch_garage_6(seed=3)
# value_function = GaussianMLPValueFunction(env_spec=env.spec,
#                                               hidden_sizes=(32, 32),
#                                               hidden_nonlinearity=torch.tanh,
#                                               output_nonlinearity=None)

# block2D_ppo_torch_garage_7(seed=2)
# value_function = GaussianMLPValueFunction(env_spec=env.spec,
#                                               hidden_sizes=(64, 64),
#                                               hidden_nonlinearity=torch.tanh,
#                                               output_nonlinearity=None)

# block2D_ppo_torch_garage_8(seed=1) *best*
# block2D_ppo_torch_garage_9(seed=2)
# value_function = GaussianMLPValueFunction(env_spec=env.spec,
#                                               hidden_sizes=(128, 128),
#                                               hidden_nonlinearity=torch.tanh,
#                                               output_nonlinearity=None)

# block2D_ppo_torch_garage_10(seed=1)
# block2D_ppo_torch_garage_11(seed=2)
# value_function = GaussianMLPValueFunction(env_spec=env.spec,
#                                               hidden_sizes=(256, 256),
#                                               hidden_nonlinearity=torch.tanh,
#                                               output_nonlinearity=None)

# block2D_ppo_torch_garage_12(seed=2)
# block2D_ppo_torch_garage_13(seed=3)
# trainer.setup(algo, env, n_workers=4)
# S = 50
# value_function = GaussianMLPValueFunction(env_spec=env.spec,
#                                               hidden_sizes=(128, 128),
#                                               hidden_nonlinearity=torch.tanh,
#                                               output_nonlinearity=None)

# block2D_ppo_torch_garage_14(seed=1)
# block2D_ppo_torch_garage_15(seed=2) *best*
# trainer.setup(algo, env, n_workers=4)
# S = 50
# value_function = GaussianMLPValueFunction(env_spec=env.spec,
#                                               hidden_sizes=(32, 32),
#                                               hidden_nonlinearity=torch.tanh,
#                                               output_nonlinearity=None)