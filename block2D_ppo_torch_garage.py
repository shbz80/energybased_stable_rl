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

    # trainer.setup(algo, env, n_workers=15, sampler_cls=LocalSampler, worker_class=DefaultWorker)
    trainer.setup(algo, env, n_workers=4)
    trainer.train(n_epochs=N, batch_size=T*S, plot=True, store_episodes=True)

block2D_ppo_torch_garage(seed=3)

# block2D_ppo_torch_garage(seed=2)1_5
# pos 2
# policy = GaussianMLPPolicy(env.spec,
#                                hidden_sizes=[16, 16],
#                                hidden_nonlinearity=torch.tanh,
#                                output_nonlinearity=None,
#                                init_std=3)
# value_function = LinearFeatureBaseline(env_spec=env.spec)
# N = 100  # number of epochs
# S = 15  # number of episodes in an epoch
# algo = PPO(env_spec=env.spec,
#            policy=policy,
#            value_function=value_function,
#            discount=0.99,
#            lr_clip_range=0.2,)
# trainer.setup(algo, env, n_workers=4)
# trainer.train(n_epochs=N, batch_size=T*S, plot=True, store_episodes=True)
# OFFSET = np.array([0.4, -0.6])
# OFFSET_1 = np.array([0, 0])         # NFPPPO fixed init
# GOAL = np.array([0, 0.5])+OFFSET
# INIT = np.array([0.0, -0.1])+OFFSET+OFFSET_1
# T = 200
# POS_SCALE = 1
# VEL_SCALE = 0.1
# ACTION_SCALE = 1e-3
# v = 2
# w = 1
# TERMINAL_STATE_SCALE = 10
# size="0.05 0.048 0.05"

# block2D_ppo_torch_garage_1(seed=1)
# pos 1
# block2D_ppo_torch_garage_1(seed=4)3_5

# block2D_ppo_torch_garage_2(seed=5)4_5

# block2D_ppo_torch_garage_3(seed=2)

# block2D_ppo_torch_garage_4(seed=1)

# block2D_ppo_torch_garage_5(seed=6)

# block2D_ppo_torch_garage_6(seed=7)

# block2D_ppo_torch_garage_7(seed=8)

