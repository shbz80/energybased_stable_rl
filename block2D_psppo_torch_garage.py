import torch
import torch.nn as nn
import gym
from garage import wrap_experiment
from garage.envs import GymEnv
from garage.experiment.deterministic import set_seed
from garage.torch.algos import PPO
from energybased_stable_rl.policies.gaussian_ps_mlp_policy import GaussianPSMLPPolicy
from garage.sampler import LocalSampler, RaySampler
from garage.sampler.default_worker import DefaultWorker
from garage.np.baselines import LinearFeatureBaseline
# from garage.torch.value_functions import GaussianMLPValueFunction
from garage.trainer import Trainer
from energybased_stable_rl.envs.block2D import T
from garage.torch.optimizers import OptimizerWrapper

# @wrap_experiment
@wrap_experiment(snapshot_mode='all')
def block2D_psppo_torch_garage(ctxt=None, seed=1):
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

    jac_update_rate = 1
    policy = GaussianPSMLPPolicy(env.spec,
                               hidden_sizes=[16, 16],
                               hidden_nonlinearity=torch.tanh,
                               output_nonlinearity=None,
                               # hidden_w_init=nn.init.xavier_uniform_,
                               # hidden_b_init=nn.init.zeros_,
                               # hidden_b_init=nn.init.xavier_uniform_,
                               init_std=1.,
                               full_std=True,
                               jac_update_rate=jac_update_rate,
                               jac_batch_size = 32)
    # policy_optimizer = OptimizerWrapper(
    #     (torch.optim.Adam, dict(lr=2.5e-4)),
    #     policy,
    #     max_optimization_epochs=10,
    #     minibatch_size=64)
    policy_optimizer = OptimizerWrapper(
        (torch.optim.Adam, dict(lr=2.5e-4)),
        policy,
        max_optimization_epochs=5,
        minibatch_size=64)

    value_function = LinearFeatureBaseline(env_spec=env.spec)

    N = 100  # number of epochs
    S = 15  # number of episodes in an epoch
    algo = PPO(env_spec=env.spec,
               policy=policy,
               policy_optimizer=policy_optimizer,
               value_function=value_function,
               discount=0.995,
               lr_clip_range=0.2,)

    # resume_dir = '/home/shahbaz/Software/garage36/energybased_stable_rl/data/local/experiment/block2D_psppo_torch_garage'
    # trainer.restore(resume_dir, from_epoch=45)
    # trainer.resume(n_epochs=100)
    trainer.setup(algo, env, n_workers=S, sampler_cls=LocalSampler, worker_class=DefaultWorker)
    trainer.train(n_epochs=N, batch_size=T*S, plot=True, store_episodes=True)

block2D_psppo_torch_garage(seed=2)

# block2D_psppo_torch_garage(seed=1)
# block2D_psppo_torch_garage_1
# jac_update_rate = 1
# policy = GaussianPSMLPPolicy(env.spec,
#                              hidden_sizes=[16, 16],
#                              hidden_nonlinearity=torch.tanh,
#                              output_nonlinearity=None,
#                              init_std=3.,
#                              full_std=True,
#                              jac_update_rate=jac_update_rate,
#                              jac_batch_size=64)
# policy_optimizer = OptimizerWrapper(
#     (torch.optim.Adam, dict(lr=2.5e-4)),
#     policy,
#     max_optimization_epochs=5,
#     minibatch_size=64)
#
# value_function = LinearFeatureBaseline(env_spec=env.spec)
#
# N = 50  # number of epochs
# S = 15  # number of episodes in an epoch
# algo = PPO(env_spec=env.spec,
#            policy=policy,
#            policy_optimizer=policy_optimizer,
#            value_function=value_function,
#            discount=0.995,
#            lr_clip_range=0.2,)
#
# trainer.setup(algo, env, n_workers=S, sampler_cls=LocalSampler, worker_class=DefaultWorker)
# trainer.train(n_epochs=N, batch_size=T*S, plot=True, store_episodes=True)
# OFFSET = np.array([0.4, -0.6])
# OFFSET_1 = np.array([0, 0])         # NFPPPO fixed init
# GOAL = np.array([0, 0.5])+OFFSET
# INIT = np.array([-0.3, 0.8])+OFFSET+OFFSET_1
# T = 200
# POS_SCALE = 1
# VEL_SCALE = 0.1
# ACTION_SCALE = 1e-3
# v = 2
# w = 1
# TERMINAL_STATE_SCALE = 10
# size="0.05 0.045 0.05"

# block2D_psppo_torch_garage_2(seed=1)
# jac_update_rate = 1
# policy = GaussianPSMLPPolicy(env.spec,
#                              hidden_sizes=[16, 16],
#                              hidden_nonlinearity=torch.tanh,
#                              output_nonlinearity=None,
#                              init_std=1.,
#                              full_std=True,
#                              jac_update_rate=jac_update_rate,
#                              jac_batch_size=64)

# block2D_psppo_torch_garage_3(seed=1)
# policy_optimizer = OptimizerWrapper(
#     (torch.optim.Adam, dict(lr=2.5e-4)),
#     policy,
#     max_optimization_epochs=10,
#     minibatch_size=64)
# S = 10  # number of episodes in an epoch