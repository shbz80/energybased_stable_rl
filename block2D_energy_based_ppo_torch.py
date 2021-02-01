import torch
import gym
from garage import wrap_experiment
from garage.envs import GymEnv
from garage.experiment.deterministic import set_seed
from garage.torch.algos import PPO
from energybased_stable_rl.policies.energy_based_control_policy import GaussianEnergyBasedPolicy
from garage.sampler import LocalSampler, RaySampler
from garage.sampler.default_worker import DefaultWorker
from garage.np.baselines import LinearFeatureBaseline
from garage.trainer import Trainer
from energybased_stable_rl.envs.block2D import T
from garage.torch.optimizers import OptimizerWrapper

# @wrap_experiment
@wrap_experiment(snapshot_mode='all')
def block2D_energy_based_ppo(ctxt=None, seed=1):
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

    policy = GaussianEnergyBasedPolicy(env.spec,
                                       icnn_hidden_sizes=(8, 8),
                                       damper_hidden_sizes=(8, 8),
                                       damper_full_mat=True,
                                       init_std=0.5,
                                       full_std=False,
                                       jac_update_rate=5,
                                       init_quad_pot=1.0,
                                       min_quad_pot=1e-3,
                                       max_quad_pot=1e1,
                                       icnn_min_lr=1e-1)
    # policy_optimizer = OptimizerWrapper(
    #     (torch.optim.Adam, dict(lr=2.5e-4)),
    #     policy,
    #     max_optimization_epochs=10,
    #     minibatch_size=64)
    policy_optimizer = OptimizerWrapper(
        (torch.optim.Adam, dict(lr=2.5e-4*10.0)),
        policy,
        max_optimization_epochs=1,
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

    trainer.setup(algo, env, n_workers=S, sampler_cls=LocalSampler, worker_class=DefaultWorker)
    trainer.train(n_epochs=N, batch_size=T*S, plot=True, store_episodes=True)

block2D_energy_based_ppo(seed=1)

# block2D_energy_based_ppo(seed=1)
# policy = GaussianEnergyBasedPolicy(env.spec,
#                                    icnn_hidden_sizes=(8, 8),
#                                    damper_hidden_sizes=(8, 8),
#                                    damper_full_mat=False,
#                                    init_std=0.1,
#                                    full_std=False,
#                                    jac_update_rate=15,
#                                    init_quad_pot=1.0,
#                                    min_quad_pot=1e-3,
#                                    max_quad_pot=1e1,
#                                    icnn_min_lr=1e-1)
# policy_optimizer = OptimizerWrapper(
#     (torch.optim.Adam, dict(lr=2.5e-4*10.0)),
#     policy,
#     max_optimization_epochs=1,
#     minibatch_size=64)
#
# value_function = LinearFeatureBaseline(env_spec=env.spec)
#
# N = 100  # number of epochs
# S = 5  # number of episodes in an epoch
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
# T = 150
# POS_SCALE = 1
# VEL_SCALE = 0.1
# ACTION_SCALE = 1e-3
# v = 2
# w = 1
# TERMINAL_STATE_SCALE = 10
# size="0.05 0.048 0.05"
# timestep="0.02"

# block2D_energy_based_ppo_1(seed=1)
# policy = GaussianEnergyBasedPolicy(env.spec,
#                                        icnn_hidden_sizes=(8, 8),
#                                        damper_hidden_sizes=(8, 8),
#                                        damper_full_mat=True,
#                                        init_std=0.1,
#                                        full_std=True,
#                                        jac_update_rate=5,
#                                        init_quad_pot=1.0,
#                                        min_quad_pot=1e-3,
#                                        max_quad_pot=1e1,
#                                        icnn_min_lr=1e-1)