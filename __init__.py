from gym.envs.registration import register

register(
     id='Block2D-v1',
     entry_point='energybased_stable_rl.envs:Block2DEnv',
     max_episode_steps=1000,
)

register(
    id='YumiPegCart-v1',
    entry_point='energybased_stable_rl.envs:YumiPegCartEnv',
    max_episode_steps=1000,
)