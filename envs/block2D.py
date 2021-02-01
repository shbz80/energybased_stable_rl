import os

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

############final values for the block exp#############
# GOAL = np.array([0, 0.5])
# INIT = np.array([-0.3, 0.8]) # pos1
# INIT = np.array([0.0, -0.1]) # pos2
# INIT = np.array([0.5, 0.3]) # pos3
# assumes <body name="blockx" pos="0.4 -0.6 0"> in xml
############final values for the block exp#############

#assumes <body name="blockx" pos="0 0 0"> in xml
OFFSET = np.array([0.4, -0.6])
# OFFSET_1 = np.array([0, -0.1])  # NFPPPO random init
OFFSET_1 = np.array([0, 0])         # NFPPPO fixed init
GOAL = np.array([0, 0.5])+OFFSET
# INIT = np.array([-0.3, 0.8])+OFFSET # pos1
INIT = np.array([-0.3, 0.8])+OFFSET+OFFSET_1
# INIT = np.array([0.0, -0.1])+OFFSET # pos2
# INIT = np.array([0.5, 0.3])+OFFSET # pos3

# ACTION_SCALE = 1e-3
# STATE_SCALE = 1
# TERMINAL_SCALE = 100
# EXP_SCALE = 2.
T = 200
POS_SCALE = 1
VEL_SCALE = 0.1
ACTION_SCALE = 1e-3
v = 2
w = 1
TERMINAL_STATE_SCALE = 10

SIGMA = np.array([0.05, 0.1]) # NFPPO

def cart_rwd_shape_1(d, v=1, w=1):

    alpha = 1e-5
    d_sq = d.dot(d)
    r = w*d_sq + v*np.log(d_sq + alpha) - v*np.log(alpha)
    assert (r >= 0)
    return r

def cart_rwd_func_1(x, f, terminal=False):
    '''
    This is for a regulation type problem, so x needs to go to zero.
    Magnitude of f has to be small
    :param x:
    :param f:
    :param g:
    :return:
    '''
    assert(x.shape==(4,))
    assert(f.shape==(2,))

    x_pos = x[:2]
    x_vel = x[2:]

    dx_pos = cart_rwd_shape_1(x_pos, v=v, w=w)
    dx_vel = x_vel.dot(x_vel)
    du = f.dot(f)

    reward_pos = -POS_SCALE*dx_pos
    if terminal:
        reward_pos = TERMINAL_STATE_SCALE * reward_pos

    reward_vel = -VEL_SCALE*dx_vel

    reward_state = reward_pos + reward_vel
    reward_action = -ACTION_SCALE*du
    reward = reward_state + reward_action
    rewards = np.array([reward_pos, reward_vel, reward_action])

    return reward, rewards


class Block2DEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)

        self.t = 0

        fullpath = os.path.join(os.path.dirname(__file__), "mujoco_assets", 'block2D.xml')
        mujoco_env.MujocoEnv.__init__(self, fullpath, 1)
        
        self.reset_model()

    def step(self, a):
        # self.do_simulation(a, self.frame_skip)
        # obs = self._get_obs()
        # obs[:2] -= GOAL
        # dist = obs[:2]
        # reward_dist = -STATE_SCALE * np.linalg.norm(dist)
        # reward_ctrl = -ACTION_SCALE * np.square(a).sum()
        # reward = reward_dist + reward_ctrl
        # done = False
        # self.t+=1
        # if self.t >= T:
        #     done = True
        # return obs, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

        self.do_simulation(a, self.frame_skip)
        obs = self._get_obs()
        state = obs     # state is linked to obs not a copy #todo

        state[:2] = state[:2] - GOAL
        reward, rewards = cart_rwd_func_1(state, a)
        done = False
        self.t+=1
        if self.t >= T:
            reward, rewards = cart_rwd_func_1(obs, a, terminal=True)
            done = True
        return obs, reward, done, dict(reward_dist=np.sum(rewards[:2]), reward_ctrl=rewards[2])

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        # self.viewer.cam.trackbodyid = -1
        # self.viewer.cam.distance = 4.0

    def reset_model(self):
        var = SIGMA ** 2
        cov = np.diag(var * np.ones(2))
        mu = INIT
        # init_qpos = np.random.multivariate_normal(mu, cov)
        init_qpos = INIT
        init_qvel = np.zeros(2)
        self.set_state(init_qpos, init_qvel)
        self.t = 0
        obs = self._get_obs()
        obs[:2] -= GOAL
        return obs

    def _get_obs(self):
        return np.concatenate([
            # self.model.data.qpos.flat[:7],
            # self.model.data.qvel.flat[:7],
            self.sim.data.qpos.flat[:2],
            self.sim.data.qvel.flat[:2],
            # self.get_body_com("blocky")[:2],
        ])