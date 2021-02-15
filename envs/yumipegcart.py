import os
import numpy as np
from yumikin.YumiKinematics import YumiKinematics
from energybased_stable_rl.rewards import cart_rwd_func_1
from gym import utils
from gym.envs.mujoco import mujoco_env
from akro.box import Box
from garage import EnvSpec
import pickle

# GOAL 1
GOAL = np.array([-1.50337106, -1.24545874,  1.21963181,  0.46298941,  2.18633697,  1.51383283,
  0.57184653])
# obs in operational space
GOAL_CART = [ 0.46473501,  0.10293446,  0.10217953, -0.00858317,  0.69395054,  0.71995417,
              0.00499788,  0.,          0.,          0.,          0.,          0.,      0.]


# GOAL 2
# GOAL = np.array([-1.63688, -1.22777, 1.28612, 0.446995, 2.21936, 1.57011, 0.47748]) # goal

# list_init_qpos = pickle.load(open("rnd_qpos_init.p", "rb")) # SIGMA_JT = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])*4.
# INIT 1
# INIT = np.array([-1.14, -1.21, 0.965, 0.728, 1.97, 1.49, 0.]) #todo
# INIT = list_init_qpos[4]
# INIT = np.array([-1.38930236, -0.85174226, 1.11545407,  0.57388455,  1.81274445,  1.49625972, 0.09324886]) # nf rnd init 1 of 3
# INIT = np.array([-1.07308613, -1.33489273,  1.10659514,  0.77592789,  2.00738834,  1.65108292, 0.0409894])  # nf rnd init 2 of 3 todo this starts inside the block
INIT = np.array([-0.91912945, -0.93873615,  1.03494441,  0.56895099,  1.69821677,  1.67984028, -0.06353955]) # nf rnd init 3 of 3


# obs in operational space
# [0.43412841 0.16020995 0.17902697 0.00230701 0.45355798 0.89116592
#  0.01015589 0.         0.         0.         0.         0.
#  0.        ]
# INIT = np.array([-1.14762187, -1.09474318,  0.72982478,  0.23000484,  1.7574765,   1.53849862,   0.4464969 ]) # init pos 2
# INIT = np.array([-1.60661071, -0.89088649,  1.0070413,   0.33067306,  1.8419217,   1.66532153, -0.06107046]) # init pos 3

T = 200
dA = 3
dO = 6
dJ = 7
D_rot = np.eye(3)*4
SIGMA = np.array([0.05,0.05,0.01])
SIGMA_JT = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])*2. #todo
rand_init = False
rand_joint_space = False

kin_params_yumi = {}
kin_params_yumi['urdf'] = '/home/shahbaz/Software/yumi_kinematics/yumikin/models/yumi_ABB_left.urdf'
kin_params_yumi['base_link'] = 'world'
# kin_params_yumi['end_link'] = 'left_tool0'
kin_params_yumi['end_link'] = 'left_contact_point'
kin_params_yumi['euler_string'] = 'sxyz'
kin_params_yumi['goal'] = GOAL

class YumiPegCartEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        fullpath = os.path.join(os.path.dirname(__file__), "mujoco_assets", 'yumi_stable_vic_mjcf.xml')
        self.initialized = False
        mujoco_env.MujocoEnv.__init__(self, fullpath, 5)
        self.kinparams = kin_params_yumi
        self.yumikin = YumiKinematics(self.kinparams)
        self.M_rot = np.diag(self.yumikin.get_cart_intertia_d(INIT))[3:]
        self.K_rot = 4*np.eye(3)
        self.D_rot = np.max(np.sqrt(np.multiply(self.M_rot,self.K_rot)))*np.eye(3)
        self.J_Ad_curr = None
        self.initialized = True
        self.action_space = Box(low=-10, high=10, shape=(dA,))
        self.observation_space = Box(low=-2, high=2.0, shape=(dO,))
        self.rest_count = 0
        # self.list_init_qpos = pickle.load(open("rnd_qpos_init.p", "rb"))
        # self.reset_model()

    def step(self, a):
        if self.initialized:
            ex, jx = self._get_obs()
            f_t = a
            f_r = -np.matmul(self.K_rot,ex[3:6]) - np.matmul(self.D_rot,ex[9:])
            f = np.concatenate((f_t,f_r))
            reward, rewards = cart_rwd_func_1(ex, f)
            jtrq = self.J_Ad_curr.T.dot(f)
            assert (jtrq.shape == (dJ,))
            # jtrq = np.zeros(7) # todo
            self.do_simulation(jtrq, self.frame_skip)
            done = False
            # return ex, reward, done, dict(reward_dist=np.sum(rewards[:-1]), reward_ctrl=rewards[-1])
            obs = np.concatenate((ex[:3],ex[6:9]))
            return obs, reward, done, dict({'er': ex[3:6],'erdot': ex[9:],'jx':jx, 'fr': f_r,'jt':jtrq})
        else:
            # a = np.zeros(7) # todo
            self.do_simulation(a, self.frame_skip)
            obs = self._get_obs()
            reward = None
            done = False
            return obs, reward, done, dict(reward_dist=None, reward_ctrl=None)

    def viewer_setup(self):
        # self.viewer.cam.trackbodyid = 0
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 4.0

    def reset(self):
        self.sim.reset()
        obs = self.reset_model()
        return obs

    def reset_model(self):
        init_qpos = INIT
        init_qvel = np.zeros(7)
        if self.initialized:
            del self.yumikin
            self.yumikin = YumiKinematics(self.kinparams)
            if rand_init and not rand_joint_space:
                rot_mat_goal = self.yumikin.Rd
                rot_mat_init = rot_mat_goal
                pose_init_mean_mat = self.yumikin.kdl_kin.forward(INIT)
                trans_init_mean = np.array(pose_init_mean_mat[:3, 3])
                trans_init_mean = trans_init_mean.reshape(-1)
                trans_init_cov = np.diag(SIGMA**2)
                trans_init = np.random.multivariate_normal(trans_init_mean, trans_init_cov)
                # trans_init = trans_init_mean + np.array([0, 0.2, 0])
                pose_init_mat = np.zeros((4,4))
                pose_init_mat[:3, :3] = rot_mat_init
                pose_init_mat[:3, 3] = trans_init
                pose_init_mat[3, 3] = 1.
                init_qpos = self.yumikin.kdl_kin.inverse(pose_init_mat, q_guess=INIT)
                pose_init_mat_new = self.yumikin.kdl_kin.forward(init_qpos)
                inv_kin_err = np.linalg.norm(pose_init_mat * pose_init_mat_new**-1 - np.mat(np.eye(4)))
                if (init_qpos is None) or (inv_kin_err>1e-4):
                    print('IK failed')
                    assert(False)
            elif rand_init and rand_joint_space:
                while True:   # todo
                    init_qpos = np.random.multivariate_normal(INIT, np.diag(SIGMA_JT**2))
                    max_rot = np.max(np.abs(self.yumikin.fwd_pose(init_qpos)[3:]-self.yumikin.goal_cart[3:]))
                    if max_rot < np.pi/4.:
                        break
                # init_qpos = self.list_init_qpos[self.rest_count]
                # self.rest_count += 1
            else:
                None
            self.set_state(init_qpos, init_qvel)
            ex, jx = self._get_obs()
            return np.concatenate((ex[:3],ex[6:9]))
        else:
            self.set_state(init_qpos, init_qvel)
            return self._get_obs()

    def _get_obs(self):
        if self.initialized:
            jx = np.concatenate([
                self.sim.data.qpos.flat,
                self.sim.data.qvel.flat,
            ])
            assert (jx.shape[0] == dJ * 2)
            q = jx[:dJ]
            q_dot = jx[dJ:]
            x_d_e, x_dot_d_e, J_Ad = self.yumikin.get_cart_error_frame_terms(q, q_dot)
            self.J_Ad_curr = J_Ad
            ex = np.concatenate((x_d_e, x_dot_d_e))
            # assert (ex.shape == (dO,))
            return ex, jx
        else:
            obs =  np.concatenate([
                self.sim.data.qpos.flat,
                self.sim.data.qvel.flat,
            ])
            return obs
