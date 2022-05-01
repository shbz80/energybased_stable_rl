# Learning Deep Energy Shaping Policies for Stability-Guaranteed Manipulation
This repo is the code base for the paper _Learning Deep Energy Shaping Policies for Stability-Guaranteed Manipulation_, Khader, S. A., Yin, H., Falco, P., & Kragic, D. (2021), IEEE Robotics and Automation Letters (RA-L). [[IEEE]](https://ieeexplore.ieee.org/abstract/document/9536404) [[arXiv]](https://arxiv.org/abs/2103.16432)

https://www.youtube.com/watch?v=5iwF-_Ecuag

## Paper abstract
Deep reinforcement learning (DRL) has been successfully used to solve various robotic manipulation tasks. However, most of the existing works do not address the issue of control stability. This is in sharp contrast to the control theory community where the well-established norm is to prove stability whenever a control law is synthesized. What makes traditional stability analysis difficult for DRL are the uninterpretable nature of the neural network policies and unknown system dynamics. In this work, stability is obtained by deriving an interpretable deep policy structure based on the energy shaping control of Lagrangian systems. Then, stability during physical interaction with an unknown environment is established based on passivity . The result is a stability guaranteeing DRL in a model-free framework that is general enough for contact-rich manipulation tasks. With an experiment on a peg-in-hole task, we demonstrate, to the best of our knowledge, the first DRL with stability guarantee on a real robotic manipulator.

## Prerequisites
* [garage](https://github.com/rlworkgroup/garage) Deep reinforcement learning toolkit
* [PyTorch](https://pytorch.org/) Deep learning framework
* [MuJoCo](https://mujoco.org/) Physics simulator

## Run experiments

### 2D Block-Insertion
* Run cem_energybased_blocks2d.py for the proposed method.
* Run cem_NF_blocks2d.py for the baseline normalizing flow baseline method.
* Run blocks2D_ppo_torch_garage.py for the standard deep RL baseline method.

### YuMi Peg-In-Hole
* Run cem_energybased_yumi.py for the proposed method.
* Run cem_NF_yumi.py for the baseline normalizing flow baseline method.
* Run yumipeg_ppo_torch_garage.py for the standard deep RL baseline method.
