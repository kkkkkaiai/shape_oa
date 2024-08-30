import numpy as np
import os
from datetime import datetime

import isaacgym
from shape_oa.envs import *
from shape_oa.utils import get_args, export_policy_as_jit, task_registry, Logger

import torch


def test_env(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs =  min(env_cfg.env.num_envs, 1)

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    env.terrain

    # change_fre = 100
    # mode = False
    # for i in range(int(10*env.max_episode_length)):
    #     actions = 1.0*torch.ones(env.num_envs, env.num_actions, device=env.device)
    #     # get the names of joint for the actions
    #     joint_names = env.dof_names
    #     body_names = env.body_names
    #     # check the dof velocities and 
    #     if i % change_fre == 0:
    #         mode = not mode
    #     if mode:
    #         s_vels = 1.0
    #         w_vels = 0.
    #     else:
    #         s_vels = 0.
    #         w_vels = 1.0
    #     print(body_names)
    #     print(joint_names)
    #     actions[:, 0] = s_vels
    #     actions[:, 1] = w_vels
    #     actions[:, 2] = s_vels
    #     actions[:, 3] = w_vels
    #     actions[:, 4] = s_vels
    #     actions[:, 5] = -w_vels
    #     actions[:, 6] = s_vels
    #     actions[:, 7] = -w_vels
    #     # print(actions)
    #     print(env.contact_forces)
    #     # env.gym.find_actor_rigid_body_handle(env.actor[0], env.actor_handles[0], )
    #     obs, _, rew, done, info = env.step(actions)
    # print("Done")

if __name__ == "__main__":
    args = get_args()
    test_env(args)

