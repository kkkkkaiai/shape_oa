import numpy as np
import os
from datetime import datetime

import isaacgym 
from shape_oa.utils import get_args, task_registry
import torch


def retrain(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    train_cfg.runner.resume = True

    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args()
    retrain(args)