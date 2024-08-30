from time import time
import numpy as np
import os

from isaacgym.torch_utils import *
# from isaacgym import LEGGED_GYM_ROOT_DIR
from shape_oa.envs.base.shape_robot import ShapeRobot

from .flat.swerve_flat_config import SwerveFlatCfg


class Swerve(ShapeRobot):
    cfg : SwerveFlatCfg
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        pass
        # load actuator network
        # if self.cfg.control.use_actuator_network:
        #     actuator_network_path = self.cfg.control.actuator_net_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        #     self.actuator_network = torch.jit.load(actuator_network_path).to(self.device)


    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self.sea_hidden_state_per_env[:, env_ids] = 0.
        self.sea_cell_state_per_env[:, env_ids] = 0.

    def _init_buffers(self):
        super()._init_buffers()
        # Additionally initialize actuator network hidden state tensors
        self.sea_input = torch.zeros(self.num_envs*self.num_actions, 1, 2, device=self.device, requires_grad=False)
        self.sea_hidden_state = torch.zeros(2, self.num_envs*self.num_actions, 8, device=self.device, requires_grad=False)
        self.sea_cell_state = torch.zeros(2, self.num_envs*self.num_actions, 8, device=self.device, requires_grad=False)
        self.sea_hidden_state_per_env = self.sea_hidden_state.view(2, self.num_envs, self.num_actions, 8)
        self.sea_cell_state_per_env = self.sea_cell_state.view(2, self.num_envs, self.num_actions, 8)

    def _compute_torques(self, actions):
        # Choose between pd controller and actuator network
        # if self.cfg.control.use_actuator_network:
        #     with torch.inference_mode():
        #         self.sea_input[:, 0, 0] = (actions * self.cfg.control.horizontal_scale + self.default_dof_pos - self.dof_pos).flatten()
        #         self.sea_input[:, 0, 1] = self.dof_vel.flatten()
        #         torques, (self.sea_hidden_state[:], self.sea_cell_state[:]) = self.actuator_network(self.sea_input, (self.sea_hidden_state, self.sea_cell_state))
        #     return torques
        # else:
        # pd controller
        return super()._compute_torques(actions)    