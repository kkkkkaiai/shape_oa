
from time import time
import numpy as np
import random
import os
import copy

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch

from shape_oa import SHAPE_OA_ROOT_DIR
from shape_oa.envs.base.base_task import BaseTask
from .shape_config import ShapeRobotCfg
from .shape_obstacle import DistanceCheck
from shape_oa.utils.math import quat_apply_yaw, wrap_to_pi
from shape_oa.utils.terrain import Terrain

from shape_oa.utils.helpers import class_to_dict

class ShapeRobot(BaseTask):
    def __init__(self, cfg: ShapeRobotCfg, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        # 用于解析配置文件
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)

        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render()
        # decimation的意思是每隔几个step进行一次物理仿真，这样可以减少计算量
        for _ in range(self.cfg.control.decimation):
            # self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            # self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.set_dof_velocity_target_tensor(self.sim, gymtorch.unwrap_tensor(self.actions))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        # 在每个step之后，进行一些后处理
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        # clipped obs是观测值，clipped states是状态，rewards是奖励，dones是是否结束，infos是额外信息
        # 观测值用于训练，状态用于调试，奖励用于训练，是否结束用于判断是否需要重置环境，额外信息用于记录一些额外的信息
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        # print(self.dof_vel)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras
    
    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_pos[:] = self.root_states[:, :3]

        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self._post_physics_step_callback()
        
        # compute distance to obstacle
        if self.cfg.asset.obstacle_enabled:
            self._calculate_obstacle_distance()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        # env_ids是需要重置的环境的id
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        # observations是用于训练的观测值，它包括了机器人的状态，动作，命令等信息
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        # if self.viewer and self.enable_viewer_sync and self.debug_viz:
        #     self._draw_debug_vis()
    
    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 0.01, dim=1)
        # 如果reset_buf中有不为0的值，就打印
        # if torch.any(self.reset_buf):
        #     print('test', self.reset_buf, self.contact_forces[:, self.termination_contact_indices, :])
        #     print(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1))
        #     print(self.reset_buf.nonzero(as_tuple=False).flatten())
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf
        
    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        # if self.cfg.terrain.curriculum:
        #     self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)
        
        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self._resample_commands(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        # curriculum是用来逐渐增加难度的，比如逐渐增加机器人的速度，逐渐增加地形的难度等
        # if self.cfg.terrain.curriculum:
        #     self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
    
    def compute_observations(self):
        """ Computes observations
        """
        # buf中包括机器人的线速度，角速度，重力，命令，关节位置，关节速度，动作

        self.obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    self.obstacle_info
                                    ),dim=-1)
        # add perceptive inputs if not blind
        # if self.cfg.terrain.measure_heights:
        #     heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
        #     self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def create_sim(self):
        """ Creates simulation, and environments
        """
        self.up_axis_idx = 2
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type=='plane':
            # use plane as environment floor
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")

        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
    
    #------------------ Callbacks ------------------#
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        # env_id为0时，将关节的位置，速度，力矩限制存储到dof_pos_limits, dof_vel_limits, torque_limits中
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        """
        用来给每个个体的刚体的质量添加一些随机噪声，来提高模型的泛化能力
        """
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            props[0].mass += np.random.uniform(rng[0], rng[1])
        return props

    def _calculate_obstacle_distance(self):
        """ Compute distance to obstacles
        """
        # 先简单计算所有机器人到障碍物的距离，然后再使用sdf计算
        # self.base_pos - 机器人的位置 | self.obstacle_pos - 障碍物的位置
        # print(self.base_pos, self.obstacle_pos)
        dists = torch.norm(self.base_pos.unsqueeze(1) - self.obstacle_pos, dim=-1)
        # 找出每一行中最小距离的index
        min_dists, min_indices = torch.min(dists, dim=1)
        # 获取每一列中最小距离小于1m的index
        close_indices = (min_dists < self.cfg.asset.obs_dis_preset).nonzero(as_tuple=False).flatten() # distance thresh

        # 重置
        self.obstacle_info = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        
        self.obstacle_info[:, 0] = self.cfg.asset.obs_dis_preset * \
                                    torch.ones((self.num_envs), dtype=torch.float, device=self.device)
        self.obstacle_info[:, 1:4] = self.base_lin_vel

        # t1 = time.time()
        for i in range(len(close_indices)):
            # t2 = time.time()
            idx = close_indices[i]

            robot_shape = {
                'position': self.base_pos[idx],
                'orientation': self.base_quat[idx],
            }

            obstacle_shape = {
                'name': self.obstacle_name[self.obstacle_type[min_indices[idx]]],
                'position': self.obstacle_pos[min_indices[idx]],
                'orientation': torch.tensor([0,0,0,1], device=self.device),
            }

            obs_info = self.dis_check.check(obstacle_shape, robot_shape, if_tensor=True)
            robot_info = self.dis_check.check(obstacle_shape, robot_shape, reverse=True, if_tensor=True) 

            # 距离 = 机器人位置到障碍物的sdf距离 + 障碍物位置到机器人的sdf距离 - 两个中心位置的距离
            self.obstacle_info[idx, 0] = obs_info[0] + robot_info[0] - min_dists[idx]
            # 梯度使用的是障碍物的梯度
            self.obstacle_info[idx, 1:4] = obs_info[1]
            # print(f"Time taken for one distance check: {time.time() - t2}")
        # print(f"Time taken for all distance checks: {time.time() - t1}")


    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        # if self.cfg.terrain.measure_heights:
        #     self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        if control_type=="P":
            torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
                                    
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            # temp_ids = copy.copy(env_ids)
            # for i in range(len(env_ids)):
            #     if env_ids[i] in self.random_idx:
            #         # 随机从self.random_idx_inv中选择一个索引作为初始位置
            #         temp_ids[i] = random.choice(self.random_idx_inv)
            self.root_states[env_ids, :3] += self.env_origins[self.log_env_idx[env_ids]]
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        # print('1111111', env_ids_int32)
        # 给self.root_states添加障碍物的状态，障碍物不发生移动。该处主要是为了避免生成红色的报错提示信息
        # self.temp_states = gymtorch.wrap_tensor(self.gym.acquire_actor_root_state_tensor(self.sim))
        # self.temp_states[:self.num_envs] = self.root_states
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))


    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)


    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0. # commands
        noise_vec[12:24] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[24:36] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[36:43] = 0. # previous actions
        # if self.cfg.terrain.measure_heights:
        #     noise_vec[48:235] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements
        return noise_vec


    # -----------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
         # 获取机器人的root系的状态
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        # 获取机器人的关节状态
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        # 获取机器人的接触力
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        # 刷新机器人的状态
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)[:self.num_envs,:]
        # 关节状态包括位置和速度
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        self.base_pos = self.root_states[:, :3]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces)[:self.num_envs*self.num_bodies,:].view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        # 加入障碍物检测的部分，需要知道到障碍物的距离和梯度，在这里初始化每个actor的buffer
        if self.cfg.asset.obstacle_enabled:
            self.dis_check = DistanceCheck()
        # 如果没有学习避开障碍物，这个buffer会一直为0    
        self.obstacle_info = self.cfg.asset.obs_dis_preset * torch.ones((self.num_envs, 4)).to(self.device)

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        # 重力方向，指的是z轴的方向
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        # 足步置空的时间
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        # 将机器人的重力投影到机器人的root系
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        # if self.cfg.terrain.measure_heights:
        #     self.height_points = self._init_height_points()
        self.measured_heights = 0

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = 1.0
        plane_params.dynamic_friction = 1.0
        plane_params.restitution = 0.
        self.gym.add_ground(self.sim, plane_params)

    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows 
        hf_params.transform.p.x = -self.terrain.cfg.border_size 
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)   
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        
        asset_path = self.cfg.asset.file.format(SHAPE_OA_ROOT_DIR=SHAPE_OA_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        self.body_names = body_names
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        penalized_friction_names = []
        for name in self.cfg.asset.penalize_friction_on:
            penalized_friction_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        
        self.envs = []

        if self.cfg.asset.obstacle_enabled:
            # random choose the position of the obstacle according to the env_origins
            # 重新调整障碍物的数量，如果障碍物的数量超过了环境的数量，将障碍物的数量调整为环境数量的70%
            if self.cfg.asset.num_obstacles > self.num_envs:
                self.cfg.asset.num_obstacles = int(self.num_envs * 0.7)
                print('The number of obstacles is too large, it has been adjusted to 70% of the number of environments')
            # 非重复的生成随机数
            
            self.random_idx = random.sample(range(self.num_envs), self.cfg.asset.num_obstacles)
            self.random_idx = torch.tensor(self.random_idx, dtype=torch.int32, device=self.device)
            
            # 生成一个与random_idx相反的数组
            self.random_idx_inv = torch.tensor([i for i in range(self.num_envs) if i not in self.random_idx], dtype=torch.int32, device=self.device)

        # 记录所有机器人的env_idx
        self.log_env_idx = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            
            temp_idx = i
            if self.cfg.asset.obstacle_enabled and i in self.random_idx:
                # 如果产生位置与障碍物重叠，则随机从random_idx_inv中选择一个位置
                temp_idx = random.choice(self.random_idx_inv)

            self.log_env_idx[i] = temp_idx

            pos = self.env_origins[temp_idx].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
                
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            # print(dof_props_asset['lower'], dof_props_asset['upper'], dof_props_asset['velocity'], dof_props_asset['effort'])
            if(self.cfg.asset.default_dof_drive_mode == gymapi.DOF_MODE_VEL):
                dof_props['driveMode'].fill(gymapi.DOF_MODE_VEL)
                dof_props['stiffness'].fill(0.0)
                dof_props['damping'].fill(300)
            # print(dof_props['stiffness'])
            
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        if self.cfg.asset.obstacle_enabled:
            self.obs_env = []
            self.obstacle_handlers = []

            obs_root = self.cfg.asset.obstacle_root.format(SHAPE_OA_ROOT_DIR=SHAPE_OA_ROOT_DIR)
            for i in range(len(self.cfg.asset.obstacle_file)):
                if 'triangle' in self.cfg.asset.obstacle_file[i]:
                    triangle_asset_path = os.path.join(obs_root, self.cfg.asset.obstacle_file[i])
                if 'pentagon' in self.cfg.asset.obstacle_file[i]:
                    pentagon_asset_path = os.path.join(obs_root, self.cfg.asset.obstacle_file[i])
                if 'rect' in self.cfg.asset.obstacle_file[i]:
                    rect_asset_path = os.path.join(obs_root, self.cfg.asset.obstacle_file[i])
                if 'cylinder' in self.cfg.asset.obstacle_file[i]:
                    cylinder_asset_path = os.path.join(obs_root, self.cfg.asset.obstacle_file[i])
        
            # os.path.basename
            obstacle_asset_options = gymapi.AssetOptions()
            obstacle_asset_options.fix_base_link = True # obstacle should not move
            obstacle_asset_options.disable_gravity = False
            obstacle_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

            # asset_path = os.path.join(obs_root, self.cfg.asset.obstacle_file[0])
            # asset_obstacle_root = os.path.dirname(asset_path)
            # asset_obstacle_file = os.path.basename(asset_path)

            triangle_asset_path = os.path.join(obs_root, triangle_asset_path)
            triangle_obstacle_asset = self.gym.load_asset(self.sim,
                                                          os.path.dirname(triangle_asset_path),
                                                          os.path.basename(triangle_asset_path),
                                                          obstacle_asset_options)
            
            pentagon_asset_path = os.path.join(obs_root, pentagon_asset_path)
            pentagon_obstacle_asset = self.gym.load_asset(self.sim,
                                                          os.path.dirname(pentagon_asset_path),
                                                          os.path.basename(pentagon_asset_path),
                                                          obstacle_asset_options)
            rect_asset_path = os.path.join(obs_root, rect_asset_path)
            rect_obstacle_asset = self.gym.load_asset(self.sim,
                                                      os.path.dirname(rect_asset_path),
                                                      os.path.basename(rect_asset_path),
                                                      obstacle_asset_options)
            cylinder_asset_path = os.path.join(obs_root, cylinder_asset_path)
            cylinder_obstacle_asset = self.gym.load_asset(self.sim,
                                                          os.path.dirname(cylinder_asset_path),
                                                          os.path.basename(cylinder_asset_path),
                                                          obstacle_asset_options)
            
            triangle_rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(triangle_obstacle_asset)
            pentagon_rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(pentagon_obstacle_asset)
            rect_rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(rect_obstacle_asset)
            cylinder_rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(cylinder_obstacle_asset)


            # obstacle_asset = self.gym.load_asset(self.sim, asset_obstacle_root, asset_obstacle_file, obstacle_asset_options)
            # consider the obstalce is a rigid body and not update its state
            # obstacle_rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(obstacle_asset)
            self.obs_asset = [triangle_obstacle_asset, 
                              pentagon_obstacle_asset, 
                              rect_obstacle_asset, 
                              cylinder_obstacle_asset]
            

            self.obs_props_asset = [triangle_rigid_shape_props_asset, 
                                    pentagon_rigid_shape_props_asset, 
                                    rect_rigid_shape_props_asset, 
                                    cylinder_rigid_shape_props_asset]

            self.obstacle_name = ['triangle', 'pentagon', 'rect', 'cylinder']
            self.obstacle_type = [] # 保存障碍物的类型
            self.obstacle_pos = []  # 保存障碍物的位置
            self.obstacle_orientation = []

            obs_visable = True

            for i in range(self.cfg.asset.num_obstacles):
                pos = self.env_origins[self.random_idx[i]].clone()
                # pos = torch.Tensor([3,3,0]).to(self.device)
                # add random noise to the position
                # pos[:2] += 5.0 * torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
                start_pose.p = gymapi.Vec3(*pos)
                rand_idx = torch.randint(0, 4, (1,))

                # self.gym.set_asset_rigid_shape_properties(obstacle_asset, obstacle_rigid_shape_props_asset)
                # create the obstacles that can collide with all the robots
                self.obstacle_type.append(rand_idx.item())
                self.obstacle_pos.append(pos.cpu().numpy())

                if obs_visable:
                    env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
                    self.gym.set_asset_rigid_shape_properties(self.obs_asset[rand_idx.item()], self.obs_props_asset[rand_idx.item()])
                    obstalce_handle = self.gym.create_actor(env_handle, self.obs_asset[rand_idx.item()], start_pose, 'obstacle', -1, -1, 0)
                    self.obs_env.append(env_handle)
                    self.obstacle_handlers.append(obstalce_handle)

            self.obstacle_pos = torch.tensor(self.obstacle_pos).to(self.device)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])
        
        self.penalised_friction_indices = torch.zeros(len(penalized_friction_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_friction_names)):
            self.penalised_friction_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_friction_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        # 这里没有用到其它的地形，所以直接创建一个网格
        # if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
        #     self.custom_origins = True
        #     self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        #     # put robots at the origins defined by the terrain
        #     max_init_level = self.cfg.terrain.max_init_terrain_level
        #     if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
        #     self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
        #     self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
        #     self.max_terrain_level = self.cfg.terrain.num_rows
        #     self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
        #     self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        # else:
        self.custom_origins = False
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        # create a grid of robots
        num_cols = np.floor(np.sqrt(self.num_envs))
        num_rows = np.ceil(self.num_envs / num_cols)
        xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
        spacing = self.cfg.env.env_spacing
        self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
        self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
        self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        # 观测的尺度
        self.obs_scales = self.cfg.normalization.obs_scales
        # 奖励的尺度，用于计算奖励。
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        # 控制指令的范围
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        # if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
        # self.cfg.terrain.curriculum = False # TODO delete
        # 重置环境的时候，是否随机推动机器人
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)
        # 机器人的动作空间
        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

    # def _draw_debug_vis(self):
    #     """ Draws visualizations for dubugging (slows down simulation a lot).
    #         Default behaviour: draws height measurement points
    #     """
    #     # draw height lines
    #     if not self.terrain.cfg.measure_heights:
    #         return
    #     self.gym.clear_lines(self.viewer)
    #     self.gym.refresh_rigid_body_state_tensor(self.sim)
    #     sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
    #     for i in range(self.num_envs):
    #         base_pos = (self.root_states[i, :3]).cpu().numpy()
    #         heights = self.measured_heights[i].cpu().numpy()
    #         height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
    #         for j in range(heights.shape[0]):
    #             x = height_points[j, 0] + base_pos[0]
    #             y = height_points[j, 1] + base_pos[1]
    #             z = heights[j]
    #             sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
    #             gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 

    #------------ reward functions----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.square(base_height - self.cfg.rewards.base_height_target)

    # def _reward_torques(self):
    #     # Penalize torques
    #     return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_pos(self):
        # Penalize dof positions
        return torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_friction(self):
        # Penalize high friction by the contact forces, when the contact forces are small, the friction is high
        # When other wheels have contact, the wheel without contact is penalized. 
        # Because the direction of its wheels may be inconsistent with other wheels, causing friction.
        contact_forces = self.contact_forces[:, self.penalised_friction_indices, :]
        # check if there are non-zero contact forces
        contact = torch.norm(contact_forces, dim=-1) >= 1.0
        contact_sum = torch.sum(contact, dim=1)
        return torch.sum(contact_sum!=4)

    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    # def _reward_torque_limits(self):
    #     # penalize torques too close to the limit
    #     return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime

    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)
    
    # def _reward_feet_contact_forces(self):
    #     # penalize high contact forces
    #     return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

    def _reward_obstacle_dis(self):
        # Penalize the distance to the obstacle, the distance is less, the penalty is greater
        return torch.exp(-(self.obstacle_info[:, 0]-self.cfg.asset.obs_dis_preset))-1

    def _reward_obstacle_grad(self):
        # penalize the robot move direction and the obstacle gradient direction are not consistent
        # 计算base_lin_vel的方向和obstacle_grad的方向，并计算两个的夹角
        dot_product = torch.sum(self.base_lin_vel[:, :2] * self.obstacle_info[:, 1:3], dim=1)
        theta = torch.acos(dot_product / ((torch.norm(self.base_lin_vel[:, :2], dim=1) * torch.norm(self.obstacle_info[:, 1:3], dim=1))))
        # 当距离越小且夹角越大时，惩罚越大
        theta[torch.isnan(theta)] = 0
        # theta[0] = 1.0
        # self.obstacle_info[0, 0] = 0.1
        theta *= (torch.exp(-(self.obstacle_info[:, 0]-self.cfg.asset.obs_dis_preset))-1)
        # print(torch.exp(-theta)-1.0)

        return torch.exp(-theta)-1.0
