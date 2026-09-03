# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Amal Dev Haridevan

from __future__ import annotations

import typing

import gymnasium as gym
import gzdrl
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

class HoverEnv(gym.Env):
    _envids = []
    _sdf_file =  "world_hover.sdf"
        
    def __init__(self, N_envs: int, envid: int = 0) -> None:
        """
        Initialize the HoverEnv
        
        This initializes the hover task with ``gzdrl.AsyncDRLServerPool``.
        Native worker threads execute the selected servers in parallel while
        the binding releases the Python GIL.
        
        Parameters
        ----------
        N_envs : int
            Number of batched envs to run. This is similar to the envpool API
        envid : int
            A unique envid to properly partition GZ_SERVER. This is not required to be unique, if no visualization
            is desired.
            
        Examples
        --------
        >>> env = HoverEnv(10, 0)
        """
        super().__init__()
        envid = int(envid)
        while envid in HoverEnv._envids:
            envid += 1
            
        self.envid = envid
        HoverEnv._envids.append(envid)
        
        """setup the environment"""
        partition = f"{envid}"
        sdf_file = str(gzdrl.get_sdf_path(HoverEnv._sdf_file))
        """Name as specified in the sdf file"""
        self.uav_name = "quadrotor"
        self.uav_link_name = "quadrotor/base_link"
        self.server = gzdrl.AsyncDRLServerPool(N_envs, partition, sdf_file, [self.uav_name], False)
        self.batch = N_envs
        # action buffer
        self.action_buffer1 = np.zeros((self.batch, 4)) 
        self.action_buffer2 = np.zeros((self.batch, 4))
        self.max_steps = 10000
        self.current_step = np.zeros((self.batch))
        self.done = np.zeros((self.batch), dtype=bool)
        self.maximum_bounds  = np.array((7.0, 7.0, 7.0))
        self.position_spawn_bound_mean = (np.array((-5.0, -5.0, 0.0))+ np.array((5.0, 5.0, 3.0)))/2
        self.position_spawn_bound_diff = (-np.array((-5.0, -5.0, 0.0))+ np.array((5.0, 5.0, 3.0)))/2
        self.yaw_mean = 0
        self.yaw_diff = 0.72
        self.desired_pos = np.zeros((self.batch, 3,))
        self.desired_yaw =  np.zeros((self.batch,))
        self.max_yaw = 1.57
        """define action and observation spaces"""
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4, ), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(22, ), dtype=np.float32)
        # cache
        self.control_states = None 
        self.cache_vars = {}
        # obs cache to prveent re-alloc
        self.obs_cache = np.zeros((self.batch, 22), dtype=np.float32)
        # RNG
        self.rng = np.random.default_rng(seed=self.envid)
        # cache envids and model names 
        self.env_ids = list(range(self.batch))
        self.uav_b =["quadrotor"]*self.batch
        self.link_b = ["quadrotor/base_link"]*self.batch
        self.uav_states = None
        self.reset(self.env_ids)


    def step(self, action: NDArray) -> typing.Tuple[NDArray, float, bool, bool, typing.Dict]:
        """Steps the vectorized environemnt using AsyncDRLServer API, and returns the
        vectorized obs, reward, done, trunc, and info

        Parameters
        ----------
        action : NDArray
            Actions for the agents. The shape of action should be [N_envs, ....]

        Returns
        -------
        typing.Tuple[NDArray, float, bool, bool, typing.Dict]
            Returns [N_envs, ...] obs, N_envs rewards, N_envs done and trunc, N_envs infos
        """
        self.current_step += 1
        action[np.isnan(action)] = 0.0
        action = ((action) + 1.0)*2300.0 /2.0
        self.server.set_rotor_velocity_cmd(self.env_ids, self.uav_b, self.link_b, action)
        self.server.run_once(self.env_ids)
        self.server.update_control_states(self.env_ids)
        self.action_buffer2 = self.action_buffer1
        self.action_buffer1 = action 
        obs, reward, terminated = self.get_obs_reward_done()
        truncated = self.current_step >= self.max_steps
        # self.make_obs()
        return self.obs_cache.copy(), reward, terminated, truncated, {}

    def get_obs_reward_done(self) ->typing.Tuple[NDArray, float, bool]:
        """Helper method for obtaining vectorized obs, reward and info. See step for
        details on the dimensions.

        Returns
        -------
        typing.Tuple[NDArray, float, bool]
            Obs, rew, and info
        """
        self.control_states  =  self.server.get_control_states(self.env_ids)
        self.uav_states = np.array([self.control_states[i]["quadrotor"]["quadrotor/base_link"][0] for i in range(self.batch)])

        current_pos = self.uav_states[:, :3]
        relative_pos = self.desired_pos - current_pos
        omega = self.uav_states[:, 10:13]
        quat_ = self.uav_states[:, 6:10]
        rot = Rotation.from_quat(quat_, scalar_first=True)
        current_yaw = rot.as_euler('ZYX')[:, 0]
        yaw_error = self.desired_yaw - current_yaw

        state_err =  1.6*1.6*(np.linalg.norm(relative_pos, axis=1)**2 + yaw_error*yaw_error);
        reward_pos = 1.0 / (1.0 + state_err);
        reward_omega = 0.01 / (1.0 + np.linalg.norm(omega, axis=1)**2)
        # reward_effort = np.clip(-1e-6*(np.linalg.norm(self.action_buffer[-1])), -1.0, 1.0)
        # act1 = self.action_buffer[-1]
        # act2 = self.action_buffer[-2] if len(self.action_buffer) > 1 else act1 
        # reward_smooth = np.clip(-1e-6*(np.linalg.norm(act2-act1)), -1.0, 1.0);
        yaw_oob_cost = 0.0;
        done = np.zeros(self.batch, dtype=bool)
        done[np.abs(current_yaw) > self.max_yaw] = True 

        bounds_err = self.maximum_bounds - current_pos
        done[np.any(np.abs(current_pos)-self.maximum_bounds > 0, axis=1)  >0.0 ] = True
        rotmat =rot.as_matrix()
        cosang = np.abs(np.arccos(np.clip(rotmat[:, 2,2], -1.0, 1.0)))
        
        done[cosang>1.57] = True
        self.done = done.copy()
        for i in range(3):
            self.obs_cache[:, i] = relative_pos[:, i]
        self.obs_cache[:, 3] = yaw_error
        for i in range(4, 14):
            self.obs_cache[:, i] = self.uav_states[:, i-1]
        
        self.obs_cache[:, 14 : 18 ] = self.action_buffer1/1300.0 -1.0
        self.obs_cache[:, 18 : 22 ] = self.action_buffer2/1300.0 -1.0
        total_rew = reward_pos + reward_omega 
        return self.obs_cache.copy(),  total_rew, done
    
    def reset(self, reset_idxs: NDArray=None, seed: int=None, **kwargs) ->typing.Tuple[NDArray, typing.Dict]:
        """Resets the environments specified by the index @reset_idxs. If reset_idxs is None
        all the environments will be reset.

        Parameters
        ----------
        reset_idxs : NDArray, optional
            Indexes of environments to be reset, by default None
        seed : int, optional
            Seed for RNG, by default None

        Returns
        -------
        typing.Tuple[NDArray, typing.Dict]
            Obs, and info
        """
        if not isinstance(seed, int):
            seed = int(seed[0]) if seed is not None else None
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        if reset_idxs is None:
            reset_idxs = np.array(self.env_ids, dtype=np.int32)
    
        reset_env_nums = len(reset_idxs)
        pos = self.rng.uniform(-1.0, 1.0, size = (reset_env_nums, 4))
        random_pos = pos[:, :3]
        random_pos =  self.position_spawn_bound_mean + random_pos*self.position_spawn_bound_diff
        yaw = pos[:, 3]
        yaw = self.yaw_mean + self.yaw_diff*yaw;
        orientation= np.zeros((reset_env_nums, 3))
        orientation[:,2] = yaw 
        self.server.reset_pos(reset_idxs, [self.uav_name]*reset_env_nums, random_pos, orientation)
        self.server.run_once(reset_idxs)
        self.server.update_control_states(reset_idxs)
        
        pos = self.rng.uniform(-1.0, 1.0, size = (reset_env_nums, 4))
        self.desired_pos[reset_idxs, :3] =pos[:, :3]
        self.desired_pos[reset_idxs, :3]  = self.position_spawn_bound_mean + self.desired_pos[reset_idxs, :3] *self.position_spawn_bound_diff
        self.desired_yaw[reset_idxs]  = self.yaw_mean + self.yaw_diff*pos[:, 3]
        self.action_buffer1[reset_idxs, :]  = np.zeros((reset_env_nums, 4))
        self.action_buffer2[reset_idxs, :]  = np.zeros((reset_env_nums, 4))
        
        # reset cyrr step and dones
        self.done[reset_idxs] = False
        self.current_step[reset_idxs] = 0
        # self.control_states  = self.server.control_states[self.uav_name][self.uav_link_name][0]
        # self.get_reward()
        obs, rew, done = self.get_obs_reward_done()
        if reset_idxs is not None:
            return obs[reset_idxs], {}
        return obs, {}

    def close(self) -> None:
        """Stop worker threads, release servers, and free the partition ID."""
        if self.server is not None:
            self.server.close()
            self.server = None
        if self.envid in HoverEnv._envids:
            HoverEnv._envids.remove(self.envid)
    
