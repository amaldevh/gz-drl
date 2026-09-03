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
    """HoverEnv using DRLServer python API
    """
    _envids = []
    _sdf_file =  "world_hover.sdf"
    
    def __init__(self, envid: int = 0) -> None:
        """
        Initialize the HoverEnv
        
        This initializes the HoverEnv using the DRLServer API.
        
        Parameters
        ----------
        envid : int
            A unique envid to properly partition GZ_SERVER. This is not required to be unique, if no visualization
            is desired.
            
        Examples
        --------
        >>> env = HoverEnv(0)
        """
        super().__init__()
        envid = int(envid)
        while envid in HoverEnv._envids:
            envid += 1
            
        self.envid = envid
        HoverEnv._envids.append(envid)
        
        partition = f"{envid}"
        sdf_file = str(gzdrl.get_sdf_path(HoverEnv._sdf_file))
        self.uav_name = "quadrotor"
        self.uav_link_name = "quadrotor/base_link"
        self.server = gzdrl.DRLServer(partition, sdf_file, [self.uav_name], False)
        # action buffer
        self.action_buffer1 = np.zeros((1, 4)) 
        self.action_buffer2 = np.zeros((1, 4))
        self.max_steps = 20000
        self.current_step = 1
        self.done = True
        self.maximum_bounds  = np.array((7.0, 7.0, 7.0))
        self.position_spawn_bound_mean = (np.array((-5.0, -5.0, 0.0))+ np.array((5.0, 5.0, 3.0)))/2
        self.position_spawn_bound_diff = (-np.array((-5.0, -5.0, 0.0))+ np.array((5.0, 5.0, 3.0)))/2
        self.yaw_mean = 0
        self.yaw_diff = 0.72
        self.desired_pos = np.zeros(( 3,))
        self.desired_yaw =  np.zeros((1))
        self.max_yaw = 1.57
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4, ), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(22, ), dtype=np.float32)
        # obs cache to prveent re-alloc
        self.obs_cache = np.zeros(( 22), dtype=np.float32)
        # RNG
        self.rng = np.random.default_rng(seed=self.envid)

        self.model_name = "quadrotor"
        self.link_name = "quadrotor/base_link"
        self.uav_states = None
        self.reset()


    def step(self, action: NDArray)->typing.Tuple[NDArray, float, bool, bool, typing.Dict]:
        """Steps the environment using DRLServer API, obtains the new state, and returns the
            new obs, reward, done, and info

        Parameters
        ----------
        action : NDArray
            Action for the agents

        Returns
        -------
        typing.Tuple[NDArray, float, bool, bool, typing.Dict]
            Obs, Reward, Done, Truncated, Info
        """
        self.current_step += 1
        action[np.isnan(action)] = 0.0
        action = ((action) + 1.0)*2300.0 /2.0
        self.server.set_rotor_velocity_cmd( self.model_name, self.link_name, action)
        self.server.run_once()
        self.server.update_control_states()
        self.action_buffer2 = self.action_buffer1
        self.action_buffer1 = action 
        obs, reward, terminated = self.get_obs_reward_done()
        truncated = self.current_step >= self.max_steps
        # self.make_obs()
        return self.obs_cache.copy(), reward, terminated, truncated, {}

    def get_obs_reward_done(self)->typing.Tuple[NDArray, float, bool]:
        """Helper method for obtaining obs, reward, and done

        Returns
        -------
        typing.Tuple[NDArray, float, bool]
            Obs, rew, done
        """
        self.control_states  =  self.server.control_states
        self.uav_states = self.control_states[self.model_name][self.link_name][0] 

        current_pos = self.uav_states[:3]
        relative_pos = self.desired_pos - current_pos
        omega = self.uav_states[10:13]
        quat_ = self.uav_states[6:10]
        rot = Rotation.from_quat(quat_, scalar_first=True)
        current_yaw = rot.as_euler('ZYX')[ 0]
        yaw_error = self.desired_yaw - current_yaw

        state_err =  1.6*1.6*(np.linalg.norm(relative_pos)**2 + yaw_error*yaw_error);
        reward_pos = 1.0 / (1.0 + state_err);
        reward_omega = 0.01 / (1.0 + np.linalg.norm(omega)**2 )
        yaw_oob_cost = 0.0;
        done = bool(np.abs(current_yaw) > self.max_yaw)

        bounds_err = np.any(np.abs(current_pos)-self.maximum_bounds > 0)
        done = done or bounds_err
        rotmat =rot.as_matrix()
        cosang = np.abs(np.arccos(np.clip(rotmat[2,2], -1.0, 1.0)))
        
        done = (cosang > 1.57) or done
        self.done = done
        for i in range(3):
            self.obs_cache[ i] = relative_pos[i]
        self.obs_cache[ 3] = yaw_error
        for i in range(4, 14):
            self.obs_cache[ i] = self.uav_states[i-1]
        
        self.obs_cache[14 : 18 ] = self.action_buffer1/1300.0 -1.0
        self.obs_cache[18 : 22 ] = self.action_buffer2/1300.0 -1.0
        total_rew = reward_pos + reward_omega 
        return self.obs_cache.copy(),  total_rew, done
    
    def reset(self, seed: int=None, **kwargs)->typing.Tuple[NDArray, typing.Dict]:
        """Resets the environment

        Parameters
        ----------
        seed : int, optional
            seed for RNG, by default None

        Returns
        -------
        typing.Tuple[NDArray, typing.Dict]
            Observation and info
        """
        if not isinstance(seed, int):
            seed = int(seed[0]) if seed is not None else None
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        pos = self.rng.uniform(-1.0, 1.0, size = ( 4))
        random_pos = pos[:3]
        random_pos =  self.position_spawn_bound_mean + random_pos*self.position_spawn_bound_diff
        yaw = pos[ 3]
        yaw = self.yaw_mean + self.yaw_diff*yaw;
        orientation= np.zeros(( 3))
        orientation[2] = yaw 
        self.server.reset_pos(self.uav_name, random_pos, orientation)
        self.server.run_once()
        self.server.update_control_states()
        
        pos = self.rng.uniform(-1.0, 1.0, size = (4))
        self.desired_pos[:3] =pos[ :3]
        self.desired_pos[:3]  = self.position_spawn_bound_mean + self.desired_pos[:3] *self.position_spawn_bound_diff
        self.desired_yaw  = self.yaw_mean + self.yaw_diff*pos[ 3]
        self.action_buffer1[ :]  = np.zeros((1, 4))
        self.action_buffer2[ :]  = np.zeros((1, 4))
        
        # reset cyrr step and dones
        self.done = False
        self.current_step = 0
        # self.control_states  = self.server.control_states[self.uav_name][self.uav_link_name][0]
        # self.get_reward()
        obs, rew, done = self.get_obs_reward_done()
        return obs, {}

    def close(self) -> None:
        """Release the native server and make the partition ID reusable."""
        self.server = None
        if self.envid in HoverEnv._envids:
            HoverEnv._envids.remove(self.envid)
    
