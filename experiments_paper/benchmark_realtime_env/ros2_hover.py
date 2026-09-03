# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Amal Dev Haridevan

"""This environment is the exact impl of the envpool HoverEnv-v0 and python HoverEnv, but  now
using ros2-Gz. This is for benchmarking the performance, as well as demonstrating """

import gymnasium as gym
import os 
import gzdrl

import numpy as np
import random
from typing import List, Tuple
from scipy.spatial.transform import Rotation
import time 
from experiments_paper.benchmark_realtime_env.ros2_drl_env import BaseRos2DRLEnv
from multiprocessing import Process

class HoverEnv(gym.Env):
    """Global attrs, these are common for all envs"""
    _envids = []
    _sdf_file = str(gzdrl.get_sdf_path("world_hover.sdf"))
        
    def __init__(self,envid: int = 0, sep_proc: bool = True) -> None:
        """
        Initialize the HoverEnv
        
        This initializes the ROS 2 comparison environment through its
        middleware-based server API. The implementation is intentionally kept
        separate from GzDRL's in-process stepping path for the synchronization
        and throughput comparison.
        
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
        partition = f"env_{envid}"
        sdf_file = HoverEnv._sdf_file
        """Name as specified in the sdf file"""
        self.uav_name = "quadrotor"
        self.uav_link_name = "quadrotor/base_link"
        self.server = BaseRos2DRLEnv(partition ,[self.uav_name], [self.uav_link_name], construct_ros2_env=True ,
                                     ros2_env_sdf=sdf_file, sep_proc=sep_proc)
        self.batch = 1
        # action buffer
        self.action_buffer1 = np.zeros((self.batch, 4)) 
        self.action_buffer2 = np.zeros((self.batch, 4))
        self.max_steps = 1000
        self.current_step = np.zeros((self.batch))
        self.done = np.zeros((1), dtype=bool)
        self.maximum_bounds  = np.array((7.0, 7.0, 7.0))
        self.position_spawn_bound_mean = (np.array((-5.0, -5.0, 0.0))+ np.array((5.0, 5.0, 3.0)))/2
        self.position_spawn_bound_diff = (-np.array((-5.0, -5.0, 0.0))+ np.array((5.0, 5.0, 3.0)))/2
        self.yaw_mean = 0
        self.yaw_diff = 0.72
        self.desired_pos = np.zeros(( 3,))
        self.desired_yaw =  0
        self.max_yaw = 1.57
        """define action and observation spaces"""
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4, ), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(22, ), dtype=np.float32)
        # cache
        self.control_states = None 
        self.cache_vars = {}
        # obs cache to prveent re-alloc
        self.obs_cache = np.zeros((22), dtype=np.float32)
        # RNG
        self.rng = np.random.default_rng(seed=self.envid)
        # cache envids and model names 
        self.env_ids = list(range(self.batch))
        self.uav_b =["quadrotor"]*self.batch
        self.link_b = ["quadrotor/base_link"]*self.batch
        self.uav_states = None
        self.reset()


    def step(self, action):
        self.done = self.current_step >= self.max_steps
        self.current_step += 1
        action[np.isnan(action)] = 0.0
        action = (np.tanh(action) + 1.0)*2300.0 /2.0
        self.server.set_rotor_velocity_cmd(self.uav_name, self.uav_link_name, action.ravel().tolist())
        
        self.action_buffer2 = self.action_buffer1
        self.action_buffer1 = action 
        obs, reward, done = self.get_obs_reward_done()
        # self.make_obs()
        return self.obs_cache.copy(), reward, done, False, {}

    def get_obs_reward_done(self):
        self.uav_states = self.server.control_states(self.uav_name)[self.uav_link_name]

        current_pos = self.uav_states[ :3]
        relative_pos = self.desired_pos - current_pos
        omega = self.uav_states[10:13]
        quat_ = self.uav_states[6:10]
        rot = Rotation.from_quat(quat_, scalar_first=True)
        current_yaw = rot.as_euler('ZYX')[0]
        yaw_error = self.desired_yaw - current_yaw

        state_err =  1.6*1.6*(np.linalg.norm(relative_pos)**2 + yaw_error*yaw_error);
        reward_pos = 1.0 / (1.0 + state_err);
        reward_omega = 0.01 / (1.0 + np.linalg.norm(omega)**2 )
        # reward_effort = np.clip(-1e-6*(np.linalg.norm(self.action_buffer[-1])), -1.0, 1.0)
        # act1 = self.action_buffer[-1]
        # act2 = self.action_buffer[-2] if len(self.action_buffer) > 1 else act1 
        # reward_smooth = np.clip(-1e-6*(np.linalg.norm(act2-act1)), -1.0, 1.0);
        yaw_oob_cost = 0.0;
        done = self.done
        done = np.abs(current_yaw) > self.max_yaw 

        bounds_err = self.maximum_bounds - current_pos
        done= np.any(np.abs(current_pos)-self.maximum_bounds > 0)  >0.0 
        rotmat =rot.as_matrix()
        cosang = np.abs(np.arccos(np.clip(rotmat[2,2], -1.0, 1.0)))
        
        done = cosang>1.57
        for i in range(3):
            self.obs_cache[ i] = relative_pos[ i]
        self.obs_cache[3] = yaw_error
        for i in range(4, 14):
            self.obs_cache[i] = self.uav_states[i-1]
        
        self.obs_cache[14 : 18 ] = self.action_buffer1/1300.0 -1.0
        self.obs_cache[18 : 22 ] = self.action_buffer2/1300.0 -1.0
        total_rew = reward_pos + reward_omega 
        return self.obs_cache.copy(),  total_rew, done
    
    def reset(self,  seed=None, **kwargs):
        if not isinstance(seed, int):
            seed = int(seed[0]) if seed is not None else None
        super().reset(seed=seed)
        
        
    
        reset_env_nums =1
        pos = self.rng.uniform(-1.0, 1.0, size = (reset_env_nums, 4))
        random_pos = pos[:, :3]
        random_pos =  self.position_spawn_bound_mean + random_pos*self.position_spawn_bound_diff
        yaw = pos[:, 3]
        yaw = self.yaw_mean + self.yaw_diff*yaw;
        orientation= np.zeros((reset_env_nums, 3))
        orientation[:,2] = yaw 
        self.server.reset_pose( self.uav_name, random_pos.ravel(), orientation.ravel())
        time.sleep(0.1)
        
        pos = self.rng.uniform(-1.0, 1.0, size = (reset_env_nums, 4))
        self.desired_pos[:3] =pos[:, :3]
        self.desired_pos[:3]  = self.position_spawn_bound_mean + self.desired_pos[:3] *self.position_spawn_bound_diff
        self.desired_yaw  = float( self.yaw_mean + self.yaw_diff*pos[:, 3])
        self.action_buffer1[:]  = np.zeros((reset_env_nums, 4))
        self.action_buffer2[:]  = np.zeros((reset_env_nums, 4))
        
        # reset cyrr step and dones
        self.done = False
        self.current_step = 0
        # self.control_states  = self.server.control_states[self.uav_name][self.uav_link_name][0]
        # self.get_reward()
        obs, rew, done = self.get_obs_reward_done()
        return obs, {}
    
if __name__ == "__main__":
    import gzdrl as grl 
    from scipy.spatial.transform import Rotation
    gain_map = grl.GAIN_MAP()
    param_map = grl.PARAMETER_MAP()
    geom_gains = gain_map["qdrone2"]["geometric_controller"]
    geom_params = param_map["qdrone2"]["geometric_controller"]

    controller = grl.GeometricController(geom_gains["kp"], geom_gains["kd"],
            geom_gains["kp_att"], geom_gains["kd_att"],
            geom_params.max_accel,
            geom_params.gravity_vec,
            geom_params.mass,
            geom_params.inertia)
    mapper = lambda u : np.arctanh(2*np.clip(u, 0, 2299)/2300 - 1)
    env = HoverEnv(0)
    env.reset()
    des_state = np.zeros((13))
    des_state[:3] = [1,2,3]
    des_state[6] = 1.0
    ctbt_2_rt = np.array([[  0.25      ,  -1.96850394,   2.46062992, -17.19394773],
       [  0.25      ,  -1.96850394,  -2.46062992,  17.19394773],
       [  0.25      ,   1.96850394,   2.46062992,  17.19394773],
       [  0.25      ,   1.96850394,  -2.46062992, -17.19394773]])
    rt_2_rvel = lambda rt : np.sqrt(np.clip(rt,0, 20)/1.9533e-06)
    state_dot = np.zeros((13))
    for i in range(60000):
        state = env.uav_states
        ctbt = controller.calculate_thrust_moments(state, state_dot, des_state)
        # rvels = rt_2_rvel(ctbt_2_rt@ctbt)
        # rvels[rvels==np.nan] = 0
        # print(mapper(rvels), ": ", rvels)
        # env.step(mapper(rvels))
        R = Rotation.from_quat(state[6:10], scalar_first=True).as_matrix()
        F_i = R@[0, 0, ctbt[0]]
        M_i = R@ctbt[1:]
        env.server.set_wrench("quadrotor", "quadrotor/base_link",(*F_i, *M_i))
        time.sleep(1e-3)
