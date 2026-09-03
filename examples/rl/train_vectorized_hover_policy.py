# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Amal Dev Haridevan

from stable_baselines3.ppo import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import  VecMonitor
from stable_baselines3.common.vec_env import VecNormalize
from gzdrl.envs.python_envs.vectorized_hover_env import HoverEnv
from gzdrl.envs.env_adapters import AsyncDRLServerVecAdapter

if __name__ == "__main__":
    # Create vectorized environment
    base_vec_env = HoverEnv(N_envs=15)
    vec_env = VecMonitor(AsyncDRLServerVecAdapter(base_vec_env))

    # Define the model
    model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log="/tmp/hover_vec_tensorboard/")

    # Define a checkpoint callback to save the model every 10000 steps
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='/tmp/hover_vec_models/',
                                             name_prefix='hover_vec_policy')

    # Train the model
    model.learn(total_timesteps=500000, callback=checkpoint_callback)

    # Save the final model
    model.save("/tmp/hover_vec_models/hover_vec_policy_final")
