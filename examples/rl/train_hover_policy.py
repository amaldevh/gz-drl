# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Amal Dev Haridevan

from stable_baselines3.ppo import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize
from gzdrl.envs.python_envs.hover_env import HoverEnv

if __name__ == "__main__":
    # Create vectorized environment
    def make_env():
        env = HoverEnv()
        env = Monitor(env)
        return env

    vec_env = make_vec_env(make_env, n_envs=15)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # Define the model
    model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log="/tmp/hover_tensorboard/")

    # Define a checkpoint callback to save the model every 10000 steps
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='/tmp/hover_models/',
                                             name_prefix='hover_policy')

    # Train the model
    model.learn(total_timesteps=500000, callback=checkpoint_callback)

    # Save the final model
    model.save("/tmp/hover_models/hover_policy_final")
