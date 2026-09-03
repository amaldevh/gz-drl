# SPDX-License-Identifier: MIT
# Copyright (c) 2019 Antonin Raffin
# Modifications copyright (c) 2025-2026 Amal Dev Haridevan
# Adapted from the Stable-Baselines3 callback examples.

import os
import gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, unwrap_vec_normalize
from stable_baselines3.common.results_plotter import load_results, ts2xy

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model based on the training reward (in practice, we recommend using EvalCallback).

    :param check_freq: (int) Frequency of checking the reward
    :param save_path: (str) Path to the folder where the model will be saved
    :param verbose: (int) Verbosity level: 0 for no output, 1 for info messages
    :param monitor_dir: (str) Path to the folder where the monitor files are saved
    """
    def __init__(self, check_freq: int, save_path: str, verbose=1, monitor_dir: str = None):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.monitor_dir = monitor_dir
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Retrieve training rewards from the Monitor wrapper
            log_dir = self.monitor_dir if self.monitor_dir is not None else self.locals['self'].logger.dir
            x, y = ts2xy(load_results(log_dir), 'timesteps')
            if len(x) > 0:
                # Mean reward over the last 100 episodes
                mean_reward = np.mean(y[-100:]) 
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}. Mean reward over 100 episodes: {mean_reward:.2f}. Best mean reward: {self.best_mean_reward:.2f}")

                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}/best_model.zip")
                    self.model.save(os.path.join(self.save_path, 'best_model.zip'))
                    
                    # Save VecNormalize statistics if present
                    if self.training_env is not None:
                        vec_normalize_env = unwrap_vec_normalize(self.training_env)
                        if vec_normalize_env is not None:
                            vec_normalize_env.save(os.path.join(self.save_path, 'best_vecnormalize.pkl'))
        return True


class SaveVecNormalizeCallback(BaseCallback):
    """Save the training VecNormalize state when an evaluation model is saved."""

    def __init__(self, save_path: str, filename: str = "best_vecnormalize.pkl"):
        super().__init__(verbose=0)
        self.save_path = save_path
        self.filename = filename

    def _init_callback(self) -> None:
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        vec_normalize_env = unwrap_vec_normalize(self.training_env)
        if vec_normalize_env is None:
            raise RuntimeError(
                "SaveVecNormalizeCallback requires a VecNormalize training "
                "environment."
            )
        vec_normalize_env.save(os.path.join(self.save_path, self.filename))
        return True
