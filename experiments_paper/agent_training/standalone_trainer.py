# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Amal Dev Haridevan

import os 
import argparse 
import random
import numpy as np
import torch
import gzdrl.envs as envs 
from gzdrl.envs.env_adapters import VecAdapter, VecAdapterImageEnv
import stable_baselines3
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecEnvWrapper, VecMonitor, VecNormalize
from gzdrl.envs.sb3_custom_callbacks import (
    SaveOnBestTrainingRewardCallback,
    SaveVecNormalizeCallback,
)
from stable_baselines3.common.callbacks import CheckpointCallback
import yaml 
from functools import partial


def set_seed(seed: int = 42, torch_threads: int = 1) -> None:
    """Set all Python-side stochastic sources before constructing policy/env."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(torch_threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


algo_singlerl = ["ppo", "sac", "td3", "ddpg", "a2c"]
algo_singlerl = algo_singlerl + [algo.upper() for algo in algo_singlerl]
algo_marl = ["mappo", "maddpg", "masac", "ippo"]
algo_marl = algo_marl + [algo.upper() for algo in algo_marl]




def algorithm_kwargs(algo: str, env_name: str):
    """Return task-specific optimizer settings."""
    if algo.lower() == "ppo" :
        return {
            "learning_rate": 1.0e-4,
            "n_steps": 2048,
            "batch_size": 512,
            "n_epochs": 3,
            "target_kl": 0.02,
            "ent_coef": 1.0e-3,
        }
    return {}

def parse():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-algos", nargs="+", required=True, type=str, help="Algorithms to run", choices = algo_singlerl+algo_marl)
    parser.add_argument("-max_steps", type=float, required=True, help="Maximum number of steps to train")
    parser.add_argument("-seed", type=int, required=True, help="Seed for torch and numpy")
    parser.add_argument("-logdir", type=str, required=True, help="logging directory")
    parser.add_argument("-env_name", type=str, required=True, help="Name of the environment", choices=envs.list_all_envs())
    parser.add_argument("-n_envs", type=int, required=True, help="Number of parallel envs")
    parser.add_argument("-gz_partition_offset", type=int, required=True, help="Partition offset when spawning multiple standalone trainers")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse()
    set_seed(args.seed)
    gz_partition_offset = args.gz_partition_offset
    Adapter = VecAdapter 
    args_ = () 
    log_dir = args.logdir
    normalize_observations = True
    eval_env = envs.make(task_id=args.env_name, env_type="gymnasium", num_envs=1, batch_size=1, num_threads=1, test_env=True,
         gz_partition_offset=gz_partition_offset, seed=args.seed )
    eval_env.spec.id = args.env_name
    eval_normalizer = VecNormalize(
        Adapter(eval_env, *args_),
        training=False,
        norm_obs=normalize_observations,
        norm_reward=False,
    )
    _eval_env = VecMonitor(eval_normalizer)
    train_env = envs.make( task_id=args.env_name, env_type="gymnasium", num_envs=args.n_envs, test_env=False,
        gz_partition_offset=gz_partition_offset , seed=args.seed)
    train_env.spec.id = args.env_name
    vec_train_env = VecNormalize(
        Adapter(train_env, *args_),
        training=True,
        norm_obs=normalize_observations,
        norm_reward=False,
    )
    max_steps = int(args.max_steps)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    seed = args.seed
    for algo in args.algos:
        _eval_env.reset()
        algo_path = log_dir+f"/exp_{seed}_{algo}"
        os.makedirs(algo_path, exist_ok=True)
        monitor_path = os.path.join(algo_path, "monitor.csv")
        _train_env = VecMonitor(vec_train_env, filename=monitor_path)
        _train_env.reset()
        eval_path = algo_path + "/best_model"
        train_save_path = algo_path + "/train_best_model"
        eval_normalization_callback = (
            SaveVecNormalizeCallback(eval_path)
            if normalize_observations
            else None
        )
        eval_callback = EvalCallback(_eval_env, best_model_save_path=eval_path, log_path=eval_path,
                                    eval_freq=max(10000, 1),
                                    n_eval_episodes=(5 if normalize_observations else 1),
                                    deterministic=True, render=False,
                                    callback_on_new_best=eval_normalization_callback)
        
        ALGO = getattr(stable_baselines3, algo.upper())
        kwargs={}
        policy_type = "MlpPolicy"
        algo_kwargs = algorithm_kwargs(algo, args.env_name)
        policy_kwargs = dict(net_arch=[512,512])
        model = ALGO(policy_type, _train_env, policy_kwargs=policy_kwargs, **algo_kwargs,  verbose=1, tensorboard_log=algo_path)
        model.learn(total_timesteps=max_steps, tb_log_name=algo, log_interval=1  , 
                    callback=[eval_callback , SaveOnBestTrainingRewardCallback(check_freq=1000,
                                                            save_path=train_save_path,                
                                                            monitor_dir=algo_path)], progress_bar=True )
        # save final model
        model.save(algo_path + "/final_model")
        if normalize_observations:
            vec_train_env.save(algo_path + "/final_vecnormalize.pkl")
            
