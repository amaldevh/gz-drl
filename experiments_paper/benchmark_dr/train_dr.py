# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Amal Dev Haridevan

import os 
import argparse 
import random
import numpy as np
import torch
import gzdrl.envs as envs as envpool
from gzdrl.envs.env_adapters import VecAdapter, VecAdapterImageEnv
import stable_baselines3
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.vec_env import VecEnvWrapper, VecMonitor, VecNormalize
from gzdrl.envs.sb3_custom_callbacks import SaveOnBestTrainingRewardCallback
from experiments_paper.benchmark_dr.asymmetric_ppo import (
    AsymmetricActorCriticPolicy,
    NoOpFeaturesExtractor,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



class SyncNormStats(BaseCallback):
    def __init__(self, train_venv, eval_venv, enabled):
        super().__init__()
        if enabled:
            self.train_venv = train_venv.venv
            self.eval_venv = eval_venv.venv
        self.enabled = enabled

    def _on_step(self):
        if self.enabled:
            self.eval_venv.obs_rms = self.train_venv.obs_rms
        return True


algo_singlerl = ["ppo", "sac", "td3", "ddpg", "a2c"]
algo_singlerl = algo_singlerl + [algo.upper() for algo in algo_singlerl]

def parse():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-algos", nargs="+", required=True, type=str, help="Algorithms to run", choices = algo_singlerl)
    parser.add_argument("-max_steps", type=float, required=True, help="Maximum number of steps to train")
    parser.add_argument("-seeds", type=int, nargs="+", required=True, help="Seed for torch and numpy")
    parser.add_argument("-logdir", type=str, required=True, help="logging directory")
    parser.add_argument("-env_name", type=str, required=True, help="Name of the environment", choices=envpool.list_all_envs())
    parser.add_argument("-n_envs", type=int, required=True, help="Number of parallel envs")
    parser.add_argument("-domain_randomization", action="store_true", help="Whether to use domain randomization during training")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse()
    Adapter = VecAdapter
    args_ = ()
    log_dir = args.logdir
    norm_obs = False
    privileged_obs = True
    eval_env = envpool.make(task_id=args.env_name, env_type="gymnasium", num_envs=1, test_env=True ,
                            privileged_obs =privileged_obs)
    obs, info = eval_env.reset()
    policy_obs_dim = int(info['policy_obs_dim'])
    privileged_info_dim = int(info['privileged_obs_dim'])
    eval_env.spec.id = args.env_name
    _eval_env = VecMonitor(VecNormalize(Adapter(eval_env,*args_), norm_obs=norm_obs, norm_reward=False))
    train_env = envpool.make( task_id=args.env_name, env_type="gymnasium", num_envs=args.n_envs, test_env=False ,
                             domain_randomization=args.domain_randomization,
                             privileged_obs =privileged_obs )
    train_env.spec.id = args.env_name
    vec_train_env = VecNormalize(Adapter(train_env, *args_), norm_obs=norm_obs, norm_reward=False)
    max_steps = args.max_steps

    for seed in args.seeds:
        for algo in args.algos:
            _eval_env.reset()
            set_seed(seed)
            algo_path = log_dir+f"/exp_{seed}_{algo}"
            os.makedirs(algo_path, exist_ok=True)
            monitor_path = os.path.join(algo_path, "monitor.csv")
            _train_env = VecMonitor(vec_train_env, filename=monitor_path)
            _train_env.reset()
            eval_path = algo_path + "/best_model"
            train_save_path = algo_path + "/train_best_model"
            if algo in algo_singlerl:
                eval_callback = EvalCallback(_eval_env, best_model_save_path=eval_path, log_path=eval_path,
                                            eval_freq=max(10000, 1), n_eval_episodes=1, deterministic=True,
                                            render=False) 
                
                policy_kwargs = dict(
                    features_extractor_class=NoOpFeaturesExtractor,
                    policy_obs_dim = policy_obs_dim,
                    privileged_info_dim=privileged_info_dim
                )
                if privileged_obs:
                    model = stable_baselines3.PPO(AsymmetricActorCriticPolicy, _train_env, verbose=1,
                        policy_kwargs=policy_kwargs, tensorboard_log=algo_path)
                else:
                    model = stable_baselines3.PPO("MlpPolicy", _train_env, verbose=1,
                                                policy_kwargs=dict(net_arch=[512, 512], ), tensorboard_log=algo_path)
                model.learn(total_timesteps=max_steps, tb_log_name=algo, log_interval=1  , 
                            callback=[eval_callback , SyncNormStats(_train_env, _eval_env, norm_obs),
                                      SaveOnBestTrainingRewardCallback(check_freq=1000,
                                                                   save_path=train_save_path,                
                                                                    monitor_dir=algo_path)], progress_bar=True )
                # save final model
                model.save(algo_path + "/final_model")
                # model.learn(total_timesteps=max_steps, tb_log_name=algo, log_interval=1  , progress_bar=True )
