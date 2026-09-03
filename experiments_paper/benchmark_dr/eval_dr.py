# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Amal Dev Haridevan

import os 
import argparse 
import random
import numpy as np
import torch
import gzdrl.envs as envs as envpool
from gzdrl.envs.env_adapters import VecAdapter
import stable_baselines3
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize
import tqdm 


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


algo_singlerl = ["ppo", "sac", "td3", "ddpg", "a2c"]
algo_singlerl = algo_singlerl + [algo.upper() for algo in algo_singlerl]

def parse():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-algo", nargs=1, required=True, type=str, help="Algorithms to run", choices = algo_singlerl)
    parser.add_argument("-env_name", type=str, required=True, help="Name of the environment", choices=envpool.list_all_envs())
    parser.add_argument("-model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("-logdir", type=str, required=True, help="logging directory")
    parser.add_argument("-vecnormalize_path", default="", type=str, help="Path to the VecNormalize statistics file (if applicable)")
    parser.add_argument("-episodes", type=int, default=1000,
                        help="Number of randomized evaluation episodes")
    parser.add_argument("-realtime", action="store_true",
                        help="Pace inference for video capture")
    return parser.parse_args()

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    args = parse()
    Adapter = VecAdapter
    args_ = ()
    log_dir = args.logdir
    os.makedirs(log_dir, exist_ok=True)
    
    # We create the eval_env with domain randomization explicitly active to test robustness
    eval_env = envpool.make(task_id=args.env_name, env_type="gymnasium", 
                            num_envs=1, domain_randomization=True)
    
    eval_env.spec.id = args.env_name
    
    # Setup VecNormalize if path is provided, otherwise skip normalization
    if args.vecnormalize_path and os.path.isfile(args.vecnormalize_path):
        print(f"Loading VecNormalize stats from {args.vecnormalize_path}")
        vec_env = VecNormalize.load(args.vecnormalize_path, Adapter(eval_env, *args_))
        vec_env.training = False # Ensure we don't update statistics during evaluation
        vec_env.norm_reward = False 
        _eval_env = VecMonitor(vec_env)
    else:
        _eval_env = VecMonitor(VecNormalize(Adapter(eval_env, *args_), norm_obs=False, norm_reward=False))
        
    # Get the correct SB3 algorithm class
    alg_name = args.algo[0].upper()
    algo_class = getattr(stable_baselines3, alg_name)
    
    # Load model
    print(f"Loading trained {alg_name} model from {args.model_path}")
    model = algo_class.load(args.model_path, env=_eval_env)
    
    # Evaluation Loop
    total_episodes = args.episodes
    episode_rewards = []
    
    print(f"Starting {total_episodes} evaluation episodes...")
    obs = _eval_env.reset()
    total_elapsed_step = 0
    avg_elapsed_step = 0
    def infer(obs, realtime=False):
        import time
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = _eval_env.step(action)
            if realtime:
                time.sleep(0.6e-2)
            # SB3 VecMonitor puts the true episodic return in infos under the "episode" key
            if dones[0]:
                
                if "episode" in infos[0]:
                    ep_reward = infos[0]["episode"]["r"]
                    episode_rewards.append(ep_reward)
                    # total_elapsed_step += infos[0]['elapsed_step']
                    # avg_elapsed_step = total_elapsed_step/len(episode_rewards)
                    if len(episode_rewards) % 10 == 0:
                        print(f"Finished {len(episode_rewards)}/{total_episodes} episodes. Recent reward: {ep_reward:.2f}")
                    obs = _eval_env.reset()
                    break
        return obs
    for i in tqdm.tqdm(range(total_episodes)):
        obs = infer(obs, realtime=args.realtime)

    # Save results
    episode_rewards = np.array(episode_rewards)
    np_save_path = os.path.join(log_dir, "eval_dr_rewards.npy")
    np.save(np_save_path, episode_rewards)
    print(f"\nStats - Mean: {episode_rewards.mean():.2f} +/- {episode_rewards.std():.2f}")
    print(f"Saved evaluation rewards array to {np_save_path}")
    
    # Plot histogram for the paper
    plt.figure(figsize=(8, 6))
    plt.hist(episode_rewards, bins=20, color='skyblue', edgecolor='black', alpha=0.8)
    plt.title(f"Evaluation Rewards - {args.env_name}\n({alg_name})")
    plt.xlabel("Episodic Return")
    plt.ylabel("Frequency")
    plt.grid(axis='y', alpha=0.5)

    plot_path = os.path.join(log_dir, "eval_dr_histogram.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved evaluation histogram plot to {plot_path}")
