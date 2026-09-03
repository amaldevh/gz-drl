# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Amal Dev Haridevan

import os 
from experiments_paper.gpu_cpu_util import HardwareMonitor
import gzdrl.envs as envs as envpool
import argparse
import time 
import numpy as np
import yaml 
import pandas as pd
from multiprocessing import Process, Pipe
import multiprocessing as mp
mp.set_start_method('spawn', force=True)


def run_process(kwargs, conn):
    """Runs envpool in separate process,
    this ensures that the envpool envs are destroyed properly after use
    Otherwise C++ side may keep accumulating envs and lead to memory leak
    """
    import gzdrl.envs as envs as envpool
    
    n_envs = kwargs["num_envs"]
    env_name = kwargs["env_name"]
    batch_size = kwargs["batch_size"]
    thread_affinity = kwargs["thread_affinity_offset"] 
    num_threads = kwargs["num_threads"] 
    n_steps = kwargs.get("n_steps", 500)
    
    env = envpool.make_gym( env_name, num_envs=n_envs,
                            batch_size=batch_size,
                            thread_affinity_offset=thread_affinity,
                            num_threads=num_threads)
    env.async_reset()
    action = 0.0*np.array([env.action_space.sample() for _ in range(batch_size)])
    t = time.perf_counter()
    for _ in range(n_steps):
        info = env.recv()[-1]
        env.send(action, info["env_id"])
    duration = time.perf_counter() - t
    fps = n_steps * batch_size / duration 
    print(f"Duration = {duration:.2f}s")
    print(f"GazeboEnvPool FPS = {fps:.2f}")
    del env
    conn.send((duration, fps))
    conn.close()
            
def parse():
    parser = argparse.ArgumentParser(description="Benchmark GazeboEnvPool with different configurations")
    parser.add_argument("--num-agents",  type=int, required=True, help="Number of agents in the formation")
    parser.add_argument("--warmup", type=int, required=True, help="Warmup steps")
    parser.add_argument("--n-steps", type=int, required=True, help="Total steps to run")
    parser.add_argument("--n-envs", type=int, required=True, help="Total envs")
    parser.add_argument("--out-file", type=str, required=True, default="results.csv", help="file to save benchmark results (csv format)")
    return parser.parse_args()

if __name__ == "__main__":
    """Normal benchmark run"""
    args = parse()
    n_envs = args.n_envs
    env_name = "GazeboPoolMultiAgentFormationLLEnv-v0"
    batch_size = 11
    thread_affinity = 0 
    num_threads = os.cpu_count()
    warmup=args.warmup
    n_steps = args.n_steps
    env = envpool.make_gym( env_name, num_envs=n_envs,
                            batch_size=batch_size,
                            thread_affinity_offset=thread_affinity,
                            num_threads=num_threads, num_agents=args.num_agents)
    env.async_reset()
    action = 0.0*np.array([env.action_space.sample() for _ in range(batch_size)])
    # warmup
    for _ in range(warmup):
        for _ in range(n_envs):
            info = env.recv()[-1]
            env.send(action, info["env_id"])
    # actual benchmark
    obs = []
    for _ in range(n_envs):
        info = env.recv()[-1]
        env.send(action, info["env_id"])
    monitor = HardwareMonitor(interval=0.1)
    monitor.start()
    t = time.perf_counter()
    for _ in range(n_steps):
        info = env.recv()[-1]
        env.send(action, info["env_id"])
    duration = time.perf_counter() - t
    monitor.stop()
    monitor.join()
    avg_cpu, avg_gpu, avg_ram = monitor.get_averages()
    fps = n_steps * batch_size / duration 
    print(f"Duration = {duration:.2f}s")
    print(f"GazeboEnvPool FPS = {fps:.2f}")
    if not os.path.exists(args.out_file):
        with open(args.out_file, "w") as f:
            f.write("n_envs,n_agents,n_steps,fps,avg_cpu,avg_gpu,avg_ram\n")
    with open(args.out_file, "a") as f:
        f.write(f"{n_envs},{args.num_agents},{n_steps},{fps:.2f},{avg_cpu},{avg_gpu},{avg_ram}\n")
