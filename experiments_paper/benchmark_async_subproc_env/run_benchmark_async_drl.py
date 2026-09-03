# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Amal Dev Haridevan

import os 
from experiments_paper.gpu_cpu_util import HardwareMonitor
from gzdrl.envs.python_envs.vectorized_hover_env import HoverEnv
import time
from multiprocessing import Process, Pipe, get_context
import numpy as np 
import pickle
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark ROS2 VecEnv implementations")
    parser.add_argument("--n-envs", type=int, required=True,default=1, help="Number of parallel environments to create")
    parser.add_argument("--n-steps", type=int, required=True,default=10000, help="Number of steps to run in the benchmark")
    parser.add_argument("--warmup", type=int, required=True, default=1000, help="Warmup steps, before actual benchmark")
    parser.add_argument("--out-file", type=str, required=True, default="results.csv", help=" file to save benchmark results (csv format)")
    return parser.parse_args()

def save_results(results, filename):
    if not os.path.exists(filename):
        with open(filename, "w") as f:
            f.write("n_envs,n_steps,fps,avg_cpu,avg_gpu,avg_ram\n")
    with open(filename, "a") as f:
        f.write(",".join(map(str, results)) + "\n")

if __name__ == "__main__":
    args = parse_args()
    n_envs = args.n_envs
    n_steps = args.n_steps
    warmup = args.warmup
    env = HoverEnv(n_envs, 0)
    action = 0.0 * np.array([env.action_space.sample() for _ in range(n_envs)])
    # warmup
    for _ in range(warmup):
        __ = env.step(action)
    # actual benchmark
    monitor = HardwareMonitor(interval=0.1)
    monitor.start()
    ts = time.perf_counter()
    for _ in range(n_steps):
        __ = env.step(action)
    duration = time.perf_counter() - ts
    monitor.stop()
    monitor.join()
    avg_cpu, avg_gpu, avg_ram = monitor.get_averages()
    fps = n_steps * n_envs / duration
    print(f"fps: {fps:.2f} (env-steps/sec)")
    print("avg_cpu, gpu: ", avg_cpu,", ", avg_gpu)
    save_results([n_envs, n_steps, fps, avg_cpu, avg_gpu, avg_ram], args.out_file)
    print("completed")
    os._exit(0)
