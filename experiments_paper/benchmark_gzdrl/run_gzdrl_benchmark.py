# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Amal Dev Haridevan

import os 
import gzdrl.envs as envs
from experiments_paper.gpu_cpu_util import HardwareMonitor
import time
from multiprocessing import Process, Pipe, get_context
import numpy as np 
from concurrent.futures import ThreadPoolExecutor
import pickle
import argparse


def benchmark_n_envs(n_envs, batch_size, num_threads, n_steps, warmup):
    """Benchmark using PARALLEL stepping ."""
    print(f"\n{'='*60}")
    print(f"PARALLEL STEPPING")
    print(f"{'='*60}")
    print(f"Creating {n_envs} environments...")
    
    thread_affinity = 0
    env = envs.make_gym( "GazeboPoolHoverEnv-v0", num_envs=n_envs,
                            batch_size=batch_size,
                            thread_affinity_offset=thread_affinity,
                            num_threads=num_threads)
    env.async_reset()
    action = 0.0*np.array([env.action_space.sample() for _ in range(batch_size)])
    # warmup
    for _ in range(warmup):
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
    fps = n_steps * batch_size / duration 
    monitor.stop()
    monitor.join()
    avg_cpu, avg_gpu, avg_ram= monitor.get_averages()
    print(f"Duration: {duration:.2f}s")
    print(f"Total FPS: {fps:.2f} (env-steps/sec)")
    
    del env
    return fps, avg_cpu, avg_gpu, avg_ram

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark VecEnv implementations")
    parser.add_argument("--n-envs", type=int,  required=True,default=1, help="Number of parallel environments to create")
    parser.add_argument("--n-steps", type=int, required=True, default=10000, help="Number of steps to run in the benchmark")
    parser.add_argument("--warmup", type=int, required=True, default=1000, help="Warmup steps, before actual benchmark")
    parser.add_argument("--out-file", required=True, type=str, default="results.csv", help="file to save benchmark results (csv format)")
    parser.add_argument("--num-threads", type=int, required=True, help="Num threads for gzdrl")
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
    num_threads = min(os.cpu_count(), min(n_envs, args.num_threads))
    batch_size = max(1, args.n_envs//3)
    fps,cpu_util, gpu_util, ram_util = benchmark_n_envs(n_envs, batch_size, num_threads, n_steps, warmup,)
    
    save_results([n_envs, n_steps, fps, cpu_util, gpu_util, ram_util], args.out_file)
    print("completed")
    os._exit(0)
