# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Amal Dev Haridevan

import os 
from experiments_paper.gpu_cpu_util import HardwareMonitor
from gzdrl.envs.python_envs.hover_env import HoverEnv
import time
from multiprocessing import Process, Pipe, get_context
import numpy as np 
from concurrent.futures import ThreadPoolExecutor
import pickle
import argparse

def _env_worker(conn, env_id):
    env = HoverEnv(env_id)
    obs, _ = env.reset()
    conn.send(("reset", obs))
    while True:
        cmd, data = conn.recv()
        if cmd == "step":
            obs, reward, done, trunc, info = env.step(data)
            conn.send(("step", (obs, reward, done, trunc, info)))
        elif cmd == "reset":
            obs, _ = env.reset()
            conn.send(("reset", obs))
        elif cmd == "close":
            break
    conn.close()

class VecEnv:
    """
    Parallel stepping with ThreadPoolExecutor
    =========================================================
    This is the corrected version - uses proper parallelization
    """
    def __init__(self, env_fns, num_workers=None):
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        self.action_space = self.envs[0].action_space
        self.observation_space = self.envs[0].observation_space
        
        # Create thread pool for parallel stepping
        self.num_workers = num_workers or min(self.num_envs, 8)
        self.executor = ThreadPoolExecutor(max_workers=self.num_workers)
        
    def reset(self):
        """Reset all environments in parallel"""
        futures = [self.executor.submit(env.reset) for env in self.envs]
        return [f.result()[0] for f in futures]
    
    def step(self, actions):
        """
        ✓ SOLUTION: Parallel stepping - submit all steps at once
        Total time ≈ max(T1, T2, T3, ..., TN) not the sum
        
        This is the key change: use ThreadPoolExecutor to run environment
        steps concurrently. ROS2 communication often involves I/O wait,
        so parallelization helps significantly.
        """
        def step_env(env_idx):
            """Step a single environment"""
            env = self.envs[env_idx]
            action = actions[env_idx]
            obs, reward, done, trunc, info = env.step(action)
            # if done:
            #     obs = env.reset()[0]
            return (obs, reward, done, trunc, info)
        
        # Submit all environment steps in parallel
        futures = [
            self.executor.submit(step_env, i) 
            for i in range(self.num_envs)
        ]
        
        # Collect results in original order
        results = [None] * self.num_envs
        for idx, future in enumerate(futures):
            results[idx] = future.result()
        
        return results
    
    def __del__(self):
        """Clean up thread pool"""
        self.executor.shutdown(wait=False)


class DummyVecEnv:
    """
    Serial stepping 
    =========================================================
    This mimics the traditional DummyVecEnv behavior for comparison.
    """
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        self.action_space = self.envs[0].action_space
        self.observation_space = self.envs[0].observation_space

    def reset(self):
        return [env.reset()[0] for env in self.envs]

    def step(self, actions):
        results = []
        for env, action in zip(self.envs, actions):
            obs, reward, done, trunc, info = env.step(action)
            results.append((obs, reward, done, trunc, info))
        return results


class ProcessVecEnv:
    """
    Parallel stepping with multiprocessing
    =========================================================
    Uses one process per environment to avoid GIL/ROS2 contention.
    This maximizes total FPS if your hardware can handle it.
    """
    def __init__(self, env_ids):
        self.env_ids = list(env_ids)
        self.num_envs = len(self.env_ids)
        self.ctx = get_context("spawn")
        self.parents = []
        self.procs = []
        for env_id in self.env_ids:
            parent_conn, child_conn = Pipe()
            proc = self.ctx.Process(target=_env_worker, args=(child_conn, env_id), daemon=False)
            proc.start()
            self.parents.append(parent_conn)
            self.procs.append(proc)

        # receive initial reset
        for parent in self.parents:
            _ = parent.recv()

        # initialize spaces from a local env
        tmp_env = HoverEnv(self.env_ids[0])
        self.action_space = tmp_env.action_space
        self.observation_space = tmp_env.observation_space
        del tmp_env

    def reset(self):
        for parent in self.parents:
            parent.send(("reset", None))
        return [parent.recv()[1] for parent in self.parents]

    def step(self, actions):
        for parent, action in zip(self.parents, actions):
            parent.send(("step", action))
        return [parent.recv()[1] for parent in self.parents]

    def close(self):
        for parent in self.parents:
            parent.send(("close", None))
        for proc in self.procs:
            proc.join(timeout=1.0)

    def __del__(self):
        self.close()


def benchmark_n_envs(n_envs, n_steps, warmup, backend="process"):
    """Benchmark using PARALLEL stepping ."""
    print(f"\n{'='*60}")
    print(f"PARALLEL STEPPING")
    print(f"{'='*60}")
    print(f"Creating {n_envs} environments...")
    
    if backend == "thread":
        env = VecEnv([lambda i=i: HoverEnv(i) for i in range(n_envs)])
    else:
        env = ProcessVecEnv(range(n_envs))
    env.reset()
    action = 0.0 * np.array([env.action_space.sample() for _ in range(n_envs)])
    # warmup
    for _ in range(warmup):
        __ = env.step(action)
    monitor = HardwareMonitor(interval=0.1)
    monitor.start()
    print(f"Starting {n_steps} steps...")
    t = time.perf_counter()
    for _ in range(n_steps):
        __ = env.step(action)
    duration = time.perf_counter() - t
    monitor.stop()
    monitor.join()
    avg_cpu, avg_gpu, avg_ram = monitor.get_averages()
    fps = n_steps * n_envs / duration
    
    print(f"Duration: {duration:.2f}s")
    print(f"Total FPS: {fps:.2f} (env-steps/sec)")
    
    del env
    return fps, avg_cpu, avg_gpu, avg_ram

def benchmark_n_dummy_envs(n_envs, n_steps, warmup):
    """Benchmark using SERIAL stepping ."""
    print(f"\n{'='*60}")
    print(f"DUMMY VEC ENV (SERIAL STEPPING)")
    print(f"{'='*60}")
    print(f"Creating {n_envs} environments...")

    env = DummyVecEnv([lambda i=i: HoverEnv(i) for i in range(n_envs)])
    env.reset()
    action = 0.0 * np.array([env.action_space.sample() for _ in range(n_envs)])
    # warmup
    for _ in range(warmup):
        __ = env.step(action)
    monitor = HardwareMonitor(interval=0.1)
    monitor.start()
    print(f"Starting {n_steps} steps...")
    t = time.perf_counter()
    for _ in range(n_steps):
        __ = env.step(action)
    duration = time.perf_counter() - t
    monitor.stop()
    monitor.join()
    avg_cpu, avg_gpu, avg_ram = monitor.get_averages()
    fps = n_steps * n_envs / duration

    print(f"Duration: {duration:.2f}s")
    print(f"Total FPS: {fps:.2f} (env-steps/sec)")

    del env
    return fps, avg_cpu, avg_gpu, avg_ram

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark VecEnv implementations")
    parser.add_argument("--n-envs", type=int,  required=True,default=1, help="Number of parallel environments to create")
    parser.add_argument("--n-steps", type=int, required=True, default=10000, help="Number of steps to run in the benchmark")
    parser.add_argument("--warmup", type=int, required=True, default=1000, help="Warmup steps, before actual benchmark")
    parser.add_argument("--backend",  required=True, choices=["process", "thread"], default="process", help="Backend for parallel stepping")
    parser.add_argument("--out-file", required=True, type=str, default="results.csv", help="file to save benchmark results (csv format)")
    parser.add_argument("--benchmark-dummy", action="store_true", help="Benchamrk using dummyvecenv")
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

    if not args.benchmark_dummy:
        fps,cpu_util, gpu_util, ram_util = benchmark_n_envs(n_envs, n_steps, warmup, backend=args.backend)
    else:
        fps,cpu_util, gpu_util, ram_util = benchmark_n_dummy_envs(n_envs, n_steps, warmup)
    
    save_results([n_envs, n_steps, fps, cpu_util, gpu_util, ram_util], args.out_file)
    print("completed")
    os._exit(0)
