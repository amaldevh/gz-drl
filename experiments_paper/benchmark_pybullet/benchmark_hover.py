#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Amal Dev Haridevan

from __future__ import annotations

import argparse
import csv
from functools import partial
from pathlib import Path
import time

import gymnasium as gym
import numpy as np

from experiments_paper.gpu_cpu_util import HardwareMonitor
from experiments_paper.benchmark_pybullet.gzdrl_pybullet_envs import (
    GzDRLHoverAviary,
)


def make_env(index: int, physics_dt: float, seed: int):
    return GzDRLHoverAviary(
        physics_dt=physics_dt,
        seed=seed + index,
    )


def append_result(path, n_envs, n_steps, fps, avg_cpu, avg_gpu):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(
                ["n_envs", "n_steps", "fps", "avg_cpu", "avg_gpu"]
            )
        writer.writerow([n_envs, n_steps, fps, avg_cpu, avg_gpu])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-envs", type=int, required=True)
    p.add_argument("--n-steps", type=int, default=10000)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--physics-dt", type=float, default=0.001)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-file", required=True)
    p.add_argument(
        "--context",
        choices=("spawn", "forkserver", "fork"),
        default="spawn",
    )
    return p.parse_args()


def main():
    args = parse_args()
    env_fns = [
        partial(make_env, i, args.physics_dt, args.seed)
        for i in range(args.n_envs)
    ]

    env = gym.vector.AsyncVectorEnv(
        env_fns,
        context=args.context,
    )
    try:
        env.reset(seed=args.seed)
        action = np.zeros(
            (args.n_envs,) + env.single_action_space.shape,
            dtype=np.float32,
        )

        for _ in range(args.warmup):
            env.step(action)

        monitor = HardwareMonitor(interval=0.1)
        monitor.start()
        start = time.perf_counter()

        for _ in range(args.n_steps):
            env.step(action)

        duration = time.perf_counter() - start
        monitor.stop()
        monitor.join()
        avg_cpu, avg_gpu , avg_ram= monitor.get_averages()

        fps = args.n_envs * args.n_steps / duration
        append_result(
            args.out_file,
            args.n_envs,
            args.n_steps,
            fps,
            avg_cpu,
            avg_gpu,
        )

        print(f"Duration: {duration:.4f} s")
        print(f"FPS: {fps:.2f} environment steps/s")
        print(f"Average CPU utilization: {avg_cpu}")
        print(f"Average GPU utilization: {avg_gpu}")
        print(f"CSV: {args.out_file}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
