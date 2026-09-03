#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Amal Dev Haridevan

"""Benchmark the GazeboPoolHover equivalent in OmniDrones.

The benchmark excludes application startup, scene construction, and warmup.
It includes the public TorchRL step wrapper by default, environment logic,
PhysX stepping, observation construction, reward computation, and done checks.

Examples
--------
/isaac-sim/python.sh scripts/run_omnidrones_benchmark.py \
    --n-envs 128 \
    --n-steps 10000 \
    --warmup 1000 \
    --physics-dt 0.001 \
    --out-file results.csv
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark OmniDrones GazeboPoolHover throughput."
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        required=True,
        help="Number of synchronously vectorized OmniDrones environments.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=10000,
        help=(
            "Number of measured vector steps. Total measured environment "
            "transitions are n_envs * n_steps."
        ),
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1000,
        help="Number of unmeasured vector steps before timing.",
    )
    parser.add_argument(
        "--physics-dt",
        type=float,
        default=0.001,
        help=(
            "Physics timestep in seconds. Set this to the max_step_size in "
            "the Gazebo world for a simulated-time comparison."
        ),
    )
    parser.add_argument(
        "--substeps",
        type=int,
        default=1,
        help="OmniDrones physics substeps per environment step.",
    )
    parser.add_argument(
        "--env-spacing",
        type=float,
        default=12.0,
        help="Spacing between cloned environments.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Torch/OmniDrones random seed.",
    )
    parser.add_argument(
        "--monitor-interval",
        type=float,
        default=0.1,
        help="HardwareMonitor sampling interval in seconds.",
    )
    parser.add_argument(
        "--step-api",
        choices=("torchrl", "raw"),
        default="torchrl",
        help=(
            "'torchrl' benchmarks env.step(), including TorchRL's public "
            "wrapper. 'raw' benchmarks env._step() only."
        ),
    )
    parser.add_argument(
        "--out-file",
        type=Path,
        required=True,
        help="CSV file to append benchmark results to.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help=(
            "OmniDrones repository root. Defaults to the parent of this "
            "script's scripts directory."
        ),
    )
    return parser.parse_args()


def compose_config(args: argparse.Namespace, repo_root: Path):
    # Hydra and OmegaConf do not import Omniverse modules and are safe before
    # SimulationApp starts.
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)

    GlobalHydra.instance().clear()
    with initialize_config_dir(
        version_base=None,
        config_dir=str((repo_root / "cfg").resolve()),
    ):
        cfg = compose(
            config_name="train",
            overrides=["task=GazeboPoolHover"],
        )

    OmegaConf.set_struct(cfg, False)

    # OmniDrones exposes task-local and top-level copies of environment/sim
    # settings in the composed training configuration. Keep both consistent.
    cfg.headless = True

    cfg.env.num_envs = args.n_envs
    cfg.env.env_spacing = args.env_spacing
    cfg.env.max_episode_length = 10000

    cfg.sim.dt = args.physics_dt
    cfg.sim.substeps = args.substeps
    cfg.sim.device = "cuda:0"
    cfg.sim.use_gpu = True
    cfg.sim.use_gpu_pipeline = True

    if "env" in cfg.task:
        cfg.task.env.num_envs = args.n_envs
        cfg.task.env.env_spacing = args.env_spacing
        cfg.task.env.max_episode_length = 10000
    if "sim" in cfg.task:
        cfg.task.sim.dt = args.physics_dt
        cfg.task.sim.substeps = args.substeps
        cfg.task.sim.device = "cuda:0"
        cfg.task.sim.use_gpu = True
        cfg.task.sim.use_gpu_pipeline = True

    OmegaConf.resolve(cfg)
    return cfg


def append_result(
    filename: Path,
    n_envs: int,
    n_steps: int,
    fps: float,
    avg_cpu,
    avg_gpu,
    avg_ram,
) -> None:
    """Append one row using the supplied GzDRL hover CSV schema."""
    filename.parent.mkdir(parents=True, exist_ok=True)
    write_header = not filename.exists()

    with filename.open("a", newline="") as stream:
        writer = csv.writer(stream)
        if write_header:
            writer.writerow(
                ["n_envs", "n_steps", "fps", "avg_cpu", "avg_gpu", "avg_ram"]
            )
        writer.writerow([n_envs, n_steps, fps, avg_cpu, avg_gpu, avg_ram])


def main() -> int:
    args = parse_args()
    if args.n_envs <= 0:
        raise ValueError("--n-envs must be positive.")
    if args.n_steps <= 0:
        raise ValueError("--n-steps must be positive.")
    if args.warmup < 0:
        raise ValueError("--warmup cannot be negative.")
    if args.physics_dt <= 0:
        raise ValueError("--physics-dt must be positive.")
    if args.substeps <= 0:
        raise ValueError("--substeps must be positive.")

    script_path = Path(__file__).resolve()
    repo_root = (
        args.repo_root.resolve()
        if args.repo_root is not None
        else script_path.parents[1]
    )
    if not (repo_root / "cfg/train.yaml").exists():
        raise FileNotFoundError(
            f"Could not find cfg/train.yaml under {repo_root}. "
            "Pass --repo-root explicitly."
        )

    cfg = compose_config(args, repo_root)

    # Start Isaac Sim before importing the OmniDrones environment modules.
    from isaacsim import SimulationApp

    experience = os.path.join(
        os.environ["EXP_PATH"],
        "omni.isaac.sim.python.gym.headless.kit",
    )
    simulation_app = SimulationApp(
        {
            "headless": True,
            "multi_gpu": False,
            "disable_viewport_updates": True,
            "anti_aliasing": 0,
        },
        experience=experience,
    )

    try:
        import torch
        from gpu_cpu_util import HardwareMonitor

        # Direct import avoids re-enabling unrelated OmniDrones environments.
        from omni_drones.envs.single.gazebo_pool_hover import (
            GazeboPoolHover,
        )

        torch.set_grad_enabled(False)

        print("=" * 72)
        print("OMNIDRONES GAZEBO-POOL HOVER BENCHMARK")
        print("=" * 72)
        print(f"Environments:       {args.n_envs}")
        print(f"Measured vec steps: {args.n_steps}")
        print(f"Warmup vec steps:   {args.warmup}")
        print(f"Physics dt:         {args.physics_dt}")
        print(f"Substeps:           {args.substeps}")
        print(f"Step API:           {args.step_api}")
        print("Creating environment...")

        env = GazeboPoolHover(cfg, headless=True)
        env.set_seed(args.seed)
        env.reset()

        # env.action_spec is the leaf TensorSpec and zero() therefore
        # returns a plain Tensor. EnvBase.step() requires the complete nested
        # TensorDict, which is generated by full_action_spec.
        action_tensordict = env.full_action_spec.zero()
        step_fn = (
            env.step
            if args.step_api == "torchrl"
            else env._step
        )

        # Kernel, PhysX, allocator, and cache warmup.
        for _ in range(args.warmup):
            step_fn(action_tensordict)

        torch.cuda.synchronize()

        monitor = HardwareMonitor(interval=args.monitor_interval)
        monitor.start()

        start = time.perf_counter()
        for _ in range(args.n_steps):
            step_fn(action_tensordict)
        torch.cuda.synchronize()
        duration = time.perf_counter() - start

        monitor.stop()
        monitor.join()
        avg_cpu, avg_gpu, avg_ram = monitor.get_averages()

        total_env_steps = args.n_envs * args.n_steps
        env_steps_per_second = total_env_steps / duration

        append_result(
            args.out_file,
            args.n_envs,
            args.n_steps,
            env_steps_per_second,
            avg_cpu,
            avg_gpu,
            avg_ram
        )

        print(f"Duration:                  {duration:.4f} s")
        print(
            f"Environment steps/sec:     "
            f"{env_steps_per_second:,.2f}"
        )
        print(f"Average CPU utilization:   {avg_cpu}")
        print(f"Average GPU utilization:   {avg_gpu}")
        print(f"Results appended to:       {args.out_file}")
        return 0
    except Exception as e:
        print(e)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
