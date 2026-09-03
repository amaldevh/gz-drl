#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Amal Dev Haridevan

"""Benchmark the OmniDrones equivalent of MultiAgentFormationLLEnv.

The output CSV uses the same five-column structure as the supplied
GzDRL hover benchmark:

    n_drones,n_steps,fps,avg_cpu,avg_gpu
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark OmniDrones multi-agent formation throughput."
        )
    )
    parser.add_argument(
        "--n-drones",
        type=int,
        required=True,
        help="Number of drones in each multi-agent environment.",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=32,
        help="Number of parallel environments. Default: 32.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=10000,
        help="Number of measured vector steps.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1000,
        help="Number of unmeasured vector warmup steps.",
    )
    parser.add_argument(
        "--physics-dt",
        type=float,
        default=0.001,
        help=(
            "Physics timestep. Set this equal to the Gazebo "
            "world's max_step_size."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Environment seed.",
    )
    parser.add_argument(
        "--monitor-interval",
        type=float,
        default=0.1,
        help="HardwareMonitor sample interval.",
    )
    parser.add_argument(
        "--out-file",
        required=True,
        type=Path,
        help="CSV file to append.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="OmniDrones repository root.",
    )
    return parser.parse_args()


def compose_config(args: argparse.Namespace, repo_root: Path):
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
            overrides=["task=GazeboMultiAgentFormationLL"],
        )

    OmegaConf.set_struct(cfg, False)
    cfg.headless = True

    cfg.task.num_agents = args.n_drones

    cfg.env.num_envs = args.n_envs
    cfg.env.env_spacing = 30
    cfg.env.max_episode_length = 20000

    cfg.sim.dt = args.physics_dt
    cfg.sim.substeps = 1
    cfg.sim.device = "cuda:0"
    cfg.sim.use_gpu = True
    cfg.sim.use_gpu_pipeline = True

    # Some OmniDrones task compositions retain task-local copies.
    if "env" in cfg.task:
        cfg.task.env.num_envs = args.n_envs
        cfg.task.env.env_spacing = 30
        cfg.task.env.max_episode_length = 20000
    if "sim" in cfg.task:
        cfg.task.sim.dt = args.physics_dt
        cfg.task.sim.substeps = 1
        cfg.task.sim.device = "cuda:0"
        cfg.task.sim.use_gpu = True
        cfg.task.sim.use_gpu_pipeline = True

    OmegaConf.resolve(cfg)
    return cfg


def append_result(
    filename: Path,
    n_drones: int,
    n_steps: int,
    fps: float,
    avg_cpu,
    avg_gpu,
    avg_ram
) -> None:
    """Append one row using the common five-column benchmark schema."""
    filename.parent.mkdir(parents=True, exist_ok=True)
    write_header = not filename.exists()

    with filename.open("a", newline="") as stream:
        writer = csv.writer(stream)
        if write_header:
            writer.writerow(
                ["n_drones", "n_steps", "fps", "avg_cpu", "avg_gpu", "avg_ram"]
            )
        writer.writerow([n_drones, n_steps, fps, avg_cpu, avg_gpu, avg_ram])


def main() -> int:
    args = parse_args()

    if not 2 <= args.n_drones <= 20:
        raise ValueError("--n-drones must be in [2, 20].")
    if args.n_envs <= 0:
        raise ValueError("--n-envs must be positive.")
    if args.n_steps <= 0:
        raise ValueError("--n-steps must be positive.")
    if args.warmup < 0:
        raise ValueError("--warmup cannot be negative.")
    if args.physics_dt <= 0:
        raise ValueError("--physics-dt must be positive.")

    script_path = Path(__file__).resolve()
    repo_root = (
        args.repo_root.resolve()
        if args.repo_root is not None
        else script_path.parents[1]
    )
    cfg = compose_config(args, repo_root)

    # SimulationApp must start before importing OmniDrones environment code.
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

        from omni_drones.envs.single.gazebo_multiagent_formation_ll import (
            GazeboMultiAgentFormationLL,
        )

        torch.set_grad_enabled(False)

        print("=" * 68)
        print("OMNIDRONES MULTI-AGENT FORMATION BENCHMARK")
        print("=" * 68)
        print(f"Parallel environments: {args.n_envs}")
        print(f"Drones per environment: {args.n_drones}")
        print(f"Measured vector steps: {args.n_steps}")
        print(f"Warmup vector steps: {args.warmup}")

        env = GazeboMultiAgentFormationLL(
            cfg,
            headless=True,
        )
        env.set_seed(args.seed)
        env.reset()

        # Zero normalized action matches the supplied benchmark convention.
        # Build the complete nested TensorDict required by EnvBase.step().
        # env.action_spec.zero() returns only the leaf action Tensor.
        action = env.full_action_spec.zero()

        for _ in range(args.warmup):
            env.step(action)

        torch.cuda.synchronize()

        monitor = HardwareMonitor(
            interval=args.monitor_interval
        )
        monitor.start()

        start = time.perf_counter()
        for _ in range(args.n_steps):
            env.step(action)
        torch.cuda.synchronize()
        duration = time.perf_counter() - start

        monitor.stop()
        monitor.join()
        avg_cpu, avg_gpu, avg_ram = monitor.get_averages()

        # FPS is environment transitions per second, not agent transitions.
        fps = args.n_envs * args.n_steps / duration
        append_result(
            args.out_file,
            args.n_drones,
            args.n_steps,
            fps,
            avg_cpu,
            avg_gpu,
            avg_ram
        )

        print(f"Duration: {duration:.4f} s")
        print(f"FPS: {fps:.2f} environment steps/s")
        print(f"Average CPU utilization: {avg_cpu}")
        print(f"Average GPU utilization: {avg_gpu}")
        print(f"CSV: {args.out_file}")
        return 0
    except Exception as e:
        print(e)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
