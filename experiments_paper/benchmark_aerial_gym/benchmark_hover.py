#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Amal Dev Haridevan

"""Aerial Gym hover benchmark matched to the GzDRL/OmniDrones task."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import time

# Isaac Gym must be imported before torch.
import isaacgym  # noqa: F401
import torch

from experiments_paper.gpu_cpu_util import HardwareMonitor

from aerial_gym.config.env_config.empty_env import EmptyEnvCfg
from aerial_gym.config.sim_config.base_sim_config import BaseSimConfig
from aerial_gym.registry.env_registry import env_config_registry
from aerial_gym.registry.sim_registry import sim_config_registry
from aerial_gym.sim.sim_builder import SimBuilder

from experiments_paper.benchmark_aerial_gym.task_math import (
    quat_xyzw_to_matrix,
    quat_xyzw_to_yaw,
    yaw_to_quat_xyzw,
)


MAX_OMEGA_RAD_S = 2300.0
ACTION_HISTORY_DIVISOR = 1300.0
MAX_STEPS = 10000

# Fixed midpoint of Aerial Gym's published base-quad range.
AERIAL_GYM_THRUST_CONSTANT = (
    0.00000926312 + 0.00001826312
) * 0.5


def configure_benchmark_configs(dt: float):
    """Configure Aerial Gym's built-in base_sim and empty_env in-place.

    Using the built-in registry entries is more compatible across Aerial Gym
    revisions than registering new names. Each benchmark invocation runs in a
    fresh Python process, so these process-local class mutations cannot affect
    another benchmark run.
    """
    sim_names = list(sim_config_registry.get_sim_names())
    env_names = list(env_config_registry.get_env_names())

    print(
        "[AerialGym benchmark] registered simulations:",
        sim_names,
    )
    print(
        "[AerialGym benchmark] registered environments:",
        env_names,
    )

    if "base_sim" not in sim_names:
        raise RuntimeError(
            "Aerial Gym did not register 'base_sim'. "
            f"Available simulations: {sim_names}"
        )
    if "empty_env" not in env_names:
        raise RuntimeError(
            "Aerial Gym did not register 'empty_env'. "
            f"Available environments: {env_names}"
        )

    sim_cfg = sim_config_registry.make_sim("base_sim")
    env_cfg = env_config_registry.make_env("empty_env")

    # Common simulation fields present in both older and newer revisions.
    if not hasattr(sim_cfg, "sim"):
        raise RuntimeError(
            "Registered base_sim config has no nested 'sim' block."
        )

    sim_cfg.sim.dt = dt
    sim_cfg.sim.substeps = 1
    sim_cfg.sim.use_gpu_pipeline = True

    if hasattr(sim_cfg, "viewer"):
        sim_cfg.viewer.headless = True

    # Newer Aerial Gym revisions expose this nested block. Older revisions
    # configure PhysX elsewhere, so leave their native settings untouched.
    if hasattr(sim_cfg, "physx"):
        if hasattr(sim_cfg.physx, "num_threads"):
            sim_cfg.physx.num_threads = 10
        if hasattr(sim_cfg.physx, "solver_type"):
            sim_cfg.physx.solver_type = 1
        if hasattr(sim_cfg.physx, "num_position_iterations"):
            sim_cfg.physx.num_position_iterations = 4
        if hasattr(sim_cfg.physx, "num_velocity_iterations"):
            sim_cfg.physx.num_velocity_iterations = 1

    if not hasattr(env_cfg, "env"):
        raise RuntimeError(
            "Registered empty_env config has no nested 'env' block."
        )

    # Match the benchmark scene and one-physics-step-per-control-step setup.
    # Guard fields that did not exist in some older Aerial Gym revisions.
    env_overrides = {
        "env_spacing": 15.0,
        "lower_bound_min": [-15.0, -15.0, -2.0],
        "lower_bound_max": [-15.0, -15.0, -2.0],
        "upper_bound_min": [15.0, 15.0, 15.0],
        "upper_bound_max": [15.0, 15.0, 15.0],
        "num_physics_steps_per_env_step_mean": 1,
        "num_physics_steps_per_env_step_std": 0,
        "render_viewer_every_n_steps": 1000000000,
        "reset_on_collision": False,
        "create_ground_plane": True,
        "use_warp": False,
        "write_to_sim_at_every_timestep": False,
    }
    for name, value in env_overrides.items():
        if hasattr(env_cfg.env, name):
            setattr(env_cfg.env, name, value)

    return "base_sim", "empty_env"


class HoverTask:
    def __init__(self, n_envs: int, dt: float, seed: int):
        sim_name, env_name = configure_benchmark_configs(dt)
        self.device = torch.device("cuda:0")
        self.num_envs = n_envs

        self.env = SimBuilder().build_env(
            sim_name=sim_name,
            env_name=env_name,
            robot_name="base_quadrotor",
            controller_name="no_control",
            device="cuda:0",
            num_envs=n_envs,
            headless=True,
            use_warp=False,
        )

        self.state = self.env.global_tensor_dict["robot_state_tensor"]
        self.position = self.state[:, :3]
        self.quat_xyzw = self.state[:, 3:7]
        self.velocity = self.state[:, 7:10]
        self.angular_velocity = self.state[:, 10:13]

        origins = []
        gym = self.env.IGE_env.gym
        for handle in self.env.IGE_env.env_handles:
            o = gym.get_env_origin(handle)
            origins.append([o.x, o.y, o.z])
        self.origins = torch.tensor(
            origins, dtype=torch.float32, device=self.device
        )

        self.target_position = torch.zeros(
            n_envs, 3, dtype=torch.float32, device=self.device
        )
        self.target_yaw = torch.zeros(
            n_envs, dtype=torch.float32, device=self.device
        )
        self.action_buffer1 = torch.zeros(
            n_envs, 4, dtype=torch.float32, device=self.device
        )
        self.action_buffer2 = torch.zeros_like(self.action_buffer1)
        self.step_count = torch.zeros(
            n_envs, dtype=torch.long, device=self.device
        )

        # Make Aerial Gym's motor model deterministic and effectively
        # instantaneous, matching direct target rotor-speed commands.
        motor_model = (
            self.env.robot_manager.robot.control_allocator.motor_model
        )
        motor_model.integration_scheme = "euler"
        motor_model.motor_thrust_constant.fill_(
            AERIAL_GYM_THRUST_CONSTANT
        )
        motor_model.motor_time_constants_increasing.zero_()
        motor_model.motor_time_constants_decreasing.zero_()
        motor_model.current_motor_thrust.zero_()
        self.motor_model = motor_model

        torch.manual_seed(seed)
        self.reset_ids(
            torch.arange(n_envs, device=self.device)
        )

    def reset_ids(self, ids: torch.Tensor):
        if ids.numel() == 0:
            return

        count = ids.numel()
        spawn = torch.empty(
            count, 3, device=self.device, dtype=torch.float32
        )
        spawn[:, :2] = torch.rand(
            count, 2, device=self.device
        ) * 10.0 - 5.0
        spawn[:, 2] = torch.rand(count, device=self.device) * 3.0
        spawn_yaw = torch.rand(
            count, device=self.device
        ) * 1.44 - 0.72

        target = torch.empty_like(spawn)
        target[:, :2] = torch.rand(
            count, 2, device=self.device
        ) * 10.0 - 5.0
        target[:, 2] = torch.rand(count, device=self.device) * 3.0
        target_yaw = torch.rand(
            count, device=self.device
        ) * 1.44 - 0.72

        self.target_position[ids] = target
        self.target_yaw[ids] = target_yaw

        self.state[ids, :3] = spawn + self.origins[ids]
        self.state[ids, 3:7] = yaw_to_quat_xyzw(spawn_yaw)
        self.state[ids, 7:13] = 0.0
        self.action_buffer1[ids] = 0.0
        self.action_buffer2[ids] = 0.0
        self.step_count[ids] = 0
        self.motor_model.current_motor_thrust[ids] = 0.0
        self.env.IGE_env.write_to_sim()

    def step(self, normalized_action: torch.Tensor):
        normalized_action = torch.nan_to_num(
            normalized_action,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )
        omega = (
            (normalized_action + 1.0)
            * MAX_OMEGA_RAD_S
            * 0.5
        )

        self.action_buffer2.copy_(self.action_buffer1)
        self.action_buffer1.copy_(omega)

        # Aerial Gym's base motor model uses rotations/sec when use_rps=True.
        rps = omega / (2.0 * math.pi)
        motor_thrust = AERIAL_GYM_THRUST_CONSTANT * rps.square()

        self.step_count += 1
        self.env.step(actions=motor_thrust)

        local_pos = self.position - self.origins
        rotation = quat_xyzw_to_matrix(self.quat_xyzw)
        yaw = quat_xyzw_to_yaw(self.quat_xyzw)
        rel_pos = self.target_position - local_pos
        yaw_error = self.target_yaw - yaw

        # Exact 22-value GzDRL/OmniDrones observation convention.
        quat_wxyz = self.quat_xyzw[:, [3, 0, 1, 2]]
        obs = torch.cat(
            (
                rel_pos,
                yaw_error[:, None],
                self.velocity,
                quat_wxyz,
                self.angular_velocity,
                self.action_buffer1 / ACTION_HISTORY_DIVISOR - 1.0,
                self.action_buffer2 / ACTION_HISTORY_DIVISOR - 1.0,
            ),
            dim=-1,
        )

        state_error = 1.6 * 1.6 * (
            rel_pos.square().sum(dim=-1)
            + yaw_error.square()
        )
        reward = (
            1.0 / (1.0 + state_error)
            + 0.01
            / (
                1.0
                + self.angular_velocity.square().sum(dim=-1)
            )
        )

        tilt = torch.acos(
            rotation[:, 2, 2].clamp(-1.0, 1.0)
        ).abs()
        done = (
            (self.step_count >= MAX_STEPS)
            | (yaw.abs() > 1.57)
            | (local_pos.abs() > 7.0).any(dim=-1)
            | (tilt > 1.57)
        )

        # Benchmark output ignores transition data, but reset failed worlds so
        # the timed workload remains representative of a vectorized RL env.
        ids = done.nonzero(as_tuple=False).squeeze(-1)
        if ids.numel() > 0:
            self.reset_ids(ids)

        return obs, reward, done


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
    return p.parse_args()


def main():
    args = parse_args()
    task = HoverTask(args.n_envs, args.physics_dt, args.seed)
    action = torch.zeros(
        args.n_envs, 4, dtype=torch.float32, device="cuda:0"
    )

    with torch.no_grad():
        for _ in range(args.warmup):
            task.step(action)

        torch.cuda.synchronize()
        monitor = HardwareMonitor(interval=0.1)
        monitor.start()
        start = time.perf_counter()

        for _ in range(args.n_steps):
            task.step(action)

        torch.cuda.synchronize()
        duration = time.perf_counter() - start
        monitor.stop()
        monitor.join()
        avg_cpu, avg_gpu, _ = monitor.get_averages()

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


if __name__ == "__main__":
    main()
