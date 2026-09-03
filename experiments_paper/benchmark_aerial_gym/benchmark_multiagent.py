#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Amal Dev Haridevan

"""Multi-agent Aerial Gym benchmark.

Aerial Gym's stock EnvManager exposes one controlled robot state per cloned
environment. This benchmark therefore creates N copies of Aerial Gym's
base-quad asset in each Isaac Gym environment while retaining Aerial Gym's
published quad asset, motor geometry, and motor constants.

All drones belonging to one logical environment share the same PhysX world
and can physically collide.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
import time

# Must precede torch.
from isaacgym import gymapi, gymtorch
import torch

from aerial_gym import AERIAL_GYM_DIRECTORY

from experiments_paper.gpu_cpu_util import HardwareMonitor
from experiments_paper.benchmark_aerial_gym.task_math import (
    formation_reward_and_done,
    generate_target_formation,
    multiagent_observation,
    quat_xyzw_to_matrix,
    yaw_to_quat_xyzw,
)


MAX_OMEGA_RAD_S = 2300.0
THRUST_CONSTANT = (
    0.00000926312 + 0.00001826312
) * 0.5
THRUST_TO_TORQUE = 0.01
MOTOR_DIRECTIONS = (1.0, -1.0, 1.0, -1.0)
MOTOR_BODY_LOCAL_INDICES = (5, 6, 7, 8)


class MultiRobotAerialGymBackend:
    def __init__(
        self,
        n_envs: int,
        n_agents: int,
        dt: float,
        seed: int,
    ):
        if not 2 <= n_agents <= 20:
            raise ValueError("n_agents must be in [2, 20].")

        self.n_envs = n_envs
        self.n_agents = n_agents
        self.dt = dt
        self.device = torch.device("cuda:0")
        self.gym = gymapi.acquire_gym()

        sim_params = gymapi.SimParams()
        sim_params.dt = dt
        sim_params.substeps = 1
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        sim_params.use_gpu_pipeline = True
        sim_params.physx.use_gpu = True
        sim_params.physx.num_threads = 10
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.contact_offset = 0.002
        sim_params.physx.rest_offset = 0.001
        sim_params.physx.max_depenetration_velocity = 1.0

        self.sim = self.gym.create_sim(
            0, -1, gymapi.SIM_PHYSX, sim_params
        )
        if self.sim is None:
            raise RuntimeError("Isaac Gym create_sim() failed.")

        plane = gymapi.PlaneParams()
        plane.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane)

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.disable_gravity = False
        asset_options.collapse_fixed_joints = False
        asset_options.replace_cylinder_with_capsule = False
        asset_options.flip_visual_attachments = True
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.max_linear_velocity = 100.0
        asset_options.armature = 0.001

        asset_root = os.path.join(
            AERIAL_GYM_DIRECTORY, "resources", "robots", "quad"
        )
        self.asset = self.gym.load_asset(
            self.sim,
            asset_root,
            "quad.urdf",
            asset_options,
        )

        body_count = self.gym.get_asset_rigid_body_count(self.asset)
        if body_count <= max(MOTOR_BODY_LOCAL_INDICES):
            raise RuntimeError(
                "Aerial Gym quad asset body layout changed: "
                f"found {body_count} rigid bodies but motor application "
                f"mask requires index {max(MOTOR_BODY_LOCAL_INDICES)}."
            )

        lower = gymapi.Vec3(-15.0, -15.0, -2.0)
        upper = gymapi.Vec3(15.0, 15.0, 15.0)
        per_row = max(1, int(math.sqrt(n_envs)))

        self.env_handles = []
        self.actor_handles = []
        motor_global_indices = []
        actor_global_indices = []
        origins = []

        for env_id in range(n_envs):
            env = self.gym.create_env(
                self.sim, lower, upper, per_row
            )
            self.env_handles.append(env)
            o = self.gym.get_env_origin(env)
            origins.append([o.x, o.y, o.z])

            env_actors = []
            env_actor_indices = []
            env_motor_indices = []
            for agent in range(n_agents):
                pose = gymapi.Transform()
                pose.p = gymapi.Vec3(
                    0.0, 0.0, 1.0 + 0.3 * agent
                )
                actor = self.gym.create_actor(
                    env,
                    self.asset,
                    pose,
                    f"base_quadrotor_{agent}",
                    env_id,
                    0,
                    0,
                )
                env_actors.append(actor)
                env_actor_indices.append(
                    self.gym.get_actor_index(
                        env, actor, gymapi.DOMAIN_SIM
                    )
                )
                env_motor_indices.append(
                    [
                        self.gym.get_actor_rigid_body_index(
                            env,
                            actor,
                            local_idx,
                            gymapi.DOMAIN_SIM,
                        )
                        for local_idx in MOTOR_BODY_LOCAL_INDICES
                    ]
                )

            self.actor_handles.append(env_actors)
            actor_global_indices.append(env_actor_indices)
            motor_global_indices.append(env_motor_indices)

        self.gym.prepare_sim(self.sim)

        self.root_tensor = gymtorch.wrap_tensor(
            self.gym.acquire_actor_root_state_tensor(self.sim)
        )
        self.rigid_body_tensor = gymtorch.wrap_tensor(
            self.gym.acquire_rigid_body_state_tensor(self.sim)
        )

        self.force_tensor = torch.zeros(
            self.rigid_body_tensor.shape[0],
            3,
            dtype=torch.float32,
            device=self.device,
        )
        self.torque_tensor = torch.zeros_like(self.force_tensor)

        self.actor_indices = torch.tensor(
            actor_global_indices,
            dtype=torch.long,
            device=self.device,
        )
        self.motor_indices = torch.tensor(
            motor_global_indices,
            dtype=torch.long,
            device=self.device,
        )
        self.origins = torch.tensor(
            origins, dtype=torch.float32, device=self.device
        )

        self.target_position = torch.zeros(
            n_envs, 3, dtype=torch.float32, device=self.device
        )
        self.target_formation = generate_target_formation(
            n_agents, self.device
        )
        self.step_count = torch.zeros(
            n_envs, dtype=torch.long, device=self.device
        )

        self.motor_directions = torch.tensor(
            MOTOR_DIRECTIONS,
            dtype=torch.float32,
            device=self.device,
        )

        torch.manual_seed(seed)
        self.reset_ids(
            torch.arange(n_envs, device=self.device),
            settle=True,
        )

    def states(self):
        state = self.root_tensor[self.actor_indices]
        local_position = state[..., :3] - self.origins[:, None, :]
        quat = state[..., 3:7]
        velocity = state[..., 7:10]
        angular_velocity = state[..., 10:13]
        return local_position, quat, velocity, angular_velocity

    def _sample_spawns(self, ids):
        e = ids.numel()
        positions = torch.empty(
            e,
            self.n_agents,
            3,
            dtype=torch.float32,
            device=self.device,
        )
        target = self.target_position[ids]

        for agent in range(self.n_agents):
            unresolved = torch.ones(
                e, dtype=torch.bool, device=self.device
            )
            attempts = 0
            while unresolved.any():
                rows = unresolved.nonzero(
                    as_tuple=False
                ).squeeze(-1)
                s = torch.rand(
                    rows.numel(), 3, device=self.device
                ) * 2.0 - 1.0
                candidate = torch.empty_like(s)
                candidate[:, 0] = target[rows, 0] + s[:, 0] * 2.0
                candidate[:, 1] = target[rows, 1] + s[:, 1] * 2.0
                candidate[:, 2] = (
                    target[rows, 2]
                    + s[:, 2].clamp_min(0.0) * 1.5
                    + 0.5
                )

                if agent == 0:
                    valid = torch.ones(
                        rows.numel(),
                        dtype=torch.bool,
                        device=self.device,
                    )
                else:
                    previous = positions[rows, :agent]
                    d = torch.linalg.vector_norm(
                        previous - candidate[:, None, :],
                        dim=-1,
                    )
                    valid = (d >= 0.23).all(dim=-1)

                accepted = rows[valid]
                positions[accepted, agent] = candidate[valid]
                unresolved[accepted] = False

                attempts += 1
                if attempts > 10000:
                    raise RuntimeError(
                        "Spawn rejection sampling failed."
                    )
        return positions

    def reset_ids(self, ids: torch.Tensor, settle: bool = False):
        if ids.numel() == 0:
            return

        s = torch.rand(
            ids.numel(), 3, device=self.device
        ) * 2.0 - 1.0
        target = torch.empty_like(s)
        target[:, 0] = s[:, 0] * 10.0
        target[:, 1] = s[:, 1] * 10.0
        target[:, 2] = s[:, 2].clamp_min(0.0) * 10.0 + 1.0
        self.target_position[ids] = target

        local_pos = self._sample_spawns(ids)
        root_ids = self.actor_indices[ids]
        flat_ids = root_ids.reshape(-1)

        state = self.root_tensor[flat_ids]
        state[:, :3] = (
            local_pos
            + self.origins[ids, None, :]
        ).reshape(-1, 3)
        state[:, 3:7] = 0.0
        state[:, 6] = 1.0  # xyzw identity
        state[:, 7:13] = 0.0
        self.root_tensor[flat_ids] = state
        self.step_count[ids] = 0

        flat_ids_i32 = flat_ids.to(torch.int32).contiguous()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_tensor),
            gymtorch.unwrap_tensor(flat_ids_i32),
            flat_ids_i32.numel(),
        )

        # Source multiagent reset advances ten raw physics ticks.
        if settle:
            for _ in range(10):
                self.gym.simulate(self.sim)
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_actor_root_state_tensor(self.sim)

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
        rps = omega / (2.0 * math.pi)
        thrust = THRUST_CONSTANT * rps.square()

        self.force_tensor.zero_()
        self.torque_tensor.zero_()

        motor_idx = self.motor_indices.reshape(-1)
        flat_thrust = thrust.reshape(-1)

        self.force_tensor[motor_idx, 2] = flat_thrust
        self.torque_tensor[motor_idx, 2] = (
            -THRUST_TO_TORQUE
            * flat_thrust
            * self.motor_directions
            .view(1, 1, 4)
            .expand(self.n_envs, self.n_agents, 4)
            .reshape(-1)
        )

        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self.force_tensor),
            gymtorch.unwrap_tensor(self.torque_tensor),
            gymapi.LOCAL_SPACE,
        )
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_actor_root_state_tensor(self.sim)

        self.step_count += 1

        position, quat, velocity, angular_velocity = self.states()
        rotation = quat_xyzw_to_matrix(quat)

        obs = multiagent_observation(
            position,
            velocity,
            angular_velocity,
            rotation,
            self.target_position,
        )
        reward, done = formation_reward_and_done(
            position,
            rotation,
            self.target_position,
            self.target_formation,
            self.step_count,
        )

        ids = done.nonzero(as_tuple=False).squeeze(-1)
        if ids.numel() > 0:
            self.reset_ids(ids, settle=True)

        return obs, reward, done

    def close(self):
        self.gym.destroy_sim(self.sim)


def append_result(path, n_drones, n_steps, fps, avg_cpu, avg_gpu):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(
                ["n_drones", "n_steps", "fps", "avg_cpu", "avg_gpu"]
            )
        writer.writerow([n_drones, n_steps, fps, avg_cpu, avg_gpu])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-drones", type=int, required=True)
    p.add_argument("--n-envs", type=int, default=32)
    p.add_argument("--n-steps", type=int, default=10000)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--physics-dt", type=float, default=0.001)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-file", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    backend = MultiRobotAerialGymBackend(
        args.n_envs,
        args.n_drones,
        args.physics_dt,
        args.seed,
    )
    action = torch.zeros(
        args.n_envs,
        args.n_drones,
        4,
        dtype=torch.float32,
        device="cuda:0",
    )

    try:
        with torch.no_grad():
            for _ in range(args.warmup):
                backend.step(action)

            torch.cuda.synchronize()
            monitor = HardwareMonitor(interval=0.1)
            monitor.start()
            start = time.perf_counter()

            for _ in range(args.n_steps):
                backend.step(action)

            torch.cuda.synchronize()
            duration = time.perf_counter() - start
            monitor.stop()
            monitor.join()
            avg_cpu, avg_gpu , avg_ram= monitor.get_averages()

        fps = args.n_envs * args.n_steps / duration
        append_result(
            args.out_file,
            args.n_drones,
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
        backend.close()


if __name__ == "__main__":
    main()
