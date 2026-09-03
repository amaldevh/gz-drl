#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Amal Dev Haridevan

"""GzDRL-equivalent task definitions for gym-pybullet-drones.

These are deliberately custom tasks rather than the package's stock hover
tasks. The goal is to keep action/observation/reward/reset semantics aligned
with the user's GzDRL and OmniDrones benchmarks.

Quaternion convention exposed in observations: wxyz.
PyBullet's internal convention is xyzw and is reordered explicitly.
"""

from __future__ import annotations

import math
from typing import Optional

import gymnasium as gym
import numpy as np
import pybullet as p

from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics


RAD_S_TO_RPM = 60.0 / (2.0 * math.pi)


def _quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return np.asarray([q[3], q[0], q[1], q[2]], dtype=np.float32)


def _rotation_matrix_xyzw(q: np.ndarray) -> np.ndarray:
    return np.asarray(
        p.getMatrixFromQuaternion(q),
        dtype=np.float32,
    ).reshape(3, 3)


def generate_target_formation(num_agents: int) -> np.ndarray:
    if not 2 <= num_agents <= 20:
        raise ValueError("num_agents must be in [2, 20].")

    target = np.zeros((num_agents, 3), dtype=np.float32)
    has_center = num_agents >= 5
    perimeter_agents = num_agents - 1 if has_center else num_agents
    start_index = 1 if has_center else 0

    for i in range(perimeter_agents):
        angle = i * (2.0 * np.pi / perimeter_agents)
        target[start_index + i, 0] = np.cos(angle)
        target[start_index + i, 1] = np.sin(angle)

    target -= target.mean(axis=0, keepdims=True)
    return target


class GzDRLHoverAviary(BaseAviary):
    """Single-drone hover task matched to the supplied GzDRL HoverEnv."""

    OBS_DIM = 22
    ACTION_DIM = 4

    def __init__(
        self,
        physics_dt: float = 0.001,
        seed: int = 0,
    ):
        frequency = round(1.0 / physics_dt)
        if not np.isclose(1.0 / frequency, physics_dt):
            raise ValueError(
                "gym-pybullet-drones uses integer frequencies. "
                "physics_dt must be the reciprocal of an integer."
            )

        self.command_max_rotor_velocity = 2300.0
        self.action_history_divisor = 1300.0
        self.max_steps = 10000
        self.maximum_bounds = np.asarray([7.0, 7.0, 7.0], dtype=np.float32)
        self.max_yaw = 1.57
        self.max_tilt = 1.57
        self._task_step = 0
        self._rng = np.random.default_rng(seed)
        self.desired_pos = np.zeros(3, dtype=np.float32)
        self.desired_yaw = np.float32(0.0)
        self.action_buffer1 = np.zeros(4, dtype=np.float32)
        self.action_buffer2 = np.zeros(4, dtype=np.float32)

        super().__init__(
            drone_model=DroneModel.CF2X,
            num_drones=1,
            initial_xyzs=np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
            initial_rpys=np.zeros((1, 3), dtype=np.float32),
            physics=Physics.PYB,
            pyb_freq=frequency,
            ctrl_freq=frequency,
            gui=False,
            record=False,
            obstacles=False,
            user_debug_gui=False,
            vision_attributes=False,
        )

    def _actionSpace(self):
        return gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.ACTION_DIM,),
            dtype=np.float32,
        )

    def _observationSpace(self):
        return gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.OBS_DIM,),
            dtype=np.float32,
        )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        spawn = np.empty(3, dtype=np.float32)
        spawn[0:2] = self._rng.uniform(-5.0, 5.0, size=2)
        spawn[2] = self._rng.uniform(0.0, 3.0)
        spawn_yaw = self._rng.uniform(-0.72, 0.72)

        self.desired_pos[0:2] = self._rng.uniform(-5.0, 5.0, size=2)
        self.desired_pos[2] = self._rng.uniform(0.0, 3.0)
        self.desired_yaw = np.float32(
            self._rng.uniform(-0.72, 0.72)
        )

        self.INIT_XYZS = spawn.reshape(1, 3)
        self.INIT_RPYS = np.asarray(
            [[0.0, 0.0, spawn_yaw]],
            dtype=np.float32,
        )

        self._task_step = 0
        self.action_buffer1.fill(0.0)
        self.action_buffer2.fill(0.0)

        _, info = super().reset(seed=seed, options=options)
        return self._computeObs(), info

    def _preprocessAction(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(4)
        action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)

        omega_rad_s = (
            (action + 1.0)
            * self.command_max_rotor_velocity
            * 0.5
        ).astype(np.float32)

        self.action_buffer2[:] = self.action_buffer1
        self.action_buffer1[:] = omega_rad_s
        self._task_step += 1

        # BaseAviary's motor interface is RPM, not rad/s.
        return omega_rad_s * RAD_S_TO_RPM

    def _computeObs(self):
        pos = self.pos[0].astype(np.float32)
        vel = self.vel[0].astype(np.float32)
        ang_vel = self.ang_v[0].astype(np.float32)
        quat_wxyz = _quat_xyzw_to_wxyz(self.quat[0])
        yaw = np.float32(self.rpy[0, 2])

        rel_pos = self.desired_pos - pos
        yaw_error = np.float32(self.desired_yaw - yaw)

        obs = np.concatenate(
            (
                rel_pos,
                np.asarray([yaw_error], dtype=np.float32),
                vel,
                quat_wxyz,
                ang_vel,
                self.action_buffer1 / self.action_history_divisor - 1.0,
                self.action_buffer2 / self.action_history_divisor - 1.0,
            )
        ).astype(np.float32)

        if obs.shape != (self.OBS_DIM,):
            raise RuntimeError(f"Unexpected observation shape {obs.shape}.")
        return obs

    def _computeReward(self):
        pos = self.pos[0]
        rel_pos = self.desired_pos - pos
        yaw_error = float(self.desired_yaw - self.rpy[0, 2])
        omega = self.ang_v[0]

        state_error = 1.6 * 1.6 * (
            float(np.dot(rel_pos, rel_pos))
            + yaw_error * yaw_error
        )
        reward_pos = 1.0 / (1.0 + state_error)
        reward_omega = 0.01 / (
            1.0 + float(np.dot(omega, omega))
        )
        return float(reward_pos + reward_omega)

    def _computeTerminated(self):
        pos = self.pos[0]
        yaw = float(self.rpy[0, 2])
        rot = _rotation_matrix_xyzw(self.quat[0])
        tilt = abs(float(np.arccos(np.clip(rot[2, 2], -1.0, 1.0))))

        return bool(
            self._task_step >= self.max_steps
            or abs(yaw) > self.max_yaw
            or np.any(np.abs(pos) > self.maximum_bounds)
            or tilt > self.max_tilt
        )

    def _computeTruncated(self):
        # Preserve the source environment's truncated=False convention.
        return False

    def _computeInfo(self):
        return {}


class GzDRLMultiAgentAviary(BaseAviary):
    """N-drone formation task matched to MultiAgentFormationLLEnv."""

    ACTION_DIM = 4

    def __init__(
        self,
        num_agents: int,
        physics_dt: float = 0.001,
        seed: int = 0,
    ):
        if not 2 <= num_agents <= 20:
            raise ValueError("num_agents must be in [2, 20].")

        frequency = round(1.0 / physics_dt)
        if not np.isclose(1.0 / frequency, physics_dt):
            raise ValueError(
                "physics_dt must be the reciprocal of an integer."
            )

        self.num_agents = int(num_agents)
        self.obs_dim = 18 + (self.num_agents - 1) * 16
        self.command_max_rotor_velocity = 2300.0
        self.max_steps = 20000
        self.safe_distance = 0.5
        self.collision_distance = 0.23
        self.max_tilt = 1.57
        self._task_step = 0
        self._rng = np.random.default_rng(seed)
        self.desired_target_pos = np.zeros(3, dtype=np.float32)
        self.target_formation = generate_target_formation(self.num_agents)

        init_xyz = np.zeros((self.num_agents, 3), dtype=np.float32)
        init_xyz[:, 2] = 1.0 + 0.3 * np.arange(self.num_agents)

        super().__init__(
            drone_model=DroneModel.CF2X,
            num_drones=self.num_agents,
            initial_xyzs=init_xyz,
            initial_rpys=np.zeros((self.num_agents, 3), dtype=np.float32),
            physics=Physics.PYB,
            pyb_freq=frequency,
            ctrl_freq=frequency,
            gui=False,
            record=False,
            obstacles=False,
            user_debug_gui=False,
            vision_attributes=False,
        )

    def _actionSpace(self):
        return gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_agents, self.ACTION_DIM),
            dtype=np.float32,
        )

    def _observationSpace(self):
        return gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_agents, self.obs_dim),
            dtype=np.float32,
        )

    def _sample_spawn(self) -> np.ndarray:
        positions = np.empty(
            (self.num_agents, 3),
            dtype=np.float32,
        )

        for i in range(self.num_agents):
            for _ in range(10000):
                s = self._rng.uniform(-1.0, 1.0, size=3)
                candidate = np.asarray(
                    [
                        self.desired_target_pos[0] + s[0] * 2.0,
                        self.desired_target_pos[1] + s[1] * 2.0,
                        self.desired_target_pos[2]
                        + max(0.0, s[2]) * 1.5
                        + 0.5,
                    ],
                    dtype=np.float32,
                )
                if i == 0:
                    positions[i] = candidate
                    break
                distance = np.linalg.norm(
                    positions[:i] - candidate[None, :],
                    axis=-1,
                )
                if np.all(distance >= self.collision_distance):
                    positions[i] = candidate
                    break
            else:
                raise RuntimeError("Spawn rejection sampling failed.")

        return positions

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        s = self._rng.uniform(-1.0, 1.0, size=3)
        self.desired_target_pos[:] = (
            s[0] * 10.0,
            s[1] * 10.0,
            max(0.0, s[2]) * 10.0 + 1.0,
        )

        self.INIT_XYZS = self._sample_spawn()
        self.INIT_RPYS = np.zeros(
            (self.num_agents, 3),
            dtype=np.float32,
        )
        self._task_step = 0

        _, info = super().reset(seed=seed, options=options)

        # The source C++ reset advances ten raw physics steps after respawn.
        for _ in range(10):
            p.stepSimulation(physicsClientId=self.CLIENT)
        self._updateAndStoreKinematicInformation()

        return self._computeObs(), info

    def _preprocessAction(self, action):
        action = np.asarray(
            action,
            dtype=np.float32,
        ).reshape(self.num_agents, 4)
        action = np.nan_to_num(
            action, nan=0.0, posinf=1.0, neginf=-1.0
        )

        omega_rad_s = (
            (action + 1.0)
            * self.command_max_rotor_velocity
            * 0.5
        ).astype(np.float32)

        self._task_step += 1
        return omega_rad_s * RAD_S_TO_RPM

    def _rotation_matrices(self) -> np.ndarray:
        return np.stack(
            [
                _rotation_matrix_xyzw(self.quat[i])
                for i in range(self.num_agents)
            ],
            axis=0,
        )

    def _computeObs(self):
        rotations = self._rotation_matrices()
        obs = np.empty(
            (self.num_agents, self.obs_dim),
            dtype=np.float32,
        )

        for i in range(self.num_agents):
            cursor = 0
            pos_i = self.pos[i]
            rel_target = self.desired_target_pos - pos_i

            obs[i, cursor : cursor + 3] = rel_target
            cursor += 3
            obs[i, cursor : cursor + 3] = self.vel[i]
            cursor += 3
            obs[i, cursor : cursor + 9] = rotations[i].reshape(-1)
            cursor += 9
            obs[i, cursor : cursor + 3] = self.ang_v[i]
            cursor += 3

            for j in range(self.num_agents):
                if i == j:
                    continue

                p_rel = self.pos[j] - pos_i
                obs[i, cursor : cursor + 3] = p_rel
                cursor += 3
                obs[i, cursor] = np.linalg.norm(p_rel)
                cursor += 1
                obs[i, cursor : cursor + 3] = self.vel[j]
                cursor += 3
                obs[i, cursor : cursor + 9] = rotations[j].reshape(-1)
                cursor += 9

            if cursor != self.obs_dim:
                raise RuntimeError(
                    f"Observation cursor {cursor} != {self.obs_dim}."
                )

        return obs

    def _shared_reward_and_done(self):
        positions = self.pos.astype(np.float32)
        rotations = self._rotation_matrices()

        mean_pos = positions.mean(axis=0)
        centered = positions - mean_pos[None, :]

        dist = np.linalg.norm(
            centered[:, None, :]
            - self.target_formation[None, :, :],
            axis=-1,
        )
        d1 = np.max(np.min(dist, axis=1))
        d2 = np.max(np.min(dist, axis=0))
        cost_h = max(float(d1), float(d2))

        distance_to_target = float(
            np.linalg.norm(mean_pos - self.desired_target_pos)
        )
        reward_formation = 1.0 / (
            1.0 + (cost_h * 1.6) ** 2
        )
        reward_pos = math.exp(-distance_to_target)

        pairwise = np.linalg.norm(
            positions[:, None, :] - positions[None, :, :],
            axis=-1,
        )
        np.fill_diagonal(pairwise, np.inf)
        min_separation = float(np.min(pairwise))

        reward_separation = float(
            np.clip(
                (min_separation / self.safe_distance) ** 2,
                0.0,
                1.0,
            )
        )

        headings = rotations[:, 0, 0]
        reward_heading = float(np.mean(headings))

        excessive_tilt = bool(
            np.any(
                np.arccos(
                    np.clip(rotations[:, 2, 2], -1.0, 1.0)
                )
                > self.max_tilt
            )
        )
        collision = min_separation < self.collision_distance

        reward = reward_separation * (
            reward_formation
            + reward_formation * (reward_pos + reward_heading)
            + 0.4 * reward_pos
        )
        done = (
            self._task_step >= self.max_steps
            or collision
            or excessive_tilt
        )
        return float(reward), bool(done)

    def _computeReward(self):
        reward, _ = self._shared_reward_and_done()
        # The source assigns the identical combined reward to every agent.
        # Gymnasium's single-environment reward field is scalar, so returning
        # the shared value is lossless.
        return reward

    def _computeTerminated(self):
        _, done = self._shared_reward_and_done()
        return done

    def _computeTruncated(self):
        return False

    def _computeInfo(self):
        return {}
