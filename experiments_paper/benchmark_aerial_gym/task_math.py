#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Amal Dev Haridevan

"""Shared task math for the Aerial Gym benchmarks."""
from __future__ import annotations

import math
import torch


def quat_xyzw_to_matrix(q: torch.Tensor) -> torch.Tensor:
    q = q / torch.linalg.vector_norm(
        q, dim=-1, keepdim=True
    ).clamp_min(1.0e-8)
    x, y, z, w = q.unbind(dim=-1)

    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z

    return torch.stack(
        (
            1.0 - 2.0*(yy+zz),
            2.0*(xy-wz),
            2.0*(xz+wy),
            2.0*(xy+wz),
            1.0 - 2.0*(xx+zz),
            2.0*(yz-wx),
            2.0*(xz-wy),
            2.0*(yz+wx),
            1.0 - 2.0*(xx+yy),
        ),
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)


def quat_xyzw_to_yaw(q: torch.Tensor) -> torch.Tensor:
    x, y, z, w = q.unbind(dim=-1)
    return torch.atan2(
        2.0 * (w*z + x*y),
        1.0 - 2.0 * (y*y + z*z),
    )


def yaw_to_quat_xyzw(yaw: torch.Tensor) -> torch.Tensor:
    out = torch.zeros(
        *yaw.shape, 4,
        dtype=torch.float32,
        device=yaw.device,
    )
    out[..., 2] = torch.sin(0.5 * yaw)
    out[..., 3] = torch.cos(0.5 * yaw)
    return out


def generate_target_formation(
    num_agents: int,
    device: torch.device,
) -> torch.Tensor:
    if not 2 <= num_agents <= 20:
        raise ValueError("num_agents must be in [2, 20].")
    target = torch.zeros(
        num_agents, 3, dtype=torch.float32, device=device
    )
    has_center = num_agents >= 5
    perimeter = num_agents - 1 if has_center else num_agents
    start = 1 if has_center else 0
    for i in range(perimeter):
        angle = i * 2.0 * math.pi / perimeter
        target[start+i, 0] = math.cos(angle)
        target[start+i, 1] = math.sin(angle)
    target -= target.mean(dim=0, keepdim=True)
    return target


def formation_reward_and_done(
    position: torch.Tensor,
    rotation: torch.Tensor,
    target_position: torch.Tensor,
    target_formation: torch.Tensor,
    step_count: torch.Tensor,
    max_steps: int = 20000,
):
    n_envs, num_agents, _ = position.shape
    mean_position = position.mean(dim=1, keepdim=True)
    centered = position - mean_position

    desired = target_formation.unsqueeze(0).expand(
        n_envs, -1, -1
    )
    d = torch.cdist(centered, desired)
    d1 = d.min(dim=-1).values.max(dim=-1).values
    d2 = d.min(dim=-2).values.max(dim=-1).values
    hausdorff = torch.maximum(d1, d2)

    target_distance = torch.linalg.vector_norm(
        mean_position - target_position[:, None, :],
        dim=-1,
    ).squeeze(-1)

    reward_formation = 1.0 / (
        1.0 + (hausdorff * 1.6).square()
    )
    reward_pos = torch.exp(-target_distance)

    pair = torch.cdist(position, position)
    diagonal = torch.eye(
        num_agents, dtype=torch.bool, device=position.device
    )[None]
    pair = pair.masked_fill(diagonal, float("inf"))
    min_sep = pair.amin(dim=(-2, -1))

    reward_sep = (min_sep / 0.5).square().clamp(0.0, 1.0)
    reward_heading = rotation[..., 0, 0].mean(dim=-1)

    reward = reward_sep * (
        reward_formation
        + reward_formation * (reward_pos + reward_heading)
        + 0.4 * reward_pos
    )

    excessive_tilt = (
        torch.acos(rotation[..., 2, 2].clamp(-1.0, 1.0))
        > 1.57
    ).any(dim=-1)
    collision = min_sep < 0.23
    done = collision | excessive_tilt | (step_count >= max_steps)
    return reward, done


def multiagent_observation(
    position: torch.Tensor,
    velocity: torch.Tensor,
    angular_velocity: torch.Tensor,
    rotation: torch.Tensor,
    target_position: torch.Tensor,
) -> torch.Tensor:
    n_envs, num_agents, _ = position.shape
    obs_dim = 18 + (num_agents - 1) * 16
    rot_flat = rotation.flatten(-2)

    self_features = torch.cat(
        (
            target_position[:, None, :] - position,
            velocity,
            rot_flat,
            angular_velocity,
        ),
        dim=-1,
    )

    # [E, self_i, other_j, 3] = p_j - p_i
    relative = position[:, None, :, :] - position[:, :, None, :]
    distance = torch.linalg.vector_norm(
        relative, dim=-1, keepdim=True
    )
    other_velocity = velocity[:, None, :, :].expand(
        -1, num_agents, -1, -1
    )
    other_rotation = rot_flat[:, None, :, :].expand(
        -1, num_agents, -1, -1
    )

    all_other = torch.cat(
        (relative, distance, other_velocity, other_rotation),
        dim=-1,
    )
    mask = ~torch.eye(
        num_agents,
        dtype=torch.bool,
        device=position.device,
    )
    other = all_other[:, mask].reshape(
        n_envs, num_agents, (num_agents - 1) * 16
    )
    obs = torch.cat((self_features, other), dim=-1)
    if obs.shape[-1] != obs_dim:
        raise RuntimeError(
            f"Expected observation dim {obs_dim}, got {obs.shape[-1]}"
        )
    return obs
