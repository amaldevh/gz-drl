# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Amal Dev Haridevan
#
# Clean-room OmniDrones task adaptation of the supplied
# MultiAgentFormationLLEnv and MultiAgentFormationLLProcessor.

from __future__ import annotations

import math

import torch

from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec

from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.robots.drone import MultirotorBase
from omni_drones.utils.torch import euler_to_quaternion


class GazeboMultiAgentFormationLL(IsaacEnv):
    """OmniDrones equivalent of the supplied low-level formation environment.

    The task preserves the supplied C++ environment's externally visible
    transition workload:

    * N agents, each with four actions in [-1, 1].
    * Gazebo-style rotor target:
        rotor_velocity = (action + 1) / 2 * 2300.
    * Per-agent observation dimension:
        18 + (N - 1) * 16.
    * Shared symmetric-Hausdorff formation reward.
    * Shared collision, excessive-tilt, and horizon termination.
    * One physics advance per action by default.

    The benchmark configuration disables reset-time domain randomization.
    Randomization is outside the timed stepping region in the supplied
    benchmark and does not change the per-step tensor/physics workload.
    """

    ACTION_DIM = 4
    MIN_AGENTS = 2
    MAX_AGENTS = 20

    def __init__(self, cfg, headless: bool):
        self.num_agents = int(cfg.task.get("num_agents", 2))
        if not self.MIN_AGENTS <= self.num_agents <= self.MAX_AGENTS:
            raise ValueError(
                f"num_agents must be in [{self.MIN_AGENTS}, "
                f"{self.MAX_AGENTS}], got {self.num_agents}."
            )

        self.observation_dim = 18 + (self.num_agents - 1) * 16
        self.max_rotor_velocity = float(
            cfg.task.get("max_rotor_velocity", 2300.0)
        )
        self.safe_distance = float(cfg.task.get("safe_distance", 0.5))
        self.collision_distance = float(
            cfg.task.get("collision_distance", 0.23)
        )
        self.max_tilt = float(cfg.task.get("max_tilt", 1.57))
        self.instant_rotor_response = bool(
            cfg.task.get("instant_rotor_response", True)
        )
        self.physics_steps_per_control = int(
            cfg.task.get("physics_steps_per_control", 1)
        )
        if self.physics_steps_per_control != 1:
            raise ValueError(
                "This OmniDrones benchmark supports "
                "physics_steps_per_control=1, matching the supplied default."
            )

        super().__init__(cfg, headless)

        if self.drone.num_rotors != self.ACTION_DIM:
            raise ValueError(
                "GazeboMultiAgentFormationLL requires a four-rotor model; "
                f"{self.cfg.task.drone_model.name} exposes "
                f"{self.drone.num_rotors} rotors."
            )
        if self.controller is not None:
            raise ValueError(
                "drone_model.controller must be null because the supplied "
                "environment commands rotor velocity directly."
            )
        if self.drone.n != self.num_agents:
            raise RuntimeError(
                f"Expected {self.num_agents} spawned drones, got {self.drone.n}."
            )

        self.drone.initialize()

        # DRLServer receives the target rotor velocity directly. RotorGroup
        # normally applies first-order motor lag; tau=1 reaches the requested
        # target in the current simulation step.
        if self.instant_rotor_response:
            self.drone.tau_up.fill_(1.0)
            self.drone.tau_down.fill_(1.0)

        self.initial_velocities = torch.zeros_like(
            self.drone.get_velocities()
        )
        self.desired_target_pos = torch.zeros(
            self.num_envs, 1, 3, dtype=torch.float32, device=self.device
        )
        self.target_formation = self._generate_target_formation(
            self.num_agents
        ).to(self.device)

        self.drone_state = None
        self.rotation_matrix = None
        self.pairwise_distance = None
        self.min_separation = None

    def _design_scene(self):
        import omni_drones.utils.kit as kit_utils

        drone_cfg = self.cfg.task.drone_model
        self.drone, self.controller = MultirotorBase.make(
            drone_cfg.name, drone_cfg.controller
        )

        # Placeholder poses are replaced for every environment during reset.
        placeholder = [
            (0.0, 0.0, 1.0 + 0.3 * i)
            for i in range(self.num_agents)
        ]
        self.drone.spawn(translations=placeholder)

        kit_utils.create_ground_plane(
            "/World/defaultGroundPlane",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )
        return ["/World/defaultGroundPlane"]

    def _set_specs(self):
        self.observation_spec = CompositeSpec(
            {
                "agents": CompositeSpec(
                    {
                        "observation": UnboundedContinuousTensorSpec(
                            (self.num_agents, self.observation_dim),
                            device=self.device,
                        )
                    }
                )
            }
        ).expand(self.num_envs).to(self.device)

        per_agent_action_spec = torch.stack(
            [self.drone.action_spec] * self.num_agents,
            dim=0,
        )
        self.action_spec = CompositeSpec(
            {
                "agents": CompositeSpec(
                    {"action": per_agent_action_spec}
                )
            }
        ).expand(self.num_envs).to(self.device)

        self.reward_spec = CompositeSpec(
            {
                "agents": CompositeSpec(
                    {
                        "reward": UnboundedContinuousTensorSpec(
                            (self.num_agents, 1),
                            device=self.device,
                        )
                    }
                )
            }
        ).expand(self.num_envs).to(self.device)

        self.agent_spec["drone"] = AgentSpec(
            "drone",
            self.num_agents,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
        )

    @staticmethod
    def _generate_target_formation(
        num_agents: int,
    ) -> torch.Tensor:
        target = torch.zeros(
            num_agents, 3, dtype=torch.float32
        )
        has_center = num_agents >= 5
        perimeter_agents = (
            num_agents - 1 if has_center else num_agents
        )
        start_index = 1 if has_center else 0

        for index in range(perimeter_agents):
            angle = (
                float(index)
                * 2.0
                * math.pi
                / float(perimeter_agents)
            )
            target[start_index + index, 0] = math.cos(angle)
            target[start_index + index, 1] = math.sin(angle)

        # Agent zero is already at the origin when a center anchor is used.
        target -= target.mean(dim=0, keepdim=True)
        return target

    def _sample_spawn_positions(
        self,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Match the source's sequential rejection-sampling reset."""
        num_reset_envs = target.shape[0]
        positions = torch.empty(
            num_reset_envs,
            self.num_agents,
            3,
            dtype=torch.float32,
            device=self.device,
        )

        for agent_index in range(self.num_agents):
            unresolved = torch.ones(
                num_reset_envs,
                dtype=torch.bool,
                device=self.device,
            )
            attempts = 0

            while unresolved.any():
                count = int(unresolved.sum().item())
                samples = (
                    torch.rand(
                        count,
                        3,
                        dtype=torch.float32,
                        device=self.device,
                    )
                    * 2.0
                    - 1.0
                )
                candidates = torch.empty_like(samples)
                candidates[:, 0] = (
                    target[unresolved, 0]
                    + samples[:, 0] * 2.0
                )
                candidates[:, 1] = (
                    target[unresolved, 1]
                    + samples[:, 1] * 2.0
                )
                candidates[:, 2] = (
                    target[unresolved, 2]
                    + samples[:, 2].clamp_min(0.0) * 1.5
                    + 0.5
                )

                unresolved_indices = unresolved.nonzero(
                    as_tuple=False
                ).squeeze(-1)

                if agent_index == 0:
                    valid = torch.ones(
                        count, dtype=torch.bool, device=self.device
                    )
                else:
                    previous = positions[
                        unresolved_indices, :agent_index
                    ]
                    distance = torch.linalg.vector_norm(
                        candidates.unsqueeze(1) - previous,
                        dim=-1,
                    )
                    valid = (
                        distance >= self.collision_distance
                    ).all(dim=-1)

                if valid.any():
                    accepted_indices = unresolved_indices[valid]
                    positions[accepted_indices, agent_index] = (
                        candidates[valid]
                    )
                    unresolved[accepted_indices] = False

                attempts += 1
                if attempts > 10000:
                    raise RuntimeError(
                        "Spawn rejection sampling did not converge."
                    )

        return positions

    def _reset_idx(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return

        # Support both signatures used across OmniDrones revisions.
        try:
            self.drone._reset_idx(env_ids, self.training)
        except TypeError:
            self.drone._reset_idx(env_ids)

        self.drone.throttle[env_ids] = 0.0
        self.drone.throttle_difference[env_ids] = 0.0

        target_sample = (
            torch.rand(
                len(env_ids),
                3,
                dtype=torch.float32,
                device=self.device,
            )
            * 2.0
            - 1.0
        )
        target = torch.empty_like(target_sample)
        target[:, 0] = target_sample[:, 0] * 10.0
        target[:, 1] = target_sample[:, 1] * 10.0
        target[:, 2] = (
            target_sample[:, 2].clamp_min(0.0) * 10.0
            + 1.0
        )
        self.desired_target_pos[env_ids, 0] = target

        positions = self._sample_spawn_positions(target)
        rpy = torch.zeros(
            len(env_ids),
            self.num_agents,
            3,
            dtype=torch.float32,
            device=self.device,
        )
        orientation = euler_to_quaternion(rpy)

        self.drone.set_world_poses(
            positions
            + self.envs_positions[env_ids].unsqueeze(1),
            orientation,
            env_ids,
        )
        self.drone.set_velocities(
            self.initial_velocities[env_ids],
            env_ids,
        )

    def _pre_sim_step(self, tensordict: TensorDictBase):
        policy_action = tensordict[("agents", "action")]
        policy_action = torch.nan_to_num(
            policy_action,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )

        # Exact C++ action transformation.
        physical_rotor_target = (
            (policy_action + 1.0)
            * self.max_rotor_velocity
            * 0.5
        )

        # Invert OmniDrones RotorGroup's square-root action mapping:
        #   target_omega / max_omega = sqrt((action + 1) / 2).
        rotor_fraction = (
            physical_rotor_target / self.drone.MAX_ROT_VEL
        ).clamp(0.0, 1.0)
        omni_action = 2.0 * rotor_fraction.square() - 1.0
        self.drone.apply_action(omni_action)

    @staticmethod
    def _quaternion_to_matrix(
        quaternion: torch.Tensor,
    ) -> torch.Tensor:
        """Convert normalized wxyz quaternions to row-major matrices."""
        quaternion = quaternion / torch.linalg.vector_norm(
            quaternion, dim=-1, keepdim=True
        ).clamp_min(1.0e-8)
        w, x, y, z = quaternion.unbind(dim=-1)

        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        xz = x * z
        yz = y * z
        wx = w * x
        wy = w * y
        wz = w * z

        return torch.stack(
            (
                1.0 - 2.0 * (yy + zz),
                2.0 * (xy - wz),
                2.0 * (xz + wy),
                2.0 * (xy + wz),
                1.0 - 2.0 * (xx + zz),
                2.0 * (yz - wx),
                2.0 * (xz - wy),
                2.0 * (yz + wx),
                1.0 - 2.0 * (xx + yy),
            ),
            dim=-1,
        ).reshape(*quaternion.shape[:-1], 3, 3)

    def _compute_state_and_obs(self):
        # OmniDrones state layout:
        # position[0:3], quaternion wxyz[3:7],
        # linear velocity[7:10], angular velocity[10:13].
        self.drone_state = self.drone.get_state()
        position = self.drone_state[..., :3]
        quaternion = self.drone_state[..., 3:7]
        linear_velocity = self.drone_state[..., 7:10]
        angular_velocity = self.drone_state[..., 10:13]
        self.rotation_matrix = self._quaternion_to_matrix(
            quaternion
        )
        rotation_flat = self.rotation_matrix.flatten(-2)

        self_features = torch.cat(
            (
                self.desired_target_pos - position,
                linear_velocity,
                rotation_flat,
                angular_velocity,
            ),
            dim=-1,
        )

        # Preserve C++ ordering: for each self agent i, concatenate every
        # other agent j in ascending index order, skipping i.
        relative = (
            position.unsqueeze(1) - position.unsqueeze(2)
        )
        # relative[:, i, j] = position[j] - position[i]
        pair_distance = torch.linalg.vector_norm(
            relative, dim=-1, keepdim=True
        )
        self.pairwise_distance = pair_distance

        other_velocity = linear_velocity.unsqueeze(1).expand(
            -1, self.num_agents, -1, -1
        )
        other_rotation = rotation_flat.unsqueeze(1).expand(
            -1, self.num_agents, -1, -1
        )
        other_features_all = torch.cat(
            (
                relative,
                pair_distance,
                other_velocity,
                other_rotation,
            ),
            dim=-1,
        )

        off_diagonal = ~torch.eye(
            self.num_agents,
            dtype=torch.bool,
            device=self.device,
        )
        other_features = other_features_all[
            :, off_diagonal
        ].reshape(
            self.num_envs,
            self.num_agents,
            (self.num_agents - 1) * 16,
        )

        observation = torch.cat(
            (self_features, other_features),
            dim=-1,
        )
        if observation.shape[-1] != self.observation_dim:
            raise RuntimeError(
                f"Expected observation dimension "
                f"{self.observation_dim}, got "
                f"{observation.shape[-1]}."
            )

        return TensorDict(
            {"agents": {"observation": observation}},
            self.batch_size,
            device=self.device,
        )

    def _compute_reward_and_done(self):
        position = self.drone_state[..., :3]

        mean_position = position.mean(dim=1, keepdim=True)
        centered_position = position - mean_position

        desired = self.target_formation.unsqueeze(0).expand(
            self.num_envs, -1, -1
        )
        formation_distance = torch.cdist(
            centered_position, desired
        )
        directed_current_to_desired = (
            formation_distance.min(dim=-1).values.max(
                dim=-1
            ).values
        )
        directed_desired_to_current = (
            formation_distance.min(dim=-2).values.max(
                dim=-1
            ).values
        )
        hausdorff_cost = torch.maximum(
            directed_current_to_desired,
            directed_desired_to_current,
        )

        distance_to_target = torch.linalg.vector_norm(
            mean_position - self.desired_target_pos,
            dim=-1,
        ).squeeze(-1)

        reward_formation = 1.0 / (
            1.0 + (hausdorff_cost * 1.6).square()
        )
        reward_position = torch.exp(-distance_to_target)

        pairwise = torch.cdist(position, position)
        diagonal_mask = torch.eye(
            self.num_agents,
            dtype=torch.bool,
            device=self.device,
        ).unsqueeze(0)
        pairwise = pairwise.masked_fill(
            diagonal_mask, float("inf")
        )
        self.min_separation = pairwise.amin(dim=(-2, -1))

        reward_separation = (
            self.min_separation / self.safe_distance
        ).square().clamp(0.0, 1.0)

        heading = self.rotation_matrix[..., 0, 0]
        reward_heading = heading.mean(dim=-1)

        combined_reward = reward_separation * (
            reward_formation
            + reward_formation
            * (reward_position + reward_heading)
            + 0.4 * reward_position
        )

        r22 = self.rotation_matrix[..., 2, 2].clamp(
            -1.0, 1.0
        )
        excessive_tilt = (
            torch.acos(r22) > self.max_tilt
        ).any(dim=-1)
        collision = (
            self.min_separation < self.collision_distance
        )
        timeout = self.progress_buf >= self.max_episode_length

        terminated = (
            collision | excessive_tilt | timeout
        ).unsqueeze(-1)
        truncated = torch.zeros_like(terminated)

        shared_reward = combined_reward.view(
            self.num_envs, 1, 1
        ).expand(-1, self.num_agents, 1)

        return TensorDict(
            {
                "agents": {"reward": shared_reward},
                "done": terminated,
                "terminated": terminated,
                "truncated": truncated,
            },
            self.batch_size,
            device=self.device,
        )
