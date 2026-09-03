# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Amal Dev Haridevan
#
# This environment is a clean-room task adaptation based on the behavior of the
# user-provided Gazebo HoverEnv and the public OmniDrones environment API.

from __future__ import annotations

import torch

from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import (
    CompositeSpec,
    DiscreteTensorSpec,
    UnboundedContinuousTensorSpec,
)

from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.robots.drone import MultirotorBase
from omni_drones.utils.torch import euler_to_quaternion


class GazeboPoolHover(IsaacEnv):
    """Vectorized OmniDrones equivalent of the supplied GazeboPool HoverEnv.

    Matched task behavior
    ---------------------
    * Four normalized actions in [-1, 1].
    * Actions represent target rotor velocities:
        omega_target = (action + 1) * 2300 / 2.
    * 22-dimensional observation:
        relative position (3),
        yaw error (1),
        linear velocity (3),
        quaternion wxyz (4),
        angular velocity (3),
        latest physical rotor command history (4),
        previous physical rotor command history (4).
    * Spawn and target positions:
        x/y in [-5, 5], z in [0, 3].
    * Spawn and target yaw in [-0.72, 0.72].
    * Reward and termination logic match the supplied Python environment.
    * One OmniDrones physics step is performed per environment step.

    Notes
    -----
    OmniDrones internally accepts thrust-normalized motor actions rather than
    target rotor velocity directly. `_pre_sim_step` therefore converts the
    Gazebo-style target rotor velocity to the action required by RotorGroup.

    The supplied Gazebo observation intentionally normalizes action history as
    `physical_rotor_command / 1300 - 1`, even though the command ceiling is
    2300. That behavior is preserved exactly.
    """

    OBSERVATION_DIM = 22
    ACTION_DIM = 4

    def __init__(self, cfg, headless: bool):
        self.command_max_rotor_velocity = float(
            cfg.task.get("command_max_rotor_velocity", 2300.0)
        )
        self.action_history_divisor = float(
            cfg.task.get("action_history_divisor", 1300.0)
        )
        self.max_yaw = float(cfg.task.get("max_yaw", 1.57))
        self.max_tilt = float(cfg.task.get("max_tilt", 1.57))
        self.instant_rotor_response = bool(
            cfg.task.get("instant_rotor_response", True)
        )

        maximum_bounds = cfg.task.get("maximum_bounds", [7.0, 7.0, 7.0])
        spawn_low = cfg.task.get("spawn_low", [-5.0, -5.0, 0.0])
        spawn_high = cfg.task.get("spawn_high", [5.0, 5.0, 3.0])
        self.yaw_limit = float(cfg.task.get("spawn_yaw_limit", 0.72))

        super().__init__(cfg, headless)

        if self.drone.num_rotors != self.ACTION_DIM:
            raise ValueError(
                f"GazeboPoolHover requires exactly four rotors, but "
                f"{self.cfg.task.drone_model.name} has {self.drone.num_rotors}."
            )
        if self.controller is not None:
            raise ValueError(
                "GazeboPoolHover requires drone_model.controller=null because "
                "the supplied Gazebo environment commands rotor velocity directly."
            )

        self.drone.initialize()

        # The Gazebo API sets a rotor-velocity target directly. Setting tau=1
        # makes the native RotorGroup reach that target in the current step.
        # This can be disabled in YAML to benchmark OmniDrones' native motor lag.
        if self.instant_rotor_response:
            self.drone.tau_up.fill_(1.0)
            self.drone.tau_down.fill_(1.0)

        self.maximum_bounds = torch.as_tensor(
            maximum_bounds, dtype=torch.float32, device=self.device
        )
        self.spawn_low = torch.as_tensor(
            spawn_low, dtype=torch.float32, device=self.device
        )
        self.spawn_high = torch.as_tensor(
            spawn_high, dtype=torch.float32, device=self.device
        )
        self.spawn_mean = (self.spawn_low + self.spawn_high) * 0.5
        self.spawn_half_range = (self.spawn_high - self.spawn_low) * 0.5

        self.initial_velocities = torch.zeros_like(self.drone.get_velocities())

        self.desired_pos = torch.zeros(
            self.num_envs, 1, 3, dtype=torch.float32, device=self.device
        )
        self.desired_yaw = torch.zeros(
            self.num_envs, 1, 1, dtype=torch.float32, device=self.device
        )
        self.action_buffer1 = torch.zeros(
            self.num_envs,
            1,
            self.ACTION_DIM,
            dtype=torch.float32,
            device=self.device,
        )
        self.action_buffer2 = torch.zeros_like(self.action_buffer1)

        # Populated on every observation computation.
        self.drone_state = None
        self.relative_pos = None
        self.yaw_error = None
        self.current_yaw = None
        self.angular_velocity = None

    def _design_scene(self):
        import omni_drones.utils.kit as kit_utils

        drone_cfg = self.cfg.task.drone_model
        self.drone, self.controller = MultirotorBase.make(
            drone_cfg.name, drone_cfg.controller
        )

        # The visual spawn value is replaced during reset.
        self.drone.spawn(translations=[(0.0, 0.0, 1.5)])

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
                            (1, self.OBSERVATION_DIM), device=self.device
                        )
                    }
                )
            }
        ).expand(self.num_envs).to(self.device)

        self.action_spec = CompositeSpec(
            {
                "agents": CompositeSpec(
                    {
                        "action": self.drone.action_spec.unsqueeze(0),
                    }
                )
            }
        ).expand(self.num_envs).to(self.device)

        self.reward_spec = CompositeSpec(
            {
                "agents": CompositeSpec(
                    {
                        "reward": UnboundedContinuousTensorSpec(
                            (1, 1), device=self.device
                        )
                    }
                )
            }
        ).expand(self.num_envs).to(self.device)

        self.agent_spec["drone"] = AgentSpec(
            "drone",
            1,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
        )

    def _reset_idx(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return

        self.drone._reset_idx(env_ids, self.training)

        # The supplied Gazebo reset starts with zero rotor command/state rather
        # than OmniDrones' default hover throttle.
        self.drone.throttle[env_ids] = 0.0
        self.drone.throttle_difference[env_ids] = 0.0

        spawn_sample = (
            torch.rand(
                (*env_ids.shape, 1, 4),
                dtype=torch.float32,
                device=self.device,
            )
            * 2.0
            - 1.0
        )
        position = (
            self.spawn_mean
            + spawn_sample[..., :3] * self.spawn_half_range
        )
        yaw = spawn_sample[..., 3:4] * self.yaw_limit

        rpy = torch.zeros(
            (*env_ids.shape, 1, 3),
            dtype=torch.float32,
            device=self.device,
        )
        rpy[..., 2:3] = yaw
        orientation = euler_to_quaternion(rpy)

        self.drone.set_world_poses(
            position + self.envs_positions[env_ids].unsqueeze(1),
            orientation,
            env_ids,
        )
        self.drone.set_velocities(self.initial_velocities[env_ids], env_ids)

        target_sample = (
            torch.rand(
                (*env_ids.shape, 1, 4),
                dtype=torch.float32,
                device=self.device,
            )
            * 2.0
            - 1.0
        )
        self.desired_pos[env_ids] = (
            self.spawn_mean
            + target_sample[..., :3] * self.spawn_half_range
        )
        self.desired_yaw[env_ids] = (
            target_sample[..., 3:4] * self.yaw_limit
        )

        self.action_buffer1[env_ids] = 0.0
        self.action_buffer2[env_ids] = 0.0

    def _pre_sim_step(self, tensordict: TensorDictBase):
        normalized_action = tensordict[("agents", "action")]
        normalized_action = torch.nan_to_num(
            normalized_action,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )

        # Exact action mapping from the supplied Gazebo environment.
        physical_rotor_target = (
            (normalized_action + 1.0)
            * self.command_max_rotor_velocity
            * 0.5
        )

        # Preserve the exact newest/previous action-buffer update ordering.
        self.action_buffer2.copy_(self.action_buffer1)
        self.action_buffer1.copy_(physical_rotor_target)

        # RotorGroup uses:
        #   target_throttle = sqrt((isaac_action + 1) / 2)
        # and rotor velocity is target_throttle * MAX_ROT_VEL.
        # Invert that mapping so that the target physical rotor velocity matches
        # the supplied Gazebo command.
        rotor_fraction = (
            physical_rotor_target / self.drone.MAX_ROT_VEL
        ).clamp(0.0, 1.0)
        isaac_action = 2.0 * rotor_fraction.square() - 1.0

        self.drone.apply_action(isaac_action)

    @staticmethod
    def _yaw_from_wxyz(quaternion: torch.Tensor) -> torch.Tensor:
        """Return ZYX yaw from a wxyz quaternion."""
        w, x, y, z = quaternion.unbind(dim=-1)
        sin_yaw = 2.0 * (w * z + x * y)
        cos_yaw = 1.0 - 2.0 * (y.square() + z.square())
        return torch.atan2(sin_yaw, cos_yaw).unsqueeze(-1)

    @staticmethod
    def _body_z_world_z(quaternion: torch.Tensor) -> torch.Tensor:
        """Return R[2, 2] for a wxyz quaternion."""
        _, x, y, _ = quaternion.unbind(dim=-1)
        return 1.0 - 2.0 * (x.square() + y.square())

    def _compute_state_and_obs(self):
        # OmniDrones state layout for a multirotor:
        # position[0:3], quaternion_wxyz[3:7], world velocity[7:13],
        # heading[13:16], up[16:19], normalized throttle[19:].
        self.drone_state = self.drone.get_state()
        current_pos = self.drone_state[..., :3]
        quaternion = self.drone_state[..., 3:7]
        linear_velocity = self.drone_state[..., 7:10]
        self.angular_velocity = self.drone_state[..., 10:13]

        self.relative_pos = self.desired_pos - current_pos
        self.current_yaw = self._yaw_from_wxyz(quaternion)
        self.yaw_error = self.desired_yaw - self.current_yaw

        latest_action_history = (
            self.action_buffer1 / self.action_history_divisor - 1.0
        )
        previous_action_history = (
            self.action_buffer2 / self.action_history_divisor - 1.0
        )

        observation = torch.cat(
            [
                self.relative_pos,
                self.yaw_error,
                linear_velocity,
                quaternion,
                self.angular_velocity,
                latest_action_history,
                previous_action_history,
            ],
            dim=-1,
        )

        if observation.shape[-1] != self.OBSERVATION_DIM:
            raise RuntimeError(
                f"Expected {self.OBSERVATION_DIM} observations, got "
                f"{observation.shape[-1]}."
            )

        return TensorDict(
            {"agents": {"observation": observation}},
            self.batch_size,
            device=self.device,
        )

    def _compute_reward_and_done(self):
        current_pos = self.drone_state[..., :3]
        quaternion = self.drone_state[..., 3:7]

        state_error = 1.6 * 1.6 * (
            self.relative_pos.square().sum(dim=-1)
            + self.yaw_error.squeeze(-1).square()
        )
        reward_pos = 1.0 / (1.0 + state_error)
        reward_omega = 0.01 / (
            1.0 + self.angular_velocity.square().sum(dim=-1)
        )
        reward = reward_pos + reward_omega

        yaw_out_of_bounds = (
            self.current_yaw.abs().squeeze(-1) > self.max_yaw
        )
        position_out_of_bounds = (
            current_pos.abs() > self.maximum_bounds
        ).any(dim=-1)

        r_22 = self._body_z_world_z(quaternion).clamp(-1.0, 1.0)
        tilt_angle = torch.acos(r_22).abs()
        tilt_out_of_bounds = tilt_angle > self.max_tilt

        failure = (
            yaw_out_of_bounds
            | position_out_of_bounds
            | tilt_out_of_bounds
        )

        # The supplied Gym environment evaluates current_step >= max_steps
        # before incrementing it. With current_step initially zero, timeout
        # therefore occurs on transition max_steps + 1. Use ">" to preserve
        # that exact off-by-one behavior.
        timeout = (
            self.progress_buf > self.max_episode_length
        ).unsqueeze(-1)

        # Preserve the source Gym API semantics exactly: its fourth return
        # value is always False, so timeout is included in `terminated` rather
        # than reported as `truncated`.
        terminated = failure | timeout
        truncated = torch.zeros_like(terminated)
        done = terminated

        return TensorDict(
            {
                "agents": {"reward": reward.unsqueeze(-1)},
                "done": done,
                "terminated": terminated,
                "truncated": truncated,
            },
            self.batch_size,
            device=self.device,
        )
