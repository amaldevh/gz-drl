#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Amal Dev Haridevan

"""Run the instrumented ROS 2 + Gazebo synchronization experiment."""

from __future__ import annotations

import argparse
import math
import multiprocessing
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from common import (
    CONTROL_RATE_HZ,
    DEFAULT_RESULTS_DIR,
    DEFAULT_SDF,
    GRAVITY_M_S2,
    LINK_NAME,
    MODEL_NAME,
    PHYSICS_DT,
    UAV_MASS_KG,
    action_fields,
    configure_runtime_paths,
    generate_action_sequence,
    save_action_sequence,
    state_fields,
)


OBSERVATION_SIZE = 2 + 13
COMMAND_SIZE = 4 + 6
ACK_SIZE = 9


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure transition synchronization through ROS 2 topics"
    )
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--actions", type=int, default=200)
    parser.add_argument("--control-rate", type=float, default=CONTROL_RATE_HZ)
    parser.add_argument("--rtf", type=float, default=1.0)
    parser.add_argument("--startup-timeout", type=float, default=30.0)
    parser.add_argument("--ack-timeout", type=float, default=10.0)
    parser.add_argument("--sdf-file", type=Path, default=DEFAULT_SDF)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--link-name", default=LINK_NAME)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "raw" / "ros2_gazebo",
    )
    parser.add_argument(
        "--trial-index",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.trials <= 0 or args.actions <= 0:
        parser.error("--trials and --actions must be positive")
    for name in ("control_rate", "rtf", "startup_timeout", "ack_timeout"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if args.trial_index is not None and args.trial_index <= 0:
        parser.error("--trial-index must be positive")
    args.sdf_file = args.sdf_file.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if not args.sdf_file.is_file():
        parser.error(f"SDF file does not exist: {args.sdf_file}")


def bridge_process(
    partition: str,
    sdf_file: str,
    model_name: str,
    link_name: str,
    rtf: float,
) -> None:
    """Own Gazebo and expose precisely instrumented ROS 2 topics."""
    configure_runtime_paths()
    import gzdrl
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Bool, Float64MultiArray

    class InstrumentedBridge(Node):
        def __init__(self) -> None:
            super().__init__(f"synchronization_bridge_{partition}")
            self.server = gzdrl.DRLServer(
                partition, sdf_file, [model_name], False
            )
            self.physics_dt = float(self.server.step_size())
            if not math.isclose(self.physics_dt, PHYSICS_DT, abs_tol=1.0e-7):
                raise RuntimeError(
                    f"Expected {PHYSICS_DT:g} s physics dt, got "
                    f"{self.physics_dt:g}"
                )
            self.current_wrench = np.array(
                [
                    0.0,
                    0.0,
                    UAV_MASS_KG * GRAVITY_M_S2,
                    0.0,
                    0.0,
                    0.0,
                ],
                dtype=np.float64,
            )
            self.pending_commands: list[tuple[float, ...]] = []
            self.stopped = False
            topic_root = f"/synchronization_validation/{partition}"
            self.observation_publisher = self.create_publisher(
                Float64MultiArray, f"{topic_root}/observation", 1
            )
            self.ack_publisher = self.create_publisher(
                Float64MultiArray, f"{topic_root}/action_ack", 100
            )
            self.create_subscription(
                Float64MultiArray,
                f"{topic_root}/action",
                self.on_action,
                100,
            )
            self.create_subscription(
                Bool, f"{topic_root}/stop", self.on_stop, 1
            )

            self.server.respawn_model(
                model_name,
                np.array([0.0, 0.0, 1.0], dtype=np.float64),
                np.zeros(3, dtype=np.float64),
            )
            for _ in range(10):
                self.apply_current_wrench()
                self.server.run_once()
            self.timer = self.create_timer(
                self.physics_dt / rtf, self.physics_step
            )

        def apply_current_wrench(self) -> None:
            self.server.set_wrench(
                model_name,
                link_name,
                self.current_wrench[:3].copy(),
                self.current_wrench[3:].copy(),
            )

        def on_action(self, message) -> None:
            values = tuple(float(value) for value in message.data)
            if len(values) != COMMAND_SIZE:
                self.get_logger().error(
                    f"Ignoring action with {len(values)} rather than "
                    f"{COMMAND_SIZE} values"
                )
                return
            command_received_wall = time.perf_counter()
            command_received_iteration = float(self.server.sim_iterations())
            self.current_wrench[:] = values[4:10]
            self.pending_commands.append(
                (*values[:4], command_received_iteration, command_received_wall)
            )

        def on_stop(self, message) -> None:
            self.stopped = bool(message.data)

        def publish_observation(self, iteration: int, wall_time: float) -> None:
            self.server.update_control_states()
            state = np.asarray(
                self.server.control_states[model_name][link_name][0],
                dtype=np.float64,
            ).reshape(-1)
            message = Float64MultiArray()
            message.data = [float(iteration), wall_time, *state.tolist()]
            self.observation_publisher.publish(message)

        def physics_step(self) -> None:
            pending = self.pending_commands
            self.pending_commands = []
            self.apply_current_wrench()
            self.server.run_once()
            physics_applied_wall = time.perf_counter()
            applied_iteration = int(self.server.sim_iterations())
            self.publish_observation(applied_iteration, physics_applied_wall)

            for index, command in enumerate(pending):
                (
                    sequence,
                    observation_iteration,
                    observation_received_wall,
                    action_sent_wall,
                    command_received_iteration,
                    command_received_wall,
                ) = command
                # If multiple callbacks occur before a physics update, only the
                # final wrench reaches physics; earlier commands were superseded.
                action_applied = float(index == len(pending) - 1)
                ack = Float64MultiArray()
                ack.data = [
                    sequence,
                    action_applied,
                    observation_iteration,
                    command_received_iteration,
                    float(applied_iteration),
                    observation_received_wall,
                    action_sent_wall,
                    command_received_wall,
                    physics_applied_wall,
                ]
                self.ack_publisher.publish(ack)

    rclpy.init()
    node = InstrumentedBridge()
    try:
        while rclpy.ok() and not node.stopped:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def run_controller(
    args: argparse.Namespace,
    trial: int,
    partition: str,
    actions: np.ndarray,
) -> pd.DataFrame:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Bool, Float64MultiArray

    class InstrumentedController(Node):
        def __init__(self) -> None:
            super().__init__(f"synchronization_controller_{partition}")
            topic_root = f"/synchronization_validation/{partition}"
            self.action_publisher = self.create_publisher(
                Float64MultiArray, f"{topic_root}/action", 100
            )
            self.stop_publisher = self.create_publisher(
                Bool, f"{topic_root}/stop", 1
            )
            self.create_subscription(
                Float64MultiArray,
                f"{topic_root}/observation",
                self.on_observation,
                1,
            )
            self.create_subscription(
                Float64MultiArray,
                f"{topic_root}/action_ack",
                self.on_ack,
                100,
            )
            self.latest_observation: tuple[int, float, np.ndarray] | None = None
            self.acknowledgements: dict[int, tuple[np.ndarray, float]] = {}

        def on_observation(self, message) -> None:
            received_wall = time.perf_counter()
            values = np.asarray(message.data, dtype=np.float64)
            if values.shape != (OBSERVATION_SIZE,):
                self.get_logger().error(
                    f"Ignoring malformed observation with shape {values.shape}"
                )
                return
            self.latest_observation = (
                int(values[0]),
                received_wall,
                values[2:].copy(),
            )

        def on_ack(self, message) -> None:
            received_wall = time.perf_counter()
            values = np.asarray(message.data, dtype=np.float64)
            if values.shape != (ACK_SIZE,):
                self.get_logger().error(
                    f"Ignoring malformed acknowledgement with shape {values.shape}"
                )
                return
            self.acknowledgements[int(values[0])] = (values, received_wall)

    rclpy.init()
    node = InstrumentedController()
    base_records: dict[int, dict[str, object]] = {}
    try:
        startup_deadline = time.perf_counter() + args.startup_timeout
        while node.latest_observation is None:
            remaining = startup_deadline - time.perf_counter()
            if remaining <= 0.0:
                raise TimeoutError("Timed out waiting for the first ROS observation")
            rclpy.spin_once(node, timeout_sec=min(0.1, remaining))

        next_deadline = time.perf_counter() + 0.05
        for action_index, wrench in enumerate(actions):
            while True:
                remaining = next_deadline - time.perf_counter()
                if remaining <= 0.0:
                    break
                rclpy.spin_once(node, timeout_sec=min(remaining, 0.002))
            # Drain callbacks already ready at the decision boundary.
            for _ in range(10):
                rclpy.spin_once(node, timeout_sec=0.0)
            observation = node.latest_observation
            if observation is None:
                raise RuntimeError("No ROS observation is available")
            observation_iteration, observation_received_wall, state = observation
            action_sent_wall = time.perf_counter()
            message = Float64MultiArray()
            message.data = [
                float(action_index),
                float(observation_iteration),
                observation_received_wall,
                action_sent_wall,
                *wrench.tolist(),
            ]
            base_record: dict[str, object] = {
                "backend": "ros2_gazebo",
                "trial": trial,
                "action_index": action_index,
            }
            base_record.update(action_fields(wrench))
            base_record.update(state_fields(state))
            base_records[action_index] = base_record
            node.action_publisher.publish(message)
            next_deadline += 1.0 / args.control_rate

        acknowledgement_deadline = time.perf_counter() + args.ack_timeout
        while len(node.acknowledgements) < len(actions):
            remaining = acknowledgement_deadline - time.perf_counter()
            if remaining <= 0.0:
                missing = sorted(
                    set(range(len(actions))) - set(node.acknowledgements)
                )
                raise TimeoutError(
                    f"Timed out waiting for ROS acknowledgements: {missing[:10]}"
                )
            rclpy.spin_once(node, timeout_sec=min(0.1, remaining))

        records: list[dict[str, object]] = []
        for action_index in range(len(actions)):
            values, ack_received_wall = node.acknowledgements[action_index]
            (
                _,
                action_applied,
                observation_iteration,
                command_received_iteration,
                applied_iteration,
                observation_received_wall,
                action_sent_wall,
                command_received_wall,
                physics_applied_wall,
            ) = values
            applied = bool(round(action_applied))
            record = base_records[action_index]
            record.update(
                {
                    "action_applied": int(applied),
                    "observation_iteration": int(observation_iteration),
                    "command_received_iteration": int(
                        command_received_iteration
                    ),
                    "applied_iteration": int(applied_iteration),
                    "action_to_physics_latency_ms": (
                        1.0e3 * (physics_applied_wall - action_sent_wall)
                        if applied
                        else math.nan
                    ),
                    "command_transport_latency_ms": 1.0e3
                    * (command_received_wall - action_sent_wall),
                    "observation_age_wall_ms": (
                        1.0e3
                        * (physics_applied_wall - observation_received_wall)
                        if applied
                        else math.nan
                    ),
                    "observation_age_sim_ms": (
                        1.0e3
                        * (applied_iteration - observation_iteration)
                        * PHYSICS_DT
                        if applied
                        else math.nan
                    ),
                    "physics_updates_between_observation_and_action": (
                        max(
                            0,
                            int(applied_iteration)
                            - int(observation_iteration)
                            - 1,
                        )
                        if applied
                        else math.nan
                    ),
                    "ack_round_trip_latency_ms": 1.0e3
                    * (ack_received_wall - action_sent_wall),
                    "observation_received_wall_s": observation_received_wall,
                    "action_sent_wall_s": action_sent_wall,
                    "command_received_wall_s": command_received_wall,
                    "physics_applied_wall_s": physics_applied_wall,
                }
            )
            records.append(record)

        stop = Bool()
        stop.data = True
        node.stop_publisher.publish(stop)
        for _ in range(10):
            rclpy.spin_once(node, timeout_sec=0.01)
        return pd.DataFrame.from_records(records)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def run_single_trial(args: argparse.Namespace) -> int:
    try:
        import rclpy  # noqa: F401
        import std_msgs  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            "The ROS 2 benchmark requires rclpy and std_msgs in the active "
            "ROS environment. Source the ROS setup file before running."
        ) from error

    configure_runtime_paths()
    actions = generate_action_sequence(args.actions, args.control_rate)
    trial = int(args.trial_index)
    partition = f"sync_ros2_{trial}_{time.time_ns()}"
    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=bridge_process,
        args=(
            partition,
            str(args.sdf_file),
            args.model_name,
            args.link_name,
            args.rtf,
        ),
    )
    process.start()
    forced_termination = False
    try:
        frame = run_controller(args, trial, partition, actions)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = args.output_dir / f"run_{trial:03d}.csv"
        frame.to_csv(output_path, index=False)
        print(f"Saved {output_path}")
    finally:
        process.join(timeout=5.0)
        if process.is_alive():
            forced_termination = True
            process.terminate()
            process.join(timeout=5.0)
        if not forced_termination and process.exitcode not in (0, None):
            raise RuntimeError(
                f"ROS 2 bridge process exited with status {process.exitcode}"
            )
    return 0


def run_all_trials(args: argparse.Namespace) -> int:
    actions = generate_action_sequence(args.actions, args.control_rate)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_action_sequence(
        args.output_dir.parent.parent / "action_sequence.csv",
        actions,
        args.control_rate,
    )
    for trial in range(1, args.trials + 1):
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--trials",
            str(args.trials),
            "--actions",
            str(args.actions),
            "--control-rate",
            str(args.control_rate),
            "--rtf",
            str(args.rtf),
            "--startup-timeout",
            str(args.startup_timeout),
            "--ack-timeout",
            str(args.ack_timeout),
            "--sdf-file",
            str(args.sdf_file),
            "--model-name",
            args.model_name,
            "--link-name",
            args.link_name,
            "--output-dir",
            str(args.output_dir),
            "--trial-index",
            str(trial),
        ]
        subprocess.run(command, check=True)
        print(f"Completed ROS 2 trial {trial}/{args.trials}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(args, parser)
    if args.trial_index is not None:
        return run_single_trial(args)
    return run_all_trials(args)


if __name__ == "__main__":
    raise SystemExit(main())
