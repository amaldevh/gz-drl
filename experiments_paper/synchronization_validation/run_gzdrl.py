#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Amal Dev Haridevan

"""Run the fixed-action synchronization experiment through raw DRLServer."""

from __future__ import annotations

import argparse
import math
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
    PHYSICS_STEPS_PER_ACTION,
    UAV_MASS_KG,
    action_fields,
    configure_runtime_paths,
    generate_action_sequence,
    save_action_sequence,
    state_fields,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure transition synchronization using raw DRLServer"
    )
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--actions", type=int, default=200)
    parser.add_argument("--control-rate", type=float, default=CONTROL_RATE_HZ)
    parser.add_argument("--physics-steps", type=int, default=PHYSICS_STEPS_PER_ACTION)
    parser.add_argument("--sdf-file", type=Path, default=DEFAULT_SDF)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--link-name", default=LINK_NAME)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "raw" / "gzdrl",
    )
    parser.add_argument(
        "--realtime",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pace action dispatch to wall-clock control rate",
    )
    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.trials <= 0 or args.actions <= 0 or args.physics_steps <= 0:
        parser.error("--trials, --actions, and --physics-steps must be positive")
    if not math.isfinite(args.control_rate) or args.control_rate <= 0.0:
        parser.error("--control-rate must be finite and positive")
    if not math.isclose(
        args.physics_steps * PHYSICS_DT,
        1.0 / args.control_rate,
        abs_tol=1.0e-9,
    ):
        parser.error(
            "--control-rate must equal 1 / (--physics-steps * 0.001 s)"
        )
    args.sdf_file = args.sdf_file.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if not args.sdf_file.is_file():
        parser.error(f"SDF file does not exist: {args.sdf_file}")


def read_state(server, model_name: str, link_name: str) -> np.ndarray:
    server.update_control_states()
    state = np.asarray(
        server.control_states[model_name][link_name][0], dtype=np.float64
    ).reshape(-1)
    if state.shape != (13,) or not np.isfinite(state).all():
        raise RuntimeError(f"Invalid DRLServer state: {state.shape}")
    return state.copy()


def apply_wrench(server, model: str, link: str, wrench: np.ndarray) -> None:
    server.set_wrench(model, link, wrench[:3].copy(), wrench[3:].copy())


def run_trial(
    args: argparse.Namespace,
    trial: int,
    actions: np.ndarray,
) -> pd.DataFrame:
    configure_runtime_paths()
    import gzdrl

    partition = f"sync_gzdrl_{trial}_{time.time_ns()}"
    server = gzdrl.DRLServer(
        partition,
        str(args.sdf_file),
        [args.model_name],
        False,
    )
    physics_dt = float(server.step_size())
    if not math.isclose(physics_dt, PHYSICS_DT, abs_tol=1.0e-7):
        raise RuntimeError(f"Expected {PHYSICS_DT:g} s physics dt, got {physics_dt:g}")

    hover_wrench = np.array(
        [0.0, 0.0, UAV_MASS_KG * GRAVITY_M_S2, 0.0, 0.0, 0.0],
        dtype=np.float64,
    )
    server.respawn_model(
        args.model_name,
        np.array([0.0, 0.0, 1.0], dtype=np.float64),
        np.zeros(3, dtype=np.float64),
    )
    for _ in range(10):
        apply_wrench(server, args.model_name, args.link_name, hover_wrench)
        server.run_once()

    records: list[dict[str, object]] = []
    next_deadline = time.perf_counter()
    for action_index, wrench in enumerate(actions):
        observation = read_state(server, args.model_name, args.link_name)
        observation_iteration = int(server.sim_iterations())
        observation_received_wall = time.perf_counter()
        action_sent_wall = time.perf_counter()
        apply_wrench(server, args.model_name, args.link_name, wrench)
        command_received_wall = time.perf_counter()
        command_received_iteration = int(server.sim_iterations())
        server.run_once()
        physics_applied_wall = time.perf_counter()
        applied_iteration = int(server.sim_iterations())
        for _ in range(args.physics_steps - 1):
            apply_wrench(server, args.model_name, args.link_name, wrench)
            server.run_once()

        record: dict[str, object] = {
            "backend": "gzdrl",
            "trial": trial,
            "action_index": action_index,
            "action_applied": 1,
            "observation_iteration": observation_iteration,
            "command_received_iteration": command_received_iteration,
            "applied_iteration": applied_iteration,
            "action_to_physics_latency_ms": 1.0e3
            * (physics_applied_wall - action_sent_wall),
            "command_transport_latency_ms": 1.0e3
            * (command_received_wall - action_sent_wall),
            "observation_age_wall_ms": 1.0e3
            * (physics_applied_wall - observation_received_wall),
            "observation_age_sim_ms": 1.0e3
            * (applied_iteration - observation_iteration)
            * physics_dt,
            "physics_updates_between_observation_and_action": max(
                0, applied_iteration - observation_iteration - 1
            ),
            "ack_round_trip_latency_ms": 1.0e3
            * (physics_applied_wall - action_sent_wall),
            "observation_received_wall_s": observation_received_wall,
            "action_sent_wall_s": action_sent_wall,
            "command_received_wall_s": command_received_wall,
            "physics_applied_wall_s": physics_applied_wall,
        }
        record.update(action_fields(wrench))
        record.update(state_fields(observation))
        records.append(record)

        if args.realtime:
            next_deadline += 1.0 / args.control_rate
            remaining = next_deadline - time.perf_counter()
            if remaining > 0.0:
                time.sleep(remaining)
    return pd.DataFrame.from_records(records)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(args, parser)
    actions = generate_action_sequence(args.actions, args.control_rate)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_action_sequence(
        args.output_dir.parent.parent / "action_sequence.csv",
        actions,
        args.control_rate,
    )
    for trial in range(1, args.trials + 1):
        output_path = args.output_dir / f"run_{trial:03d}.csv"
        frame = run_trial(args, trial, actions)
        frame.to_csv(output_path, index=False)
        print(f"[{trial}/{args.trials}] saved {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
