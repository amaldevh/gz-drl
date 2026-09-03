# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Amal Dev Haridevan

"""Shared definitions for the transition-synchronization experiment."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import gzdrl


EXPERIMENT_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = EXPERIMENT_ROOT / "results"
DEFAULT_SDF = gzdrl.get_sdf_path("world_hover.sdf")
MODEL_NAME = "quadrotor"
LINK_NAME = "quadrotor/base_link"
PHYSICS_DT = 1.0e-3
CONTROL_RATE_HZ = 100.0
PHYSICS_STEPS_PER_ACTION = 10
UAV_MASS_KG = 1.52
GRAVITY_M_S2 = 9.82

STATE_COLUMNS = (
    "state_x",
    "state_y",
    "state_z",
    "state_vx",
    "state_vy",
    "state_vz",
    "state_qw",
    "state_qx",
    "state_qy",
    "state_qz",
    "state_wx",
    "state_wy",
    "state_wz",
)
ACTION_COLUMNS = (
    "force_x",
    "force_y",
    "force_z",
    "moment_x",
    "moment_y",
    "moment_z",
)

REQUIRED_RESULT_COLUMNS = (
    "backend",
    "trial",
    "action_index",
    "action_applied",
    "observation_iteration",
    "command_received_iteration",
    "applied_iteration",
    "action_to_physics_latency_ms",
    "command_transport_latency_ms",
    "observation_age_wall_ms",
    "observation_age_sim_ms",
    "physics_updates_between_observation_and_action",
    "ack_round_trip_latency_ms",
    *ACTION_COLUMNS,
    *STATE_COLUMNS,
)


def prepend_environment_path(name: str, path: Path) -> None:
    value = str(path.resolve())
    entries = [entry for entry in os.environ.get(name, "").split(":") if entry]
    if value not in entries:
        os.environ[name] = ":".join([value, *entries])


def configure_runtime_paths() -> None:
    """Ensure Gazebo can locate the installed GzDRL resources and plugins."""
    prepend_environment_path("GZ_SIM_RESOURCE_PATH", gzdrl.get_sdf_path())
    prepend_environment_path("GZ_SIM_SYSTEM_PLUGIN_PATH", gzdrl.get_plugin_path())


def generate_action_sequence(
    action_count: int,
    control_rate_hz: float = CONTROL_RATE_HZ,
) -> np.ndarray:
    """Generate the deterministic world-frame wrench used by every backend."""
    if action_count <= 0:
        raise ValueError("action_count must be positive")
    if not np.isfinite(control_rate_hz) or control_rate_hz <= 0.0:
        raise ValueError("control_rate_hz must be finite and positive")

    time_s = np.arange(action_count, dtype=np.float64) / control_rate_hz
    actions = np.empty((action_count, 6), dtype=np.float64)
    actions[:, 0] = 0.80 * np.sin(2.0 * np.pi * 0.37 * time_s)
    actions[:, 1] = 0.65 * np.sin(2.0 * np.pi * 0.53 * time_s + 0.4)
    actions[:, 2] = (
        UAV_MASS_KG * GRAVITY_M_S2
        + 0.70 * np.sin(2.0 * np.pi * 0.29 * time_s + 0.2)
    )
    actions[:, 3] = 0.012 * np.sin(2.0 * np.pi * 0.41 * time_s)
    actions[:, 4] = 0.010 * np.sin(2.0 * np.pi * 0.47 * time_s + 0.3)
    actions[:, 5] = 0.008 * np.sin(2.0 * np.pi * 0.31 * time_s + 0.7)
    return actions


def save_action_sequence(
    path: Path,
    actions: np.ndarray,
    control_rate_hz: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(actions, columns=ACTION_COLUMNS)
    frame.insert(0, "time_s", np.arange(len(frame)) / control_rate_hz)
    frame.insert(0, "action_index", np.arange(len(frame), dtype=int))
    frame.to_csv(path, index=False)


def state_fields(state: Sequence[float]) -> dict[str, float]:
    values = np.asarray(state, dtype=np.float64).reshape(-1)
    if values.shape != (13,) or not np.isfinite(values).all():
        raise ValueError(f"Expected a finite 13-element state, received {values.shape}")
    return dict(zip(STATE_COLUMNS, values.tolist()))


def action_fields(action: Sequence[float]) -> dict[str, float]:
    values = np.asarray(action, dtype=np.float64).reshape(-1)
    if values.shape != (6,) or not np.isfinite(values).all():
        raise ValueError(f"Expected a finite 6-element wrench, received {values.shape}")
    return dict(zip(ACTION_COLUMNS, values.tolist()))


def validate_result_frame(frame: pd.DataFrame, source: Path) -> pd.DataFrame:
    missing = [column for column in REQUIRED_RESULT_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"{source} is missing result columns: {missing}")
    result = frame.copy()
    numeric = [
        column
        for column in REQUIRED_RESULT_COLUMNS
        if column != "backend"
    ]
    for column in numeric:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if result[numeric].isna().any(axis=None):
        # Unapplied ROS commands intentionally have no physical-application latency.
        allowed_nan = {
            "action_to_physics_latency_ms",
            "observation_age_wall_ms",
            "observation_age_sim_ms",
            "physics_updates_between_observation_and_action",
        }
        invalid_columns = [
            column
            for column in numeric
            if column not in allowed_nan and result[column].isna().any()
        ]
        if invalid_columns:
            raise ValueError(
                f"{source} contains invalid values in {invalid_columns}"
            )
    return result
