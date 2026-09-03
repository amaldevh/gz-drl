#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Amal Dev Haridevan

"""Analyze timing and repeated-trajectory consistency across backends."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from common import (
    DEFAULT_RESULTS_DIR,
    PHYSICS_DT,
    STATE_COLUMNS,
    validate_result_frame,
)


BACKEND_ORDER = ("gzdrl", "ros2_gazebo", "gazebo_transport")
BACKEND_LABELS = {
    "gzdrl": "GzDRL",
    "ros2_gazebo": "ROS 2 + Gazebo",
    "gazebo_transport": "Gazebo Transport",
}
BACKEND_COLORS = {
    "gzdrl": "#0072B2",
    "ros2_gazebo": "#D55E00",
    "gazebo_transport": "#009E73",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "lines.linewidth": 1.4,
            "axes.linewidth": 0.7,
            "grid.alpha": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze synchronization-validation CSV files"
    )
    parser.add_argument(
        "--results-dir", type=Path, default=DEFAULT_RESULTS_DIR
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("pdf", "svg", "png"),
        default=("pdf", "png"),
    )
    return parser


def load_results(results_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    raw_directory = results_dir / "raw"
    for path in sorted(raw_directory.glob("*/*.csv")):
        frame = validate_result_frame(pd.read_csv(path), path)
        frame["source_file"] = str(path)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No result CSV files found below {raw_directory}")
    result = pd.concat(frames, ignore_index=True)
    duplicates = result.duplicated(
        ["backend", "trial", "action_index"], keep=False
    )
    if duplicates.any():
        raise ValueError(
            "Duplicate backend/trial/action rows were found: "
            f"{int(duplicates.sum())} rows"
        )
    return result


def ordered_backends(frame: pd.DataFrame) -> list[str]:
    available = set(frame["backend"].astype(str))
    return [
        *[backend for backend in BACKEND_ORDER if backend in available],
        *sorted(available - set(BACKEND_ORDER)),
    ]


def trajectory_statistics(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, tuple[np.ndarray, np.ndarray]]]:
    records: list[dict[str, object]] = []
    spreads: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    state_columns = list(STATE_COLUMNS)

    for backend in ordered_backends(frame):
        backend_frame = frame[frame["backend"] == backend]
        trials = sorted(backend_frame["trial"].unique())
        common_actions: set[int] | None = None
        trial_frames: dict[int, pd.DataFrame] = {}
        for trial in trials:
            trial_frame = backend_frame[backend_frame["trial"] == trial].sort_values(
                "action_index"
            )
            trial_frames[int(trial)] = trial_frame
            indices = set(trial_frame["action_index"].astype(int))
            common_actions = indices if common_actions is None else common_actions & indices
        if not common_actions:
            continue
        action_indices = np.array(sorted(common_actions), dtype=int)
        state_stack = np.stack(
            [
                trial_frames[int(trial)]
                .set_index("action_index")
                .loc[action_indices, state_columns]
                .to_numpy(dtype=float)
                for trial in trials
            ]
        )
        mean_trajectory = np.mean(state_stack, axis=0)
        position_error = state_stack[:, :, :3] - mean_trajectory[None, :, :3]
        position_spread = np.sqrt(
            np.mean(np.sum(np.square(position_error), axis=2), axis=0)
        )
        spreads[backend] = (action_indices, position_spread)
        terminal_positions = state_stack[:, -1, :3]
        terminal_states = state_stack[:, -1, :]
        terminal_position_center = np.mean(terminal_positions, axis=0)
        terminal_position_dispersion = float(
            np.sqrt(
                np.mean(
                    np.sum(
                        np.square(
                            terminal_positions - terminal_position_center[None, :]
                        ),
                        axis=1,
                    )
                )
            )
        )
        terminal_state_variance = float(
            np.mean(np.var(terminal_states, axis=0, ddof=0))
        )
        for index, trial in enumerate(trials):
            rmse = float(
                np.sqrt(np.mean(np.sum(np.square(position_error[index]), axis=1)))
            )
            record: dict[str, object] = {
                "backend": backend,
                "trial": int(trial),
                "trajectory_position_rmse_m": rmse,
                "terminal_position_dispersion_m": terminal_position_dispersion,
                "terminal_state_variance": terminal_state_variance,
            }
            terminal = terminal_states[index]
            record.update(
                {
                    f"terminal_{column.removeprefix('state_')}": value
                    for column, value in zip(state_columns, terminal)
                }
            )
            records.append(record)
    return pd.DataFrame.from_records(records), spreads


def confidence_interval(values: np.ndarray) -> tuple[float, float, float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan, math.nan, math.nan, math.nan
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    if values.size > 1:
        margin = float(
            stats.t.ppf(0.975, values.size - 1) * std / math.sqrt(values.size)
        )
    else:
        margin = math.nan
    return mean, std, mean - margin, mean + margin


def summarize(
    frame: pd.DataFrame,
    reproducibility: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    trial_records: list[dict[str, object]] = []
    for (backend, trial), group in frame.groupby(["backend", "trial"], sort=True):
        applied = group[group["action_applied"] > 0.5]
        record: dict[str, object] = {
            "backend": backend,
            "trial": int(trial),
            "action_application_rate_pct": 100.0
            * float(group["action_applied"].mean()),
        }
        for metric in (
            "action_to_physics_latency_ms",
            "command_transport_latency_ms",
            "observation_age_wall_ms",
            "observation_age_sim_ms",
            "physics_updates_between_observation_and_action",
            "ack_round_trip_latency_ms",
        ):
            values = applied[metric].dropna().to_numpy(dtype=float)
            record[f"{metric}_mean"] = (
                float(np.mean(values)) if values.size else math.nan
            )
            record[f"{metric}_std"] = (
                float(np.std(values, ddof=1)) if values.size > 1 else 0.0
            )
        trial_records.append(record)
    per_trial = pd.DataFrame.from_records(trial_records)
    if not reproducibility.empty:
        per_trial = per_trial.merge(
            reproducibility,
            on=["backend", "trial"],
            how="left",
            validate="one_to_one",
        )

    backend_records: list[dict[str, object]] = []
    timing_metrics = (
        "action_to_physics_latency_ms",
        "observation_age_sim_ms",
        "physics_updates_between_observation_and_action",
        "command_transport_latency_ms",
        "ack_round_trip_latency_ms",
    )
    for backend, group in per_trial.groupby("backend", sort=True):
        backend_samples = frame[frame["backend"] == backend]
        applied_samples = backend_samples[
            backend_samples["action_applied"] > 0.5
        ]
        record = {
            "backend": backend,
            "trials": int(group["trial"].nunique()),
            "action_samples": len(backend_samples),
        }
        application_rates = group["action_application_rate_pct"].to_numpy(
            dtype=float
        )
        mean, std, low, high = confidence_interval(application_rates)
        record["action_application_rate_pct_mean"] = mean
        record["action_application_rate_pct_std"] = std
        record["action_application_rate_pct_ci95_low"] = (
            max(0.0, low) if np.isfinite(low) else math.nan
        )
        record["action_application_rate_pct_ci95_high"] = (
            min(100.0, high) if np.isfinite(high) else math.nan
        )

        for metric in timing_metrics:
            values = applied_samples[metric].dropna().to_numpy(dtype=float)
            record[f"{metric}_mean"] = (
                float(np.mean(values)) if values.size else math.nan
            )
            record[f"{metric}_std"] = (
                float(np.std(values, ddof=1)) if values.size > 1 else 0.0
            )
            trial_metric = f"{metric}_mean"
            if trial_metric in group:
                _, _, low, high = confidence_interval(
                    group[trial_metric].to_numpy(dtype=float)
                )
                record[f"{metric}_ci95_low"] = (
                    max(0.0, low) if np.isfinite(low) else math.nan
                )
                record[f"{metric}_ci95_high"] = high

        if "trajectory_position_rmse_m" in group:
            mean, std, low, high = confidence_interval(
                group["trajectory_position_rmse_m"].to_numpy(dtype=float)
            )
            record["trajectory_position_rmse_m_mean"] = mean
            record["trajectory_position_rmse_m_std"] = std
            record["trajectory_position_rmse_m_ci95_low"] = (
                max(0.0, low) if np.isfinite(low) else math.nan
            )
            record["trajectory_position_rmse_m_ci95_high"] = high
        for metric in (
            "terminal_position_dispersion_m",
            "terminal_state_variance",
        ):
            if metric in group:
                record[metric] = float(group[metric].dropna().iloc[0])
        backend_records.append(record)
    return per_trial, pd.DataFrame.from_records(backend_records)


def save_figure(
    figure,
    stem: Path,
    formats: Sequence[str],
) -> None:
    for extension in formats:
        kwargs = {"bbox_inches": "tight"}
        if extension == "png":
            kwargs["dpi"] = 600
        figure.savefig(stem.with_suffix(f".{extension}"), **kwargs)
    plt.close(figure)


def plot_timing(
    frame: pd.DataFrame,
    output_dir: Path,
    formats: Sequence[str],
) -> None:
    backends = ordered_backends(frame)
    applied = frame[frame["action_applied"] > 0.5]
    metrics = (
        ("action_to_physics_latency_ms", "Action-to-physics latency (ms)"),
        ("observation_age_sim_ms", "Observation age at application (ms)"),
        (
            "physics_updates_between_observation_and_action",
            "Intervening physics updates",
        ),
    )
    figure, axes = plt.subplots(1, 3, figsize=(7.16, 2.55))
    for panel, (axis, (metric, ylabel)) in enumerate(zip(axes, metrics)):
        values = [
            applied.loc[applied["backend"] == backend, metric]
            .dropna()
            .to_numpy(dtype=float)
            for backend in backends
        ]
        plot = axis.boxplot(values, patch_artist=True, showfliers=False)
        for patch, backend in zip(plot["boxes"], backends):
            patch.set_facecolor(BACKEND_COLORS.get(backend, "#777777"))
            patch.set_alpha(0.55)
        axis.set_xticks(
            range(1, len(backends) + 1),
            [BACKEND_LABELS.get(item, item) for item in backends],
            rotation=15,
            ha="right",
        )
        axis.set_ylabel(ylabel)
        axis.grid(axis="y")
        axis.text(
            -0.13,
            1.03,
            f"({chr(ord('a') + panel)})",
            transform=axis.transAxes,
            fontweight="bold",
        )
    figure.subplots_adjust(left=0.08, right=0.995, bottom=0.27, top=0.93, wspace=0.36)
    save_figure(figure, output_dir / "fig_synchronization_timing", formats)


def plot_reproducibility(
    reproducibility: pd.DataFrame,
    spreads: dict[str, tuple[np.ndarray, np.ndarray]],
    output_dir: Path,
    formats: Sequence[str],
) -> None:
    if reproducibility.empty:
        return
    backends = ordered_backends(reproducibility)
    figure, axes = plt.subplots(1, 2, figsize=(7.16, 2.55))
    for backend in backends:
        if backend not in spreads:
            continue
        action_indices, spread = spreads[backend]
        axes[0].plot(
            action_indices,
            spread,
            color=BACKEND_COLORS.get(backend, "#777777"),
            label=BACKEND_LABELS.get(backend, backend),
        )
    axes[0].set_xlabel("Action index")
    axes[0].set_ylabel("Cross-run position spread (m)")
    axes[0].grid(True)
    axes[0].legend(frameon=False)

    values = [
        reproducibility.loc[
            reproducibility["backend"] == backend,
            "trajectory_position_rmse_m",
        ].to_numpy(dtype=float)
        for backend in backends
    ]
    plot = axes[1].boxplot(values, patch_artist=True, showfliers=False)
    for patch, backend in zip(plot["boxes"], backends):
        patch.set_facecolor(BACKEND_COLORS.get(backend, "#777777"))
        patch.set_alpha(0.55)
    axes[1].set_xticks(
        range(1, len(backends) + 1),
        [BACKEND_LABELS.get(item, item) for item in backends],
        rotation=15,
        ha="right",
    )
    axes[1].set_ylabel("Trajectory RMSE to run mean (m)")
    axes[1].grid(axis="y")
    for panel, axis in enumerate(axes):
        axis.text(
            -0.13,
            1.03,
            f"({chr(ord('a') + panel)})",
            transform=axis.transAxes,
            fontweight="bold",
        )
    figure.subplots_adjust(left=0.09, right=0.995, bottom=0.27, top=0.93, wspace=0.32)
    save_figure(figure, output_dir / "fig_trajectory_reproducibility", formats)


def estimate_cell(mean: float, std: float, decimals: int = 2) -> str:
    if not np.isfinite(mean):
        return r"\textemdash"
    return rf"\({mean:.{decimals}f} \mathbin{{\pm}} {std:.{decimals}f}\)"


def render_table(summary: pd.DataFrame) -> str:
    lines = [
        r"% Requires \usepackage{booktabs,graphicx}.",
        r"\begin{table*}[t]",
        r"\centering",
        (
            r"\caption{Transition synchronization and repeated-run consistency. "
            r"Timing entries are means $\mathbin{\pm}$ standard deviations; "
            r"trajectory RMSE is computed relative to the backend's mean trajectory.}"
        ),
        r"\label{tab:synchronization-validation}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        (
            r"\textbf{Backend} & \textbf{Applied (\%)} & "
            r"\textbf{Action latency (ms)} & \textbf{Observation age (ms)} & "
            r"\textbf{Intervening updates} & \textbf{Trajectory RMSE (m)} & "
            r"\textbf{Terminal dispersion (m)} \\"
        ),
        r"\midrule",
    ]
    order = {backend: index for index, backend in enumerate(BACKEND_ORDER)}
    ordered = summary.assign(
        _order=summary["backend"].map(order).fillna(len(order))
    ).sort_values("_order")
    for _, row in ordered.iterrows():
        cells = [
            BACKEND_LABELS.get(row["backend"], row["backend"]),
            estimate_cell(
                row["action_application_rate_pct_mean"],
                row["action_application_rate_pct_std"],
                1,
            ),
            estimate_cell(
                row["action_to_physics_latency_ms_mean"],
                row["action_to_physics_latency_ms_std"],
            ),
            estimate_cell(
                row["observation_age_sim_ms_mean"],
                row["observation_age_sim_ms_std"],
            ),
            estimate_cell(
                row["physics_updates_between_observation_and_action_mean"],
                row["physics_updates_between_observation_and_action_std"],
            ),
            estimate_cell(
                row["trajectory_position_rmse_m_mean"],
                row["trajectory_position_rmse_m_std"],
                4,
            ),
            f"{row['terminal_position_dispersion_m']:.4f}",
        ]
        lines.append(" & ".join(str(cell) for cell in cells) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results_dir = args.results_dir.expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    frame = load_results(results_dir)
    reproducibility, spreads = trajectory_statistics(frame)
    per_trial, summary = summarize(frame, reproducibility)
    frame.to_csv(results_dir / "all_transition_samples.csv", index=False)
    reproducibility.to_csv(
        results_dir / "reproducibility_by_trial.csv", index=False
    )
    per_trial.to_csv(results_dir / "synchronization_by_trial.csv", index=False)
    summary.to_csv(results_dir / "synchronization_summary.csv", index=False)
    (results_dir / "table_synchronization_validation.tex").write_text(
        render_table(summary), encoding="utf-8"
    )
    plot_timing(frame, results_dir, args.formats)
    plot_reproducibility(reproducibility, spreads, results_dir, args.formats)
    manifest = {
        "physics_dt_s": PHYSICS_DT,
        "backends": ordered_backends(frame),
        "trials_per_backend": {
            backend: int(group["trial"].nunique())
            for backend, group in frame.groupby("backend")
        },
        "metric_definitions": {
            "action_to_physics_latency_ms": (
                "wall time from action dispatch to completion of the first "
                "physics update using that action"
            ),
            "observation_age_sim_ms": (
                "(applied physics iteration - observation iteration) * physics dt"
            ),
            "physics_updates_between_observation_and_action": (
                "max(applied iteration - observation iteration - 1, 0)"
            ),
            "trajectory_position_rmse_m": (
                "per-run position RMSE relative to the backend mean trajectory, "
                "aligned by action index"
            ),
        },
    }
    (results_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
