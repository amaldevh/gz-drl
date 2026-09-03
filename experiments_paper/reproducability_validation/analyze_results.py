#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Amal Dev Haridevan

"""Aggregate fixed-evaluation returns and create paper-ready artifacts."""

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

from common import DEFAULT_RESULTS, configuration_hash, load_config, write_json


GROUP_LABELS = {
    "same_seed": "Same seed",
    "different_seed": "Different seeds",
}
GROUP_ORDER = tuple(GROUP_LABELS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--dpi", type=int, default=300)
    return parser


def sample_std(series: pd.Series) -> float:
    return float(series.std(ddof=1)) if len(series) > 1 else 0.0


def safe_cv(mean: float, standard_deviation: float) -> float:
    return standard_deviation / abs(mean) if not math.isclose(mean, 0.0) else math.nan


def load_results(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    effective_config_path = results_dir / "effective_config.json"
    if not effective_config_path.is_file():
        raise FileNotFoundError(
            f"Missing effective configuration: {effective_config_path}"
        )
    effective_config = load_config(effective_config_path)
    frames: list[pd.DataFrame] = []
    metadata_records: list[dict] = []
    for evaluation_path in sorted(
        results_dir.glob("*/run_*/checkpoint_evaluations.csv")
    ):
        run_dir = evaluation_path.parent
        metadata_path = run_dir / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Missing run metadata: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("status") != "complete":
            raise RuntimeError(f"Incomplete run: {run_dir}")
        frame = pd.read_csv(evaluation_path)
        frame["run_name"] = run_dir.name
        frames.append(frame)
        metadata_records.append(
            {
                "group": metadata["group"],
                "run_name": run_dir.name,
                "hostname": metadata["runtime"]["hostname"],
                "configuration_sha256": metadata["runtime"]["configuration_sha256"],
            }
        )
    if not frames:
        raise FileNotFoundError(
            f"No */run_*/checkpoint_evaluations.csv files under {results_dir}"
        )
    episodes = pd.concat(frames, ignore_index=True)
    metadata_frame = pd.DataFrame.from_records(metadata_records)
    if metadata_frame["hostname"].nunique() != 1:
        raise ValueError("All runs must execute on the same machine")
    if metadata_frame["configuration_sha256"].nunique() != 1:
        raise ValueError("All runs must use the same effective configuration")
    if metadata_frame["configuration_sha256"].iloc[0] != configuration_hash(
        effective_config
    ):
        raise ValueError("Run metadata does not match effective_config.json")
    for group, group_metadata in metadata_frame.groupby("group"):
        expected = (
            int(effective_config["same_seed_repetitions"])
            if group == "same_seed"
            else len(effective_config["different_seeds"])
        )
        actual = int(group_metadata["run_name"].nunique())
        if actual != expected:
            raise ValueError(f"{group} contains {actual} runs; expected {expected}")
    expected_episodes = int(effective_config["evaluation_episodes"])
    episode_counts = episodes.groupby(["group", "run_name", "checkpoint_steps"]).size()
    if not (episode_counts == expected_episodes).all():
        raise ValueError("At least one checkpoint has an incomplete evaluation set")
    checkpoint_sets = episodes.groupby(["group", "run_name"])["checkpoint_steps"].apply(
        lambda values: tuple(sorted(values.unique()))
    )
    if checkpoint_sets.map(str).nunique() != 1:
        raise ValueError("All runs must contain the same checkpoint steps")
    return episodes, metadata_frame, effective_config


def aggregate(episodes: pd.DataFrame):
    keys = ["group", "run_name", "replicate", "rl_seed", "checkpoint_steps"]
    run_statistics = (
        episodes.groupby(keys, as_index=False)
        .agg(
            evaluation_return_mean=("episode_return", "mean"),
            evaluation_return_std=("episode_return", sample_std),
            success_rate=("success", "mean"),
            evaluation_episodes=("episode", "count"),
            policy_parameter_sha256=("policy_parameter_sha256", "first"),
        )
        .sort_values(keys)
    )
    variability = (
        run_statistics.groupby(["group", "checkpoint_steps"], as_index=False)
        .agg(
            runs=("run_name", "nunique"),
            return_mean=("evaluation_return_mean", "mean"),
            return_std=("evaluation_return_mean", sample_std),
            return_min=("evaluation_return_mean", "min"),
            return_max=("evaluation_return_mean", "max"),
            success_rate_mean=("success_rate", "mean"),
            unique_parameter_hashes=("policy_parameter_sha256", "nunique"),
        )
        .sort_values(["group", "checkpoint_steps"])
    )
    variability["return_cv"] = variability.apply(
        lambda row: safe_cv(row["return_mean"], row["return_std"]), axis=1
    )
    variability["parameters_identical"] = variability["unique_parameter_hashes"] == 1

    final_indices = run_statistics.groupby(["group", "run_name"])[
        "checkpoint_steps"
    ].idxmax()
    final_runs = run_statistics.loc[final_indices].sort_values(["group", "replicate"])
    summary_records: list[dict[str, object]] = []
    for group, final_group in final_runs.groupby("group", sort=False):
        checkpoint_group = variability[variability["group"] == group]
        final_mean = float(final_group["evaluation_return_mean"].mean())
        final_std = sample_std(final_group["evaluation_return_mean"])
        summary_records.append(
            {
                "group": group,
                "configuration": GROUP_LABELS.get(group, group),
                "runs": int(final_group["run_name"].nunique()),
                "final_checkpoint_steps": int(final_group["checkpoint_steps"].min()),
                "final_return_mean": final_mean,
                "final_return_std": final_std,
                "final_return_cv": safe_cv(final_mean, final_std),
                "final_success_rate_mean": float(final_group["success_rate"].mean()),
                "final_success_rate_std": sample_std(final_group["success_rate"]),
                "mean_checkpoint_return_std": float(
                    checkpoint_group["return_std"].mean()
                ),
                "identical_parameter_checkpoints_fraction": float(
                    checkpoint_group["parameters_identical"].mean()
                ),
            }
        )
    return run_statistics, variability, final_runs, pd.DataFrame(summary_records)


def apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.constrained_layout.use": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig, base_path: Path, dpi: int) -> None:
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(base_path.with_suffix(f".{suffix}"), dpi=dpi, bbox_inches="tight")


def plot_learning_curves(run_statistics: pd.DataFrame, output: Path, dpi: int) -> None:
    available = [
        group for group in GROUP_ORDER if group in set(run_statistics["group"])
    ]
    fig, axes = plt.subplots(
        1, len(available), figsize=(3.45 * len(available), 2.65), squeeze=False
    )
    colors = plt.get_cmap("tab10")
    for axis, group in zip(axes[0], available):
        group_frame = run_statistics[run_statistics["group"] == group]
        for index, (_run_name, run_frame) in enumerate(group_frame.groupby("run_name")):
            seed = int(run_frame["rl_seed"].iloc[0])
            label = f"Run {index + 1}" if group == "same_seed" else f"Seed {seed}"
            axis.plot(
                run_frame["checkpoint_steps"].to_numpy(),
                run_frame["evaluation_return_mean"].to_numpy(),
                color=colors(index % 10),
                linewidth=1.0,
                alpha=0.72,
                label=label,
            )
        pivot = group_frame.pivot(
            index="checkpoint_steps",
            columns="run_name",
            values="evaluation_return_mean",
        )
        mean = pivot.mean(axis=1)
        std = pivot.std(axis=1, ddof=1).fillna(0.0)
        steps = mean.index.to_numpy(dtype=float)
        mean_values = mean.to_numpy(dtype=float)
        std_values = std.to_numpy(dtype=float)
        axis.plot(
            steps, mean_values, color="black", linewidth=2.0, label="Across-run mean"
        )
        axis.fill_between(
            steps,
            mean_values - std_values,
            mean_values + std_values,
            color="black",
            alpha=0.12,
        )
        axis.set_title(GROUP_LABELS[group])
        axis.set_xlabel("Environment steps")
        axis.grid(axis="y", color="0.88", linewidth=0.6)
        axis.legend(frameon=False, ncol=2, loc="best")
    axes[0, 0].set_ylabel("Fixed-set evaluation return")
    save_figure(fig, output / "fig_training_reproducibility", dpi)
    plt.close(fig)


def plot_variability(variability: pd.DataFrame, output: Path, dpi: int) -> None:
    fig, axis = plt.subplots(figsize=(3.5, 2.55))
    styles = {"same_seed": ("#0072B2", "o"), "different_seed": ("#D55E00", "s")}
    for group in GROUP_ORDER:
        frame = variability[variability["group"] == group]
        if frame.empty:
            continue
        color, marker = styles[group]
        axis.plot(
            frame["checkpoint_steps"].to_numpy(),
            frame["return_std"].to_numpy(),
            color=color,
            marker=marker,
            markersize=3.5,
            linewidth=1.4,
            label=GROUP_LABELS[group],
        )
    axis.set_xlabel("Environment steps")
    axis.set_ylabel("Across-run return standard deviation")
    axis.grid(axis="y", color="0.88", linewidth=0.6)
    axis.legend(frameon=False)
    save_figure(fig, output / "fig_training_variability", dpi)
    plt.close(fig)


def latex_number(value: float, digits: int = 2) -> str:
    return "--" if not np.isfinite(value) else f"{value:.{digits}f}"


def write_latex_table(summary: pd.DataFrame, path: Path) -> None:
    rows: list[str] = []
    for group in GROUP_ORDER:
        selected = summary[summary["group"] == group]
        if selected.empty:
            continue
        row = selected.iloc[0]
        final_return = (
            f"${latex_number(row['final_return_mean'])} \\pm "
            f"{latex_number(row['final_return_std'])}$"
        )
        success = 100.0 * float(row["final_success_rate_mean"])
        rows.append(
            f"{row['configuration']} & {int(row['runs'])} & {final_return} & "
            f"{latex_number(100.0 * row['final_return_cv'])} & "
            f"{latex_number(success)} & "
            f"{latex_number(row['mean_checkpoint_return_std'])} \\\\"
        )
    contents = "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Training reproducibility on the GzDRL hover task. Returns are measured on the same fixed deterministic evaluation set.}",
            r"\label{tab:training_reproducibility}",
            r"\footnotesize",
            r"\setlength{\tabcolsep}{3.5pt}",
            r"\renewcommand{\arraystretch}{1.15}",
            r"\begin{tabular}{lrrrrr}",
            r"\toprule",
            r"\textbf{Configuration} & \textbf{Runs} & \textbf{Final return} & \textbf{CV (\%)} & \textbf{Success (\%)} & \makecell{\textbf{Mean checkpoint}\\\textbf{$\sigma$}} \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text(contents, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results_dir = args.results_dir.expanduser().resolve()
    output = results_dir / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    episodes, metadata, effective_config = load_results(results_dir)
    run_statistics, variability, final_runs, summary = aggregate(episodes)
    run_statistics.to_csv(output / "checkpoint_run_statistics.csv", index=False)
    variability.to_csv(output / "checkpoint_variability.csv", index=False)
    final_runs.to_csv(output / "final_run_statistics.csv", index=False)
    summary.to_csv(output / "training_reproducibility_summary.csv", index=False)
    apply_plot_style()
    plot_learning_curves(run_statistics, output, args.dpi)
    plot_variability(variability, output, args.dpi)
    write_latex_table(summary, output / "table_training_reproducibility.tex")
    write_json(
        output / "analysis_manifest.json",
        {
            "hostname": metadata["hostname"].iloc[0],
            "configuration_sha256": metadata["configuration_sha256"].iloc[0],
            "environment": effective_config["environment"],
            "groups": sorted(summary["group"].tolist()),
            "run_count": int(metadata["run_name"].nunique()),
            "evaluation_episode_count": int(len(episodes)),
        },
    )
    print(f"Wrote reproducibility artifacts to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
