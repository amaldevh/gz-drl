#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Amal Dev Haridevan

"""Run process-isolated same-seed and different-seed PPO studies."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Sequence

from common import (
    DEFAULT_CONFIG,
    DEFAULT_RESULTS,
    configuration_hash,
    load_config,
    subprocess_environment,
    validate_config,
    write_json,
)


SCRIPT_ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=("same_seed", "different_seed"),
        default=("same_seed", "different_seed"),
    )
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument("--skip-determinism-check", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip completed runs/evaluations and continue an interrupted study.",
    )
    parser.add_argument("--total-timesteps", type=int)
    parser.add_argument("--same-seed-repetitions", type=int)
    parser.add_argument("--different-seeds", type=int, nargs="+")
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--env-num-threads", type=int)
    parser.add_argument("--env-batch-size", type=int)
    parser.add_argument("--episode-steps", type=int)
    parser.add_argument("--checkpoint-interval", type=int)
    parser.add_argument("--evaluation-episodes", type=int)
    parser.add_argument("--ppo-n-steps", type=int)
    parser.add_argument("--ppo-batch-size", type=int)
    return parser


def apply_overrides(config: dict, args: argparse.Namespace) -> dict:
    result = deepcopy(config)
    direct = (
        "total_timesteps",
        "same_seed_repetitions",
        "different_seeds",
        "num_envs",
        "env_num_threads",
        "env_batch_size",
        "episode_steps",
        "checkpoint_interval",
        "evaluation_episodes",
    )
    for key in direct:
        value = getattr(args, key)
        if value is not None:
            result[key] = value
    if args.ppo_n_steps is not None:
        result["ppo"]["n_steps"] = args.ppo_n_steps
    if args.ppo_batch_size is not None:
        result["ppo"]["batch_size"] = args.ppo_batch_size
    validate_config(result)
    return result


def requested_runs(config: dict, groups: Sequence[str]):
    if "same_seed" in groups:
        for replicate in range(1, int(config["same_seed_repetitions"]) + 1):
            yield "same_seed", replicate, int(config["same_seed"]), replicate - 1
    if "different_seed" in groups:
        for replicate, seed in enumerate(config["different_seeds"], start=1):
            yield (
                "different_seed",
                replicate,
                int(seed),
                int(config["same_seed_repetitions"]) + replicate - 1,
            )


def metadata_status(metadata_path: Path) -> str | None:
    if not metadata_path.is_file():
        return None
    return json.loads(metadata_path.read_text(encoding="utf-8")).get("status")


def has_training_artifacts(run_dir: Path) -> bool:
    return (
        any((run_dir / "checkpoints").glob("model_*_steps.zip"))
        or (run_dir / "monitor.csv").is_file()
    )


def training_partition_offset(config: dict, run_index: int) -> int:
    stride = max(100, int(config["num_envs"]) + 1)
    return int(config["gz_partition_offset"]) + run_index * stride


def run_command(command: list[str], config: dict, seed: int) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(
        command,
        check=True,
        cwd=SCRIPT_ROOT,
        env=subprocess_environment(
            seed, int(config["torch_threads"]), str(config["device"])
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = apply_overrides(load_config(args.config), args)
    results_dir = args.results_dir.expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    effective_config = results_dir / "effective_config.json"
    if effective_config.is_file():
        existing_config = load_config(effective_config)
        if configuration_hash(existing_config) != configuration_hash(config):
            raise ValueError(
                f"{results_dir} already contains a different effective "
                "configuration; choose a new --results-dir"
            )
    else:
        write_json(effective_config, config)

    runs = list(requested_runs(config, args.groups))

    if not args.skip_training and not args.skip_determinism_check:
        determinism_report = results_dir / "determinism_check.json"
        if determinism_report.exists():
            report = json.loads(determinism_report.read_text(encoding="utf-8"))
            if not args.resume:
                raise FileExistsError(
                    f"Refusing to overwrite determinism report: {determinism_report}"
                )
            if not report.get("passed", False):
                raise RuntimeError("Existing determinism report did not pass")
            print(f"Reusing passed determinism report: {determinism_report}")
        else:
            command = [
                sys.executable,
                str(SCRIPT_ROOT / "verify_determinism.py"),
                "--config",
                str(effective_config),
                "--output",
                str(determinism_report),
            ]
            run_command(command, config, int(config["same_seed"]))

    # Train every replicate first.  This prevents repeated Gazebo/Python native
    # module teardown from checkpoint evaluation occurring between trainings.
    for group, replicate, rl_seed, run_index in runs:
        run_dir = results_dir / group / f"run_{replicate:02d}_seed_{rl_seed}"
        metadata_path = run_dir / "metadata.json"
        if not args.skip_training:
            status = metadata_status(metadata_path)
            if status == "complete" and args.resume:
                print(f"Skipping completed training run: {run_dir}")
                continue
            if status is not None and not args.resume:
                raise FileExistsError(
                    f"Refusing to overwrite an existing run: {run_dir}. "
                    "Choose a new --results-dir or pass --resume."
                )
            if status is None and has_training_artifacts(run_dir):
                raise RuntimeError(
                    f"Found model data without metadata in {run_dir}; "
                    "refusing to overwrite it"
                )
            if args.resume and has_training_artifacts(run_dir):
                raise RuntimeError(
                    f"Cannot safely resume partially written model data in {run_dir}; "
                    "choose a new results directory"
                )
            command = [
                sys.executable,
                str(SCRIPT_ROOT / "train_one.py"),
                "--config",
                str(effective_config),
                "--run-dir",
                str(run_dir),
                "--group",
                group,
                "--replicate",
                str(replicate),
                "--rl-seed",
                str(rl_seed),
                "--partition-offset",
                str(training_partition_offset(config, run_index)),
            ]
            run_command(command, config, rl_seed)

    # Evaluate only after all requested training workers have exited.
    for group, replicate, rl_seed, run_index in runs:
        run_dir = results_dir / group / f"run_{replicate:02d}_seed_{rl_seed}"
        metadata_path = run_dir / "metadata.json"
        if metadata_status(metadata_path) != "complete":
            raise RuntimeError(f"Training run is not complete: {run_dir}")
        if not args.skip_evaluation:
            combined_evaluation = run_dir / "checkpoint_evaluations.csv"
            if args.resume and combined_evaluation.is_file():
                print(f"Skipping completed evaluation: {run_dir}")
                continue
            command = [
                sys.executable,
                str(SCRIPT_ROOT / "evaluate_checkpoints.py"),
                "--config",
                str(effective_config),
                "--run-dir",
                str(run_dir),
                "--run-index",
                str(run_index),
            ]
            run_command(command, config, int(config["evaluation_seed"]))

    if not args.skip_analysis:
        command = [
            sys.executable,
            str(SCRIPT_ROOT / "analyze_results.py"),
            "--results-dir",
            str(results_dir),
        ]
        run_command(command, config, int(config["evaluation_seed"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
