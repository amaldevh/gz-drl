#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Amal Dev Haridevan

"""Evaluate every checkpoint on an identical process-reset episode sequence."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd

from common import (
    environment_kwargs,
    load_config,
    policy_parameter_hash,
    seed_everything,
    subprocess_environment,
)


CHECKPOINT_PATTERN = re.compile(r"model_(?P<step>\d+)_steps\.zip")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--normalizer", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--run-index", type=int, default=0)
    parser.add_argument("--test-partition-id", type=int, default=None)
    return parser


def discover_checkpoints(run_dir: Path) -> list[tuple[int, Path, Path]]:
    checkpoints: dict[int, tuple[Path, Path]] = {}
    checkpoint_dir = run_dir / "checkpoints"
    for model_path in sorted(checkpoint_dir.glob("model_*_steps.zip")):
        match = CHECKPOINT_PATTERN.fullmatch(model_path.name)
        if match is None:
            continue
        step = int(match.group("step"))
        normalizer_path = checkpoint_dir / f"model_vecnormalize_{step}_steps.pkl"
        if not normalizer_path.is_file():
            raise FileNotFoundError(
                f"Missing VecNormalize state for {model_path}: {normalizer_path}"
            )
        checkpoints[step] = (model_path, normalizer_path)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    return [(step, *checkpoints[step]) for step in sorted(checkpoints)]


def evaluate_one(
    config: dict,
    run_dir: Path,
    checkpoint: Path,
    normalizer: Path,
    test_partition_id: int,
) -> pd.DataFrame:
    seed_everything(int(config["evaluation_seed"]), int(config["torch_threads"]))

    import gzdrl.envs as envs
    from gzdrl.envs.env_adapters import VecAdapter
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecMonitor, VecNormalize

    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    match = CHECKPOINT_PATTERN.fullmatch(checkpoint.name)
    if match is None:
        raise ValueError(f"Invalid checkpoint filename: {checkpoint.name}")
    checkpoint_steps = int(match.group("step"))

    raw_env = envs.make(
        task_id=str(config["environment"]),
        env_type="gymnasium",
        **environment_kwargs(
            config,
            evaluation=True,
            evaluation_partition_id=test_partition_id,
        ),
    )
    raw_env.spec.id = str(config["environment"])
    normalized_env = VecNormalize.load(str(normalizer), VecAdapter(raw_env))
    normalized_env.training = False
    normalized_env.norm_reward = False
    evaluation_env = VecMonitor(normalized_env)
    model = PPO.load(str(checkpoint), env=evaluation_env, device=str(config["device"]))
    parameter_hash = policy_parameter_hash(model.policy)

    records: list[dict[str, object]] = []
    observation = evaluation_env.reset()
    episode_return = 0.0
    episode_length = 0
    try:
        while len(records) < int(config["evaluation_episodes"]):
            action, _ = model.predict(observation, deterministic=True)
            observation, rewards, dones, _ = evaluation_env.step(action)
            episode_return += float(rewards[0])
            episode_length += 1
            if bool(dones[0]):
                records.append(
                    {
                        "group": metadata["group"],
                        "replicate": int(metadata["replicate"]),
                        "rl_seed": int(metadata["rl_seed"]),
                        "checkpoint_steps": checkpoint_steps,
                        "episode": len(records),
                        "episode_return": episode_return,
                        "episode_length": episode_length,
                        "success": int(episode_length >= int(config["episode_steps"])),
                        "policy_parameter_sha256": parameter_hash,
                        "checkpoint": str(checkpoint),
                        "normalizer": str(normalizer),
                        "test_partition_id": test_partition_id,
                    }
                )
                episode_return = 0.0
                episode_length = 0
    finally:
        evaluation_env.close()
    return pd.DataFrame.from_records(records)


def run_worker(args: argparse.Namespace) -> int:
    if args.checkpoint is None or args.normalizer is None or args.output is None:
        raise ValueError("Worker mode requires checkpoint, normalizer, and output")
    if args.test_partition_id is None:
        raise ValueError("Worker mode requires --test-partition-id")
    config = load_config(args.config)
    run_dir = args.run_dir.expanduser().resolve()
    frame = evaluate_one(
        config,
        run_dir,
        args.checkpoint.expanduser().resolve(),
        args.normalizer.expanduser().resolve(),
        int(args.test_partition_id),
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return 0


def run_parent(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    run_dir = args.run_dir.expanduser().resolve()
    evaluation_dir = run_dir / "fixed_evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    for checkpoint_index, (step, checkpoint, normalizer) in enumerate(
        discover_checkpoints(run_dir)
    ):
        test_partition_id = (
            int(config["evaluation_partition_id"])
            + int(args.run_index) * 1000
            + checkpoint_index
        )
        output = evaluation_dir / f"checkpoint_{step}_episodes.csv"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--config",
            str(args.config.expanduser().resolve()),
            "--run-dir",
            str(run_dir),
            "--checkpoint",
            str(checkpoint),
            "--normalizer",
            str(normalizer),
            "--output",
            str(output),
            "--test-partition-id",
            str(test_partition_id),
        ]
        subprocess.run(
            command,
            check=True,
            env=subprocess_environment(
                int(config["evaluation_seed"]),
                int(config["torch_threads"]),
                str(config["device"]),
            ),
        )
        frames.append(pd.read_csv(output))
        print(f"Evaluated {run_dir.name} at {step} steps")
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(run_dir / "checkpoint_evaluations.csv", index=False)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    worker_arguments = (args.checkpoint, args.normalizer, args.output)
    if any(value is not None for value in worker_arguments):
        return run_worker(args)
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
