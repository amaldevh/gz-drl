#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Amal Dev Haridevan

"""Train one process-isolated PPO replicate with a fully pinned configuration."""

from __future__ import annotations

import argparse
import faulthandler
import time
from pathlib import Path
from typing import Sequence

from common import (
    environment_kwargs,
    load_config,
    ppo_constructor_kwargs,
    runtime_metadata,
    seed_everything,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--group", choices=("same_seed", "different_seed"), required=True
    )
    parser.add_argument("--replicate", type=int, required=True)
    parser.add_argument("--rl-seed", type=int, required=True)
    parser.add_argument("--partition-offset", type=int, required=True)
    return parser


def save_checkpoint(model, vec_normalize, checkpoint_dir: Path) -> None:
    step = int(model.num_timesteps)
    model.save(checkpoint_dir / f"model_{step}_steps")
    vec_normalize.save(checkpoint_dir / f"model_vecnormalize_{step}_steps.pkl")


def main(argv: Sequence[str] | None = None) -> int:
    faulthandler.enable(all_threads=True)
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    run_dir = args.run_dir.expanduser().resolve()
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("[train] Seeding Python and PyTorch", flush=True)
    seed_everything(args.rl_seed, int(config["torch_threads"]))

    print("[train] Importing environment and PPO modules", flush=True)
    import gzdrl.envs as envs
    from gzdrl.envs.env_adapters import VecAdapter
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.vec_env import VecMonitor, VecNormalize

    metadata = {
        "status": "running",
        "group": args.group,
        "replicate": args.replicate,
        "rl_seed": args.rl_seed,
        "environment_seed": int(config["environment_seed"]),
        "gz_partition_offset": int(args.partition_offset),
        "started_unix_s": time.time(),
        "runtime": runtime_metadata(config),
    }
    write_json(run_dir / "metadata.json", metadata)

    print(
        f"[train] Constructing {config['num_envs']} hover environments at "
        f"partition offset {args.partition_offset}",
        flush=True,
    )
    raw_env = envs.make(
        task_id=str(config["environment"]),
        env_type="gymnasium",
        **environment_kwargs(
            config,
            evaluation=False,
            partition_offset=args.partition_offset,
        ),
    )
    raw_env.spec.id = str(config["environment"])
    adapted_env = VecAdapter(raw_env)
    vec_normalize = VecNormalize(
        adapted_env,
        training=True,
        norm_obs=bool(config["normalize_observations"]),
        norm_reward=False,
    )
    training_env = VecMonitor(vec_normalize, filename=str(run_dir / "monitor.csv"))

    print("[train] Constructing PPO model", flush=True)
    model = PPO(
        "MlpPolicy",
        training_env,
        **ppo_constructor_kwargs(config),
        seed=args.rl_seed,
        device=str(config["device"]),
        tensorboard_log=str(run_dir / "tensorboard"),
        verbose=1,
    )
    save_checkpoint(model, vec_normalize, checkpoint_dir)
    checkpoint_callback = CheckpointCallback(
        save_freq=max(
            1,
            int(config["checkpoint_interval"]) // int(config["num_envs"]),
        ),
        save_path=str(checkpoint_dir),
        name_prefix="model",
        save_vecnormalize=True,
        verbose=1,
    )
    try:
        print("[train] Starting PPO learning", flush=True)
        model.learn(
            total_timesteps=int(config["total_timesteps"]),
            callback=checkpoint_callback,
            log_interval=1,
            progress_bar=False,
        )
        save_checkpoint(model, vec_normalize, checkpoint_dir)
        metadata.update(
            {
                "status": "complete",
                "finished_unix_s": time.time(),
                "actual_timesteps": int(model.num_timesteps),
            }
        )
        write_json(run_dir / "metadata.json", metadata)
    except Exception as error:
        metadata.update(
            {
                "status": "failed",
                "finished_unix_s": time.time(),
                "error": f"{type(error).__name__}: {error}",
            }
        )
        write_json(run_dir / "metadata.json", metadata)
        raise
    finally:
        training_env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
