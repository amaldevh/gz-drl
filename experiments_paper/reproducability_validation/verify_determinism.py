#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Amal Dev Haridevan

"""Require two fresh hover workers to match through one PPO update."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np

from common import (
    DEFAULT_CONFIG,
    environment_kwargs,
    load_config,
    policy_parameter_hash,
    ppo_constructor_kwargs,
    seed_everything,
    subprocess_environment,
    write_json,
)


MATCH_KEYS = (
    "initial_observation_sha256",
    "transition_sequence_sha256",
    "initial_policy_sha256",
    "post_update_policy_sha256",
    "post_update_obs_rms_sha256",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--transition-steps", type=int, default=16)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    return parser


def update_array_hash(digest, label: str, value) -> None:
    array = np.ascontiguousarray(value)
    digest.update(label.encode("utf-8"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())


def observation_statistics_hash(vec_normalize) -> str:
    digest = hashlib.sha256()
    update_array_hash(digest, "mean", vec_normalize.obs_rms.mean)
    update_array_hash(digest, "var", vec_normalize.obs_rms.var)
    update_array_hash(digest, "count", vec_normalize.obs_rms.count)
    return digest.hexdigest()


def run_worker(args: argparse.Namespace) -> int:
    if args.worker_output is None:
        raise ValueError("Worker mode requires --worker-output")
    config = load_config(args.config)
    if int(config["num_envs"]) < 2:
        raise ValueError("The determinism gate requires at least two environments")

    rl_seed = int(config["same_seed"])
    seed_everything(rl_seed, int(config["torch_threads"]))

    import gzdrl.envs as envs
    from gzdrl.envs.env_adapters import VecAdapter
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize

    raw_env = envs.make(
        task_id=str(config["environment"]),
        env_type="gymnasium",
        **environment_kwargs(config, evaluation=False),
    )
    raw_env.spec.id = str(config["environment"])
    try:
        observation, info = raw_env.reset()
        initial_digest = hashlib.sha256()
        update_array_hash(initial_digest, "observation", observation)
        update_array_hash(initial_digest, "env_id", info["env_id"])

        transition_digest = hashlib.sha256()
        update_array_hash(transition_digest, "initial_observation", observation)
        num_envs = int(config["num_envs"])
        rotor_index = np.arange(4, dtype=np.float32)[None, :]
        env_index = np.arange(num_envs, dtype=np.float32)[:, None]
        for step in range(int(args.transition_steps)):
            actions = 0.05 * np.sin(
                0.17 * float(step + 1) + 0.11 * env_index + 0.07 * rotor_index
            ).astype(np.float32)
            observation, rewards, terms, truncs, step_info = raw_env.step(actions)
            update_array_hash(transition_digest, f"action_{step}", actions)
            update_array_hash(transition_digest, f"observation_{step}", observation)
            update_array_hash(transition_digest, f"reward_{step}", rewards)
            update_array_hash(transition_digest, f"term_{step}", terms)
            update_array_hash(transition_digest, f"trunc_{step}", truncs)
            update_array_hash(transition_digest, f"env_id_{step}", step_info["env_id"])

        vec_normalize = VecNormalize(
            VecAdapter(raw_env),
            training=True,
            norm_obs=bool(config["normalize_observations"]),
            norm_reward=False,
        )
        model = PPO(
            "MlpPolicy",
            vec_normalize,
            **ppo_constructor_kwargs(config),
            seed=rl_seed,
            device=str(config["device"]),
            verbose=0,
        )
        initial_policy_hash = policy_parameter_hash(model.policy)
        initial_policy_std = float(model.policy.log_std.detach().cpu().exp().mean())
        rollout_size = int(config["num_envs"]) * int(config["ppo"]["n_steps"])
        model.learn(
            total_timesteps=rollout_size,
            log_interval=None,
            progress_bar=False,
        )
        result = {
            "num_envs": num_envs,
            "transition_steps": int(args.transition_steps),
            "ppo_update_timesteps": int(model.num_timesteps),
            "initial_observation_sha256": initial_digest.hexdigest(),
            "transition_sequence_sha256": transition_digest.hexdigest(),
            "initial_policy_sha256": initial_policy_hash,
            "post_update_policy_sha256": policy_parameter_hash(model.policy),
            "post_update_obs_rms_sha256": observation_statistics_hash(vec_normalize),
            "initial_policy_std": initial_policy_std,
        }
    finally:
        raw_env.close()
    write_json(args.worker_output.expanduser().resolve(), result)
    return 0


def run_parent(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if int(config["num_envs"]) < 2:
        raise ValueError("The determinism gate requires at least two environments")
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else Path.cwd() / "determinism_check.json"
    )
    replicas: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="gzdrl_hover_determinism_") as tmp:
        temporary_root = Path(tmp)
        for replica in (1, 2):
            worker_output = temporary_root / f"replica_{replica}.json"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--config",
                str(args.config.expanduser().resolve()),
                "--transition-steps",
                str(args.transition_steps),
                "--worker-output",
                str(worker_output),
            ]
            completed = subprocess.run(
                command,
                cwd=Path(__file__).resolve().parent,
                env=subprocess_environment(
                    int(config["same_seed"]),
                    int(config["torch_threads"]),
                    str(config["device"]),
                ),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            if completed.returncode != 0:
                print(completed.stdout, file=sys.stderr)
                raise subprocess.CalledProcessError(
                    completed.returncode, command, output=completed.stdout
                )
            replicas.append(json.loads(worker_output.read_text(encoding="utf-8")))

    comparisons = {key: replicas[0][key] == replicas[1][key] for key in MATCH_KEYS}
    report = {
        "passed": all(comparisons.values()),
        "comparisons": comparisons,
        "replicas": replicas,
    }
    write_json(output, report)
    if not report["passed"]:
        failed = [key for key, matches in comparisons.items() if not matches]
        raise RuntimeError("Hover determinism gate failed for: " + ", ".join(failed))
    print(f"Hover determinism gate passed; report written to {output}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.transition_steps <= 0:
        raise ValueError("transition_steps must be positive")
    if args.worker_output is not None:
        return run_worker(args)
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
