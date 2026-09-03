# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Amal Dev Haridevan

"""Shared configuration and determinism helpers for training reproducibility."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import socket
from pathlib import Path
from typing import Any, Mapping

import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = EXPERIMENT_ROOT / "config.json"
DEFAULT_RESULTS = EXPERIMENT_ROOT / "results"


def load_config(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    required = {
        "environment",
        "total_timesteps",
        "same_seed",
        "same_seed_repetitions",
        "different_seeds",
        "environment_seed",
        "evaluation_seed",
        "gz_partition_offset",
        "evaluation_partition_id",
        "num_envs",
        "env_num_threads",
        "env_batch_size",
        "episode_steps",
        "domain_randomization",
        "normalize_observations",
        "checkpoint_interval",
        "evaluation_episodes",
        "device",
        "torch_threads",
        "ppo",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Configuration is missing keys: {missing}")
    positive_integer_keys = (
        "total_timesteps",
        "same_seed_repetitions",
        "num_envs",
        "env_num_threads",
        "env_batch_size",
        "episode_steps",
        "checkpoint_interval",
        "evaluation_episodes",
        "torch_threads",
    )
    for key in positive_integer_keys:
        if int(config[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    if int(config["env_batch_size"]) > int(config["num_envs"]):
        raise ValueError("env_batch_size cannot exceed num_envs")
    if int(config["checkpoint_interval"]) % int(config["num_envs"]) != 0:
        raise ValueError("checkpoint_interval must be divisible by num_envs")
    if str(config["device"]) not in {"cpu", "cuda"}:
        raise ValueError("device must be either 'cpu' or 'cuda'")
    different_seeds = [int(seed) for seed in config["different_seeds"]]
    if not different_seeds:
        raise ValueError("different_seeds cannot be empty")
    ppo = config["ppo"]
    for key in (
        "learning_rate",
        "n_steps",
        "batch_size",
        "n_epochs",
        "gamma",
        "gae_lambda",
        "clip_range",
        "ent_coef",
        "vf_coef",
        "max_grad_norm",
        "policy_net_arch",
    ):
        if key not in ppo:
            raise ValueError(f"ppo configuration is missing {key}")
    rollout_size = int(config["num_envs"]) * int(ppo["n_steps"])
    if rollout_size % int(ppo["batch_size"]) != 0:
        raise ValueError("PPO batch_size must divide num_envs * ppo.n_steps exactly")
    if bool(config["domain_randomization"]):
        raise ValueError("This experiment requires domain_randomization=false")


def write_json(path: Path, contents: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(contents, indent=2) + "\n", encoding="utf-8")


def configuration_hash(config: Mapping[str, Any]) -> str:
    serialized = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def subprocess_environment(
    seed: int, torch_threads: int, device: str
) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONHASHSEED": str(seed),
            "OMP_NUM_THREADS": str(torch_threads),
            "MKL_NUM_THREADS": str(torch_threads),
            "OPENBLAS_NUM_THREADS": str(torch_threads),
            "NUMEXPR_NUM_THREADS": str(torch_threads),
            "PYTHONFAULTHANDLER": "1",
        }
    )
    if device == "cpu":
        environment["CUDA_VISIBLE_DEVICES"] = ""
    return environment


def seed_everything(seed: int, torch_threads: int) -> None:
    """Set all Python-side stochastic sources before constructing policy/env."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(torch_threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def runtime_metadata(config: Mapping[str, Any]) -> dict[str, Any]:
    import stable_baselines3
    import torch

    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "stable_baselines3": stable_baselines3.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "configuration_sha256": configuration_hash(config),
    }


def policy_parameter_hash(policy) -> str:
    """Hash numerical policy state rather than ZIP metadata."""
    digest = hashlib.sha256()
    for name, tensor in sorted(policy.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def ppo_constructor_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the pinned PPO settings shared by training and its gate."""
    ppo = config["ppo"]
    target_kl = ppo.get("target_kl")
    return {
        "learning_rate": float(ppo["learning_rate"]),
        "n_steps": int(ppo["n_steps"]),
        "batch_size": int(ppo["batch_size"]),
        "n_epochs": int(ppo["n_epochs"]),
        "gamma": float(ppo["gamma"]),
        "gae_lambda": float(ppo["gae_lambda"]),
        "clip_range": float(ppo["clip_range"]),
        "ent_coef": float(ppo["ent_coef"]),
        "vf_coef": float(ppo["vf_coef"]),
        "max_grad_norm": float(ppo["max_grad_norm"]),
        "target_kl": float(target_kl) if target_kl is not None else None,
        "policy_kwargs": {
            "net_arch": [int(width) for width in ppo["policy_net_arch"]],
            # Preserve compatibility with result configs created before this
            # field was explicit; SB3's historical default is 0.0.
            "log_std_init": float(ppo.get("log_std_init", 0.0)),
        },
    }


def environment_kwargs(
    config: Mapping[str, Any],
    evaluation: bool,
    partition_offset: int | None = None,
    evaluation_partition_id: int | None = None,
) -> dict[str, Any]:
    """Return fixed construction arguments for the C++ hover environment.

    Hover keys each RNG stream by the configured seed plus EnvPool's stable
    logical environment ID.  Gazebo partition offsets are intentionally
    independent of reset randomness.
    """
    num_envs = 1 if evaluation else int(config["num_envs"])
    return {
        "num_envs": num_envs,
        "batch_size": 1 if evaluation else int(config["env_batch_size"]),
        "num_threads": 1 if evaluation else int(config["env_num_threads"]),
        "seed": int(
            config["evaluation_seed"] if evaluation else config["environment_seed"]
        ),
        "gz_partition_offset": int(
            config["gz_partition_offset"]
            if partition_offset is None
            else partition_offset
        ),
        "test_env": evaluation,
        "test_envid": int(
            config["evaluation_partition_id"]
            if evaluation_partition_id is None
            else evaluation_partition_id
        ),
        "domain_randomization": False,
        "max_steps_per_episode": int(config["episode_steps"]),
        "max_episode_steps": int(config["episode_steps"]),
    }
