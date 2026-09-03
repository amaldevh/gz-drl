# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Amal Dev Haridevan

"""Regression checks for deterministic SB3 deployment-policy semantics."""

import numpy as np
import pytest


gym = pytest.importorskip("gymnasium")
sb3 = pytest.importorskip("stable_baselines3")
torch = pytest.importorskip("torch")

from examples.rl.sb3_policy_export import DeploymentPolicy


@pytest.mark.parametrize("algorithm", ["PPO", "SAC", "TD3", "DDPG"])
def test_deployment_policy_matches_deterministic_predict(algorithm):
    env = gym.make("Pendulum-v1")
    algo_class = getattr(sb3, algorithm)
    kwargs = {"n_steps": 8, "batch_size": 8} if algorithm == "PPO" else {}
    model = algo_class("MlpPolicy", env, seed=7, device="cpu", **kwargs)
    observation, _ = env.reset(seed=11)

    expected, _ = model.predict(observation, deterministic=True)
    deployment = DeploymentPolicy(model, algorithm).eval()
    with torch.no_grad():
        actual = deployment(
            torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
        ).cpu().numpy()[0]

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    env.close()
