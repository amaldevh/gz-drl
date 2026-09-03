# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Amal Dev Haridevan

import argparse 
import pickle
import numpy as np
import torch
from pathlib import Path
import gzdrl.envs as envs 
import stable_baselines3
algo_singlerl = ["ppo", "sac", "td3", "ddpg", "a2c"]
algo_singlerl = algo_singlerl + [algo.upper() for algo in algo_singlerl]

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-algo", required=True, type=str, help="Algorithms to run", choices = algo_singlerl)
    parser.add_argument("-env_name", type=str, required=True, help="Name of the environment", choices=envs.list_all_envs())
    parser.add_argument("-checkpoint", type=str, required=True, help="Checkpoint to load")
    parser.add_argument(
        "-vecnormalize",
        "--vecnormalize",
        type=str,
        default=None,
        help=(
            "Optional VecNormalize statistics file. When omitted, the script "
            "looks beside the checkpoint for best_vecnormalize.pkl or "
            "final_vecnormalize.pkl."
        ),
    )
    # parser.add_argument("-action_scale", type=float, required=True, nargs="+", help="Action scale")
    return parser.parse_args()


def resolve_vecnormalize_path(checkpoint, explicit_path=None):
    """Resolve the normalization state paired with a model checkpoint."""
    if explicit_path is not None:
        path = Path(explicit_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"VecNormalize statistics file does not exist: {path}"
            )
        return path

    checkpoint_path = Path(checkpoint).expanduser()
    checkpoint_stem = checkpoint_path.stem
    if checkpoint_stem == "best_model":
        candidate = checkpoint_path.parent / "best_vecnormalize.pkl"
    elif checkpoint_stem == "final_model":
        candidate = checkpoint_path.parent / "final_vecnormalize.pkl"
    else:
        candidate = checkpoint_path.parent / (
            checkpoint_stem + "_vecnormalize.pkl"
        )
    return candidate if candidate.is_file() else None


class OnnxableActor(torch.nn.Module):
    """Actor-only inference, including optional observation normalization."""

    def __init__(self, policy, action_scale=None, vecnormalize_path=None):
        super().__init__()
        self.policy = policy
        self.normalize_observations = False
        self.obs_epsilon = 0.0
        self.obs_clip = 0.0
        policy_parameter = next(policy.parameters())
        buffer_device = policy_parameter.device
        buffer_dtype = policy_parameter.dtype

        obs_mean = torch.empty(
            0, dtype=buffer_dtype, device=buffer_device
        )
        obs_var = torch.empty(
            0, dtype=buffer_dtype, device=buffer_device
        )
        if vecnormalize_path is not None:
            path = Path(vecnormalize_path).expanduser()
            with path.open("rb") as file:
                vecnormalize = pickle.load(file)

            self.normalize_observations = bool(vecnormalize.norm_obs)
            if self.normalize_observations:
                if isinstance(vecnormalize.obs_rms, dict):
                    raise TypeError(
                        "OnnxableActor only supports array observations; "
                        "the provided VecNormalize file contains dict statistics."
                    )
                obs_mean = torch.as_tensor(
                    np.asarray(vecnormalize.obs_rms.mean),
                    dtype=buffer_dtype,
                    device=buffer_device,
                )
                obs_var = torch.as_tensor(
                    np.asarray(vecnormalize.obs_rms.var),
                    dtype=buffer_dtype,
                    device=buffer_device,
                )
                if obs_mean.shape != obs_var.shape:
                    raise ValueError(
                        "VecNormalize observation mean and variance shapes differ: "
                        f"{tuple(obs_mean.shape)} != {tuple(obs_var.shape)}"
                    )
                if not torch.isfinite(obs_mean).all():
                    raise ValueError("VecNormalize observation mean is not finite.")
                if not torch.isfinite(obs_var).all() or torch.any(obs_var < 0):
                    raise ValueError(
                        "VecNormalize observation variance must be finite and nonnegative."
                    )
                self.obs_epsilon = float(vecnormalize.epsilon)
                self.obs_clip = float(vecnormalize.clip_obs)
                if self.obs_epsilon <= 0 or self.obs_clip <= 0:
                    raise ValueError(
                        "VecNormalize epsilon and clip_obs must both be positive."
                    )

        # Buffers move with the module and are embedded as constants in ONNX.
        self.register_buffer("obs_mean", obs_mean)
        self.register_buffer("obs_var", obs_var)

        action_space = policy.action_space
        self.has_action_bounds = hasattr(action_space, "low") and hasattr(
            action_space, "high"
        )
        self.squash_output = bool(policy.squash_output)
        if self.has_action_bounds:
            action_low = torch.as_tensor(
                action_space.low,
                dtype=buffer_dtype,
                device=buffer_device,
            )
            action_high = torch.as_tensor(
                action_space.high,
                dtype=buffer_dtype,
                device=buffer_device,
            )
        else:
            action_low = torch.empty(
                0, dtype=buffer_dtype, device=buffer_device
            )
            action_high = torch.empty(
                0, dtype=buffer_dtype, device=buffer_device
            )
        self.register_buffer("action_low", action_low)
        self.register_buffer("action_high", action_high)
        if action_scale is not None:
            self.register_buffer("action_scale", torch.from_numpy(action_scale).to(buffer_device))
        else:
            self.register_buffer("action_scale", torch.ones(action_space.shape).to(buffer_device))

    def forward(self, observation):
        if self.normalize_observations:
            observation = (observation - self.obs_mean) / torch.sqrt(
                self.obs_var + self.obs_epsilon
            )
            observation = torch.clamp(
                observation, -self.obs_clip, self.obs_clip
            )

        # Mirror AsymmetricActorCriticPolicy._predict, deterministic.
        # extract_features handles preprocessing/flattening (FlattenExtractor for Box obs).
        features = self.policy.extract_features(observation, self.policy.features_extractor)
        # latent_pi = self.policy.actor_latent_net(features)
        latent_pi = self.policy.mlp_extractor.forward_actor(features)
        distribution = self.policy._get_action_dist_from_latent(latent_pi)
        action = distribution.mode()

        # Mirror BasePolicy.predict's Box action postprocessing. PPO's regular
        # diagonal Gaussian is clipped, not tanh-squashed.
        if self.has_action_bounds:
            if self.squash_output:
                action = self.action_low + 0.5 * (
                    action + 1.0
                ) * (self.action_high - self.action_low)
            else:
                action = torch.maximum(
                    torch.minimum(action, self.action_high), self.action_low
                )
        return action*self.action_scale

def control_loop(env, obs, policy):
    obs = np.asarray(obs, dtype=np.float32)
    device = next(policy.parameters()).device
    obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
    if obs.ndim == 1:
        obs = obs.unsqueeze(0)
    action = policy(obs)
    obs, rew, done, trunc, info = env.step(action.detach().cpu().numpy())
    return obs, rew, done, trunc, info

def plot(states, trajectories):
    import matplotlib.pyplot as plt
    from scipy.spatial.transform import Rotation
    fig, axs = plt.subplots(3,1)
    fig2, axs2 = plt.subplots(3,1)
    rpy_tracked = Rotation.from_quat(states[:,6:10], scalar_first=True).as_euler('xyz')
    rpy_des = Rotation.from_quat(trajectories[:,6:10], scalar_first=True).as_euler('xyz')

    for i in range(3):
        axs[i].plot(trajectories[:,i ], '--')
        axs[i].plot(states[:,i])
        axs2[i].plot(rpy_des[:,i ], '--')
        axs2[i].plot(rpy_tracked[:,i])
    plt.show()

if __name__ == "__main__":
    args = parse()
    algo = getattr(stable_baselines3, args.algo.upper())
    checkpt = args.checkpoint
    model = algo.load(checkpt)
    vecnormalize_path = resolve_vecnormalize_path(
        checkpt, args.vecnormalize
    )
    policy = OnnxableActor(model.policy, None,  vecnormalize_path).eval()
    env = envs.make_gymnasium(args.env_name)
    if vecnormalize_path is None:
        print("No VecNormalize statistics found; using raw observations.")
    else:
        print(f"Loaded VecNormalize statistics from {vecnormalize_path}")
    def eval():
        states = []
        trajectories = []
        obs, info = env.reset()
        states.append(info["state"])
        trajectories.append(info["trajectory"])
        done = False
        with torch.no_grad():
            while not done:
                obs, rew, done, trunc, info = control_loop(
                    env, obs, policy
                )
                states.append(info["state"])
                trajectories.append(info["trajectory"])
                done = done or trunc
        return np.array(states).squeeze(), np.array(trajectories).squeeze()
    st, tr = eval()
    plot(st, tr)
