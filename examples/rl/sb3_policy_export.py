# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Amal Dev Haridevan

"""Stable Baseline3 example"""
import os
import torch.nn as nn
import torch
import stable_baselines3 as sb3
import argparse 


algos = ["PPO", "SAC",  "TD3",  "DDPG"]
algos = algos + [algo.lower() for algo in algos]

class DeploymentPolicy(nn.Module):
    def __init__(self, sb3_model, algo_type):
        super().__init__()
        self.algo_type = algo_type.upper()
        
        if hasattr(sb3_model.policy, 'actor'):
            # For off-policy algorithms (SAC, TD3, DDPG)
            if self.algo_type == 'SAC':
                # SAC: Extract only the latent network and mu (mean) for deterministic output
                # This avoids RandomNormalLike and log_std sampling operations
                self.features_extractor = sb3_model.policy.actor.features_extractor
                self.latent_pi = sb3_model.policy.actor.latent_pi
                self.mu = sb3_model.policy.actor.mu
            else:
                # TD3/DDPG actors already include their single tanh squash.
                self.actor = sb3_model.policy.actor
        else:
            # For on-policy algorithms (PPO, A2C)
            self.features_extractor = sb3_model.policy.features_extractor
            self.mlp_extractor = sb3_model.policy.mlp_extractor
            self.action_net = sb3_model.policy.action_net

        self.register_buffer("action_scale", torch.tensor(
            (sb3_model.action_space.high - sb3_model.action_space.low) / 2.0,
            dtype=torch.float32
        ))
        self.register_buffer("action_bias", torch.tensor(
            (sb3_model.action_space.high + sb3_model.action_space.low) / 2.0,
            dtype=torch.float32
        ))
        self.register_buffer(
            "action_low", torch.as_tensor(sb3_model.action_space.low, dtype=torch.float32)
        )
        self.register_buffer(
            "action_high", torch.as_tensor(sb3_model.action_space.high, dtype=torch.float32)
        )
        
    def forward(self, obs):
        if hasattr(self, 'action_net'):
            # PPO uses the unsquashed deterministic Gaussian mean, then clips
            # it to the environment action bounds in BasePolicy.predict().
            features = self.features_extractor(obs)
            latent_pi = self.mlp_extractor.forward_actor(features)
            actions = self.action_net(latent_pi)
            return torch.clamp(actions, min=self.action_low, max=self.action_high)
        elif self.algo_type == 'SAC':
            # SAC's deterministic normalized action is tanh(mu).
            features = self.features_extractor(obs)
            latent_pi = self.latent_pi(features)
            normalized_actions = torch.tanh(self.mu(latent_pi))
        else:
            # TD3/DDPG Actor.forward() already returns a tanh-squashed action.
            normalized_actions = self.actor(obs)
        return normalized_actions * self.action_scale + self.action_bias

def parse_args_config():
    """Tries to parse args expecting a config file first"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", type=str, help="Path to config file")
    args, unknown = parser.parse_known_args()
    if args.config:
        # Load config file and parse arguments from it
        import yaml
        with open(args.config, 'r') as f:
            config_args = yaml.safe_load(f)
        return config_args
    return None

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-algo", type=str, required=True, 
                        choices=algos, help= "RL algorithm corresponding to SB3 model")
    parser.add_argument("-output", type=str, required=True, help="Output name for onnx model")
    parser.add_argument("-weights", type=str, required=True, help="Model weights to load (sb3 saves as .zip)")
    return parser.parse_args()


def export_model(weight_path, output_path, algo):
    """Exports SB3 model to ONNX format"""
    Algo = getattr(sb3, algo.upper())
    model = Algo.load(weight_path)
    deploy_model = DeploymentPolicy(model, algo)
    deploy_model.eval()
    deploy_model.to("cpu")
    input_shape = (1, model.observation_space.shape[0]) 
    print("Model summary:")
    print("Input shape:", input_shape)
    print("Output shape:", model.action_space.shape)
    example_input = torch.randn(*input_shape)
    # Export to ONNX (use dynamo=False for better compatibility with SB3)
    if output_path.endswith(".onnx"):
        output_path = output_path[:-5]  # Remove .onnx if provided
    torch.onnx.export(
        deploy_model,
        example_input,
        output_path + ".onnx",
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        dynamo=False,
        input_names=['observation'],
        output_names=['action'],
        # dynamic_axes={
        #     'observation': {0: 'batch_size'},
        #     'action': {0: 'batch_size'}
        # }
    )
    output_path, ext = os.path.splitext(output_path)
    print(f"Model exported successfully to {output_path}.onnx")
    
if __name__ == "__main__":
    """ NOTE make sure to flatten the outputs
    NOTE, example_inputs should always have first dim unsqueezed (i.e a batch dim)
    """
    # Create deployment model
    config_args = parse_args_config()
    if config_args is not None:
        for key, value in config_args.items():
            algo = value["algo"]
            weight_path = value["weights_path"]
            output_path = value["output_path"]
            export_model(weight_path, output_path, algo)
    else:
        args = parse_args()
        export_model(args.weights, args.output, args.algo)
