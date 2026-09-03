# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Amal Dev Haridevan

from evaluate import OnnxableActor
import argparse
import torch.nn as nn
import torch
import argparse
import stable_baselines3 as sb3
import os 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-algo", type=str, required=True, 
         help= "RL algorithm corresponding to SB3 model")
    parser.add_argument("-output", type=str, required=True, help="Output name for onnx model")
    parser.add_argument("-weights", type=str, required=True, help="Model weights to load (sb3 saves as .zip)")
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
    return parser.parse_args()


def export_model(weight_path, output_path, algo, vecnormalize_path=None):
    """Exports SB3 model to ONNX format"""
    Algo = getattr(sb3, algo.upper())
    model = Algo.load(weight_path)
    deploy_model = OnnxableActor(model.policy, None,  vecnormalize_path).eval()
    deploy_model.to("cpu")
    input_shape = (1, model.observation_space.shape[0]) 
    print("Model summary:")
    print("Input shape:", input_shape)
    print("Output shape:", model.action_space.shape)
    example_input = torch.randn(*input_shape)
    if output_path.endswith(".onnx"):
        output_path = output_path[:-5] 
    torch.onnx.export(
        deploy_model,
        example_input,
        output_path + ".onnx",
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['observation'],
        output_names=['action'],
    )
    output_path, ext = os.path.splitext(output_path)
    print(f"Model exported successfully to {output_path}.onnx")

if __name__ == "__main__":
    args =parse_args()
    export_model(args.weights, args.output, args.algo, args.vecnormalize)