#! /usr/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Amal Dev Haridevan

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd "${script_dir}/../.." && pwd)"
run_dir="$(pwd)"
cd "${repository_root}"

seeds=(1 42)
python -m experiments_paper.benchmark_dr.train_dr -algos PPO -max_steps 10e6 -seeds "${seeds[@]}" -logdir "${run_dir}/nodr_inverted_pendulum" \
    -env_name GazeboPoolInvertedPendulumLLEnv-v0 -n_envs 15
python -m experiments_paper.benchmark_dr.train_dr -algos PPO -max_steps 10e6 -seeds "${seeds[@]}" -logdir "${run_dir}/dr_inverted_pendulum" \
    -env_name GazeboPoolInvertedPendulumLLEnv-v0 -n_envs 15 -domain_randomization
