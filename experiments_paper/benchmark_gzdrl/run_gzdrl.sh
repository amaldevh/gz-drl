#! /usr/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Amal Dev Haridevan

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd "${script_dir}/../.." && pwd)"
run_dir="$(pwd)"
cd "${repository_root}"

envs=(1 4 8 16 32 64 128)
iters=(1 2 3 4 5)
warmup=1000
total_steps=10000
num_threads=48
for j in "${iters[@]}"; do
        for i in "${envs[@]}"; do
                python -m experiments_paper.benchmark_gzdrl.run_gzdrl_benchmark \
				 --n-envs "$i" --out-file "${run_dir}/run${j}_gzdrl.csv" --warmup="$warmup" --n-steps="$total_steps" \
				 --num-threads="$num_threads"
        done
done
