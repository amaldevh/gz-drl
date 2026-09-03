#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Amal Dev Haridevan

set -euo pipefail

# Match the environment-count sweep and repetitions in the supplied GzDRL
# runner. Override any value through the corresponding environment variable.
env_counts=(1 4 8 16 32 64 128)
runs=(1 2 3 4 5)

warmup="${WARMUP:-1000}"
n_steps="${N_STEPS:-10000}"
physics_dt="${PHYSICS_DT:-0.001}"
substeps="${SUBSTEPS:-1}"
step_api="${STEP_API:-torchrl}"

repo_root="${OMNIDRONES_ROOT:-/workspace/OmniDrones}"
benchmark="${repo_root}/scripts/run_omnidrones_benchmark.py"
results_dir="${RESULTS_DIR:-${repo_root}/benchmark_results}"

mkdir -p "${results_dir}"

for run in "${runs[@]}"; do
    out_file="${results_dir}/run${run}_omnidrones.csv"

    # Do not mix rows produced by an older CSV schema.
    rm -f "${out_file}"

    for n_envs in "${env_counts[@]}"; do

        echo
        echo "Run ${run}, environments ${n_envs}"

        /isaac-sim/python.sh "${benchmark}" \
            --repo-root "${repo_root}" \
            --n-envs "${n_envs}" \
            --n-steps "${n_steps}" \
            --warmup "${warmup}" \
            --physics-dt "${physics_dt}" \
            --substeps "${substeps}" \
            --step-api "${step_api}" \
            --out-file "${out_file}"
    done
done
