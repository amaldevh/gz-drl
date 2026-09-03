#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Amal Dev Haridevan

set -euo pipefail

# Match the multi-agent scalability experiment:
# 32 environments, N agents in {2, 5, 10, 20}, two trials.
agent_counts=(2 5 10 20)
trials=(1 2 3 4 5)

repo_root="${OMNIDRONES_ROOT:-/workspace/OmniDrones}"
results_dir="${RESULTS_DIR:-${repo_root}/benchmark_results/omnidrones_multiagent}"
n_envs="${N_ENVS:-32}"
n_steps="${N_STEPS:-10000}"
warmup="${WARMUP:-1000}"
physics_dt="${PHYSICS_DT:-0.001}"

mkdir -p "${results_dir}"

for trial in "${trials[@]}"; do
    out_file="${results_dir}/omnidrones_multiagent_fps_${trial}.csv"

    # Avoid silently appending a second sweep to an old trial file.
    rm -f "${out_file}"

    for n_drones in "${agent_counts[@]}"; do
        echo
        echo "Trial ${trial}: ${n_drones} drones, ${n_envs} environments"

        /isaac-sim/python.sh \
            "${repo_root}/scripts/run_omnidrones_multiagent_benchmark.py" \
            --repo-root "${repo_root}" \
            --n-drones "${n_drones}" \
            --n-envs "${n_envs}" \
            --n-steps "${n_steps}" \
            --warmup "${warmup}" \
            --physics-dt "${physics_dt}" \
            --seed "$((trial - 1))" \
            --out-file "${out_file}"
    done
done
