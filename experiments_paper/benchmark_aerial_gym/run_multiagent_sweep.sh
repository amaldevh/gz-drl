#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Amal Dev Haridevan

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/results/multiagent}"
PHYSICS_DT="${PHYSICS_DT:-0.001}"
N_ENVS="${N_ENVS:-32}"
N_STEPS="${N_STEPS:-10000}"
WARMUP="${WARMUP:-1000}"

agent_counts=(2 5 10 20)
trials=(1 2 3 4 5)

mkdir -p "${RESULTS_DIR}"
cd "${REPOSITORY_ROOT}"

for trial in "${trials[@]}"; do
    out="${RESULTS_DIR}/aerialgym_multiagent_fps_${trial}.csv"
    rm -f "${out}"

    for n_drones in "${agent_counts[@]}"; do
        python -m experiments_paper.benchmark_aerial_gym.benchmark_multiagent \
            --n-drones "${n_drones}" \
            --n-envs "${N_ENVS}" \
            --n-steps "${N_STEPS}" \
            --warmup "${WARMUP}" \
            --physics-dt "${PHYSICS_DT}" \
            --seed "$((trial - 1))" \
            --out-file "${out}"
    done
done
