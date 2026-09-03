#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Amal Dev Haridevan

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/results/hover}"
PHYSICS_DT="${PHYSICS_DT:-0.001}"
N_STEPS="${N_STEPS:-10000}"
WARMUP="${WARMUP:-1000}"

env_counts=(1 4 8 16 32 64 128)
trials=(1 2 3 4 5)

mkdir -p "${RESULTS_DIR}"
cd "${REPOSITORY_ROOT}"

for trial in "${trials[@]}"; do
    out="${RESULTS_DIR}/run${trial}_aerialgym.csv"
    rm -f "${out}"

    for n_envs in "${env_counts[@]}"; do
        python -m experiments_paper.benchmark_aerial_gym.benchmark_hover \
            --n-envs "${n_envs}" \
            --n-steps "${N_STEPS}" \
            --warmup "${WARMUP}" \
            --physics-dt "${PHYSICS_DT}" \
            --seed "$((trial - 1))" \
            --out-file "${out}"
    done
done
