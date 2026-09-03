#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Amal Dev Haridevan

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd "${script_dir}/../.." && pwd)"
results_dir="${RESULTS_DIR:-${script_dir}/results}"

trials=(1 2 3 4 5)
drone_counts=(2 5 10 20)

mkdir -p "${results_dir}"
results_dir="$(cd "${results_dir}" && pwd)"

# run_ma_benchmark appends rows, so do not silently mix independent sweeps.
for trial in "${trials[@]}"; do
    output_file="${results_dir}/run${trial}_multiagent.csv"
    if [[ -e "${output_file}" ]]; then
        echo "Refusing to append to existing result: ${output_file}" >&2
        echo "Choose an empty RESULTS_DIR and run again." >&2
        exit 2
    fi
done

cd "${repository_root}"

for trial in "${trials[@]}"; do
    for n_drones in "${drone_counts[@]}"; do
        python -m experiments_paper.benchmark_multiagent.run_ma_benchmark \
            --num-agents "${n_drones}" \
            --n-envs 32 \
            --warmup 1000 \
            --n-steps 10000 \
            --out-file "${results_dir}/run${trial}_multiagent.csv"
    done
done
