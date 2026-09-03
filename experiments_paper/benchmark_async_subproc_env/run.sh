#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Amal Dev Haridevan

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd "${script_dir}/../.." && pwd)"
results_dir="${RESULTS_DIR:-${script_dir}/results}"

env_counts=(1 4 8 16 32 64 128)
trials=(1 2 3 4 5)
backends=(thread process)
warmup=1000
total_steps=10000

mkdir -p "${results_dir}"
results_dir="$(cd "${results_dir}" && pwd)"

# The benchmark programs append rows. Refuse to mix a new sweep with an
# existing result instead of deleting files or terminating unrelated jobs.
for trial in "${trials[@]}"; do
    for suffix in asyncdrl thread_vecenv process_vecenv dummy_vecenv; do
        output_file="${results_dir}/run${trial}_${suffix}.csv"
        if [[ -e "${output_file}" ]]; then
            echo "Refusing to append to existing result: ${output_file}" >&2
            echo "Choose an empty RESULTS_DIR and run again." >&2
            exit 2
        fi
    done
done

cd "${repository_root}"

for trial in "${trials[@]}"; do
    for n_envs in "${env_counts[@]}"; do
        python -m experiments_paper.benchmark_async_subproc_env.run_benchmark_async_drl \
            --n-envs "${n_envs}" \
            --out-file "${results_dir}/run${trial}_asyncdrl.csv" \
            --warmup "${warmup}" \
            --n-steps "${total_steps}"

        for backend in "${backends[@]}"; do
            python -m experiments_paper.benchmark_async_subproc_env.run_benchmark \
                --n-envs "${n_envs}" \
                --out-file "${results_dir}/run${trial}_${backend}_vecenv.csv" \
                --backend "${backend}" \
                --warmup "${warmup}" \
                --n-steps "${total_steps}"
        done

        python -m experiments_paper.benchmark_async_subproc_env.run_benchmark \
            --n-envs "${n_envs}" \
            --out-file "${results_dir}/run${trial}_dummy_vecenv.csv" \
            --benchmark-dummy \
            --backend process \
            --warmup "${warmup}" \
            --n-steps "${total_steps}"
    done
done
