#! /usr/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Amal Dev Haridevan

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd "${script_dir}/../.." && pwd)"
run_dir="$(pwd)"
cd "${repository_root}"

envs=(1 4 8 16)
iters=(1) #(2 3 4 5)
backends=("process" "thread")
for j in "${iters[@]}"; do
	for i in "${envs[@]}"; do
		fastdds shm clean
               if [[ "$i" -lt 256 ]]; then
		python -m experiments_paper.benchmark_realtime_env.run_ros_benchmark --n-envs "$i" \
			--out-file "${run_dir}/run${j}_ros_dummy.csv" --benchmark-dummy --warmup 1000 --n-steps 10000 \
			--backend process
               fi
		# Thread backend hangs ros envs, so not included
		# fastdds shm clean
		# python -m experiments_paper.benchmark_realtime_env.run_ros_benchmark --n-envs $i \
		 	#--out-file run${j}_ros_thread_vec.csv --backend thread --warmup 1000 --n-steps 10000
		#fastdds shm clean
                #if  [[ "$i" -lt 128 ]]; then
		#python -m experiments_paper.benchmark_realtime_env.run_ros_benchmark --n-envs $i \
		#	--out-file run${j}_ros_process_vec.csv --backend process --warmup 1000 --n-steps 10000
		#fi
	done
done
