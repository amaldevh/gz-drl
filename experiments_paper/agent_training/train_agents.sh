#! /usr/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Amal Dev Haridevan

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd "${script_dir}/../.." && pwd)"
run_dir="$(pwd)"
cd "${repository_root}"

#envs=("GazeboPoolPayloadTransportLLEnv-v0" "GazeboPoolPayloadTransportEnv-v0" "GazeboPoolTrajectoryTrackingLLEnv-v0" "GazeboPoolTrajectoryTrackingEnv-v0")
envs=("GazeboPoolTrajectoryTrackingEnv-v0")
seeds=(1 25 42)
n_envs=15
gz_partition_offset=0
for env in ${envs[@]}; do
	logdir="${run_dir}/$env"
	for seed in ${seeds[@]}; do
		python -m experiments_paper.agent_training.standalone_trainer -algos ppo -max_steps 5e6 \
			-seed $seed -logdir $logdir -env_name $env -n_envs $n_envs  -gz_partition_offset=$gz_partition_offset & 
		gz_partition_offset=$(($gz_partition_offset + 20))
	done
done
echo "Waiting for all tasks to finish"
wait
