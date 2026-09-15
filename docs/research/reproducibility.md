# Reproduce an experiment

<span id="reproducibility-protocol"></span>
<span id="domain-randomization-interpretation"></span>

A reproducibility check needs the same simulation settings as well as the same
random seed. This page summarizes the existing experiments and points to the
source harnesses. Use {doc}`/concepts/determinism` for the settings to archive
with a new run.

## Transition synchronization

The manuscript compares direct GzDRL stepping to an asynchronous ROS 2–Gazebo
topic path under matched tasks, timing, and initial states. It measures
action-to-physics latency, observation delay, intervening physics steps, and
cross-run state trajectories.

The conclusion applies to state-based transitions produced by the tested
synchronous path. Rendered sensors are explicitly outside this guarantee.

## Training experiment

The paper's training study uses the GazeboPool hover task and two groups of five
process-isolated PPO runs:

- a same-seed group with controlled RL/environment inputs; and
- a different-seed group using five distinct RL seeds.

All runs use the same hardware, eight parallel environments, a fixed training
budget, periodic checkpoints, and deterministic evaluation on the same
fixed-seed episode set with observation normalization restored and frozen.
Domain randomization is disabled for this experiment.

The same-seed results support reproducible training under those controlled
conditions. They do not imply bitwise reproducibility across different
hardware, dependency builds, algorithms, tasks, or stochastic sensor pipelines.

## Source harnesses

The source keeps focused harnesses under:

- `experiments_paper/synchronization_validation`;
- `experiments_paper/reproducability_validation` (spelling as present on disk);
- `experiments_paper/benchmark_tests`; and
- task/baseline-specific benchmark directories.

Run them from an installed package environment and archive their configs with
results. They are not imported by the library and are excluded from generated
documentation APIs.

## Evaluate randomized dynamics

Record the randomization settings actually applied during reset, including the
sampling ranges and cadence. A requested flag or an output filename does not
establish that a task changed its dynamics. The current application points are
listed in {doc}`/guides/domain-randomization`.
