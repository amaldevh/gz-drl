# Determinism and reproducibility

GzDRL's synchronous state-transition path removes middleware scheduling from
the action → physics → state sequence. The paper evaluates two related but
separate scopes.

## Transition scope

Repeated trials compared direct GzDRL stepping with a ROS 2–Gazebo topic-based
baseline under matched initial conditions. The reported determinism applies to
**state observations** captured from the controlled simulation update.
Rendered cameras, depth images, and LiDAR involve Gazebo's rendering and sensor
pipelines and are not covered by that guarantee.

## Training scope

The training-reproducibility experiment used one hover task, identical hardware,
process-isolated PPO runs, and controlled stochastic inputs. With the same RL
seed and the other inputs fixed, the evaluated runs produced coincident curves
and identical checkpoint hashes. Different RL seeds produced variation.

This demonstrates reproducibility in that controlled experiment. It does not
mean that every algorithm, task, dependency version, device, or stochastic
sensor will be bitwise identical.

## What an experiment must record

For a reproducible run, record at least:

- the `gz-drl` commit and GzDRL version;
- OS, Gazebo, compiler, Python, NumPy, and RL-library versions;
- CPU model and environment/thread/batch counts;
- environment ID and complete generated config;
- policy, environment, trajectory, and evaluation seeds;
- SDF/model revisions and whether sensors are enabled;
- domain-randomization settings and reset cadence;
- normalization state and checkpoint/evaluation procedure.

See {doc}`/research/reproducibility` for the manuscript experiment boundary and
{doc}`/guides/domain-randomization` for runtime model changes.
