---
title: GazeboPool runtime
summary: EnvPool-derived C++ scheduling, task processing, and zero-copy batches.
order: 30
tags: [EnvPool, C++, vectorization]
---

# GazeboPool runtime

GazeboPool adapts the EnvPool core under `envs/envpool` to GzDRL tasks. The
project-specific layers are:

- `envs/env_specs`: default config, observation spec, and action spec;
- `envs/processors`: action/observation transforms and reward calculation;
- `envs/gazebo_envpool`: task/server stepping implementations; and
- `envs/registration.py`: Python task registry and Gym, Gymnasium, and dm-env
  constructors.

The native binding exports spec/pool pairs, and `envs` discovers and registers
their task IDs. State buffers are C++ owned and exposed to NumPy without a copy
at the Python boundary.

The bundled EnvPool-derived source retains its upstream Apache-2.0 notices. New
GzDRL tasks should be added by following the guide at {doc}`/environments/creating-new-env/index`.
