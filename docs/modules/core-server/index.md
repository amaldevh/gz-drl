---
title: Core server
summary: Direct, single-process control of one Gazebo server and its entities.
order: 10
tags: [C++, Python, synchronous]
---

# Core server

The core is implemented by `DRLServer` and its Gazebo helper system in
`GzDRL/include/rl_server.hh` and `GzDRL/src/rl_server.cc`. The `gzdrl._core`
pybind11 module re-exports its public classes through `gzdrl`.

Responsibilities include:

- constructing and owning a `gz::sim::Server`;
- advancing physics by an explicit number of iterations;
- writing entity-component commands before a step;
- reading link state after updates;
- pose reset, cached-SDF respawn, and physical-parameter mutation;
- optional sensor/contact access; and
- controller and actuator helper operations.

`DRLServerConfig` controls trajectory visualization fields. `RotorParameters`
describes per-rotor physics fields used by the multirotor plugin.

Use this module when an application needs exact ownership of the step loop. See
{doc}`/guides/direct-server` and the {doc}`/reference/python-api`.
