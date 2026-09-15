---
title: Hover
summary: Single-quadrotor hover with direct normalized rotor-speed control.
order: 10
env_id: GazeboPoolHoverEnv-v0
observation: "float32[22]"
action: "float32[4]"
catalog: true
---

# Hover

**Registered ID:** `GazeboPoolHoverEnv-v0`

The hover task controls four rotor velocities directly. Its default 22-element
observation contains target-relative position, yaw error, quaternion, linear
velocity, angular velocity, and two previous four-dimensional actions.

| Default | Value |
|---|---|
| World | `world_hover.sdf` |
| Model / base link | `quadrotor` / `quadrotor/base_link` |
| Action history | 2 |
| Episode steps | 20,000 |
| Physics steps per action | 1 |
| Domain randomization | disabled |

Changing `action_history_size` changes the observation dimension according to
$14 + 4h$. Actions are mapped from `[-1, 1]` to the task's rotor-speed range.
Episodes terminate at the horizon or a failure condition such as excessive
attitude, yaw, or position bounds.

This is the task used by the manuscript's controlled training-reproducibility
experiment. That study's claims remain scoped to the recorded task, hardware,
and stochastic inputs.
