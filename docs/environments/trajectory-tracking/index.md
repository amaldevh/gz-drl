---
title: Trajectory tracking
summary: Modular trajectory tracking with three-dimensional high-level actions.
order: 60
env_id: GazeboPoolTrajectoryTrackingEnv-v0
observation: "float32[680]"
action: "float32[3]"
catalog: true
---

# Trajectory tracking

**Registered ID:** `GazeboPoolTrajectoryTrackingEnv-v0`

This task tracks a reference trajectory using a three-dimensional policy
action and an integrated geometric controller. With rotation matrices enabled, the
default observation concatenates 20 history entries of 19 values and 20 future
reference entries of 15 values, for 680 total.

| Default | Value |
|---|---|
| World | `world_trajectory_tracking.sdf` |
| State history / future waypoints | 20 / 20 |
| Rotation matrices | enabled |
| Episode steps | 3,000 |
| Physics timestep | 1 ms |
| Physics steps per action | 10 |
| Domain randomization | disabled |
| Action scale / bias | `[9, 9, 15]` / `[0, 0, 0]` |

The environment also returns `info:state` and `info:trajectory`, each with 13
values. Disabling rotation matrices changes the observation formula to
$10H + 6F$ for history length $H$ and future-waypoint count $F$.
