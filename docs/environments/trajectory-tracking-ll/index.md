---
title: Trajectory tracking LL
summary: End-to-end trajectory tracking with four direct rotor-speed actions.
order: 70
env_id: GazeboPoolTrajectoryTrackingLLEnv-v0
observation: "float32[700]"
action: "float32[4]"
catalog: true
---

# Trajectory tracking LL

**Registered ID:** `GazeboPoolTrajectoryTrackingLLEnv-v0`

The low-level trajectory task gives the policy four normalized rotor-velocity
actions. Its default observation contains 20 history entries of 20 values and
20 future reference entries of 15 values, for 700 total.

| Default | Value |
|---|---|
| World | `world_trajectory_tracking.sdf` |
| State history / future waypoints | 20 / 20 |
| Rotation matrices | enabled |
| Episode steps | 3,000 |
| Physics timestep | 1 ms |
| Physics steps per action | 10 |
| Rotor range | 0–2,300 rad/s |
| Domain randomization | disabled |

`info:state` and `info:trajectory` each contain 13 values. With rotation
matrices disabled, the observation formula becomes $11H + 6F$.

Use {doc}`../trajectory-tracking/index` when comparing modular and end-to-end
control. Policies cannot be moved between the two without changing action and
observation interfaces.
