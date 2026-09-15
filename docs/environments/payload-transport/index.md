---
title: Payload transport
summary: Modular payload transport with a three-dimensional residual-force action.
order: 20
env_id: GazeboPoolPayloadTransportEnv-v0
observation: "float32[25]"
action: "float32[3]"
catalog: true
---

# Payload transport

**Registered ID:** `GazeboPoolPayloadTransportEnv-v0`

This modular-control variant maps a three-dimensional normalized policy action
to a world-frame residual force. An integrated controller computes moments and
the server maps thrust/moments to four rotor velocities.

The 25-element default observation combines payload target error, payload-to-UAV
relative position, UAV quaternion/velocity/angular velocity, payload linear and
angular velocity, and one three-dimensional action-history entry.

| Default | Value |
|---|---|
| World | `world_payload.sdf` |
| Action scaling | `[10, 10, 10]` |
| Action history | 1 |
| Episode steps | 1,000 |
| Physics steps per action | 10 |
| Domain randomization | enabled |

The observation dimension is $22 + 3h$. The environment checks task-specific
failure conditions and refreshes physical randomization at its reset cadence.
