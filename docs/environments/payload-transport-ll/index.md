---
title: Payload transport LL
summary: End-to-end payload transport with four normalized rotor actions.
order: 30
env_id: GazeboPoolPayloadTransportLLEnv-v0
observation: "float32[26]"
action: "float32[4]"
catalog: true
---

# Payload transport LL

**Registered ID:** `GazeboPoolPayloadTransportLLEnv-v0`

The low-level (LL) variant gives the policy direct four-rotor control. The
26-element default observation uses the same payload and UAV state family as
the modular task, with one four-dimensional action-history entry.

| Default | Value |
|---|---|
| World | `world_payload.sdf` |
| Maximum rotor velocity | 2,246 rad/s |
| Action history | 1 |
| Episode steps | 1,000 |
| Physics steps per action | 10 |
| Domain randomization | enabled |

The observation dimension is $22 + 4h$. Use this task for end-to-end control
comparisons with {doc}`../payload-transport/index`; their actions are not
interchangeable.
