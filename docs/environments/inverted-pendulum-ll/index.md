---
title: Inverted pendulum LL
summary: Quadrotor inverted-pendulum task with history and optional privileged observations.
order: 40
env_id: GazeboPoolInvertedPendulumLLEnv-v0
observation: "float32[526]"
action: "float32[4]"
catalog: true
---

# Inverted pendulum LL

**Registered ID:** `GazeboPoolInvertedPendulumLLEnv-v0`

The task balances an inverted pendulum carried by a quadrotor using four
normalized rotor-thrust actions.

With ten state-history entries, ten action-history entries, and privileged
observations enabled, the default observation has 526 values:

- 220 state-history values (UAV and payload features);
- 40 prior-action values; and
- 266 privileged values (state derivatives plus six physical/actuator fields).

`info:policy_obs_dim` and `info:privileged_obs_dim` expose the split. Setting
`privileged_obs=False` removes the privileged 266 values and leaves a
260-element policy observation under the default history sizes.

| Default | Value |
|---|---|
| World | `world_inverted_pendulum.sdf` |
| State / action history | 10 / 10 |
| Episode steps | 1,000 |
| Physics steps per action | 10 |
| Maximum rotor thrust | 12 |
| Domain randomization | disabled |
| Privileged observation | enabled |

Enable `domain_randomization` explicitly to vary physical parameters during
reset. See {doc}`/guides/domain-randomization` for the application cadence.
