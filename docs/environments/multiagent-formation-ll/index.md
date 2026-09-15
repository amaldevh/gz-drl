---
title: Multi-agent formation LL
summary: Joint low-level control of interacting quadrotors in one Gazebo world.
order: 50
env_id: GazeboPoolMultiAgentFormationLLEnv-v0
observation: "float32[2, 34]"
action: "float32[2, 4]"
catalog: true
---

# Multi-agent formation LL

**Registered ID:** `GazeboPoolMultiAgentFormationLLEnv-v0`

One environment contains multiple interacting quadrotors. For each agent, the
observation contains 18 self features and 16 features for every other agent:

$$d_{obs} = 18 + 16(N_A - 1).$$

The action has four normalized rotor commands per agent. At the default two
agents, observation and action shapes are `(2, 34)` and `(2, 4)`.

| Default | Value |
|---|---|
| Agents | 2 |
| World naming | `world_formation<N>.sdf` |
| Episode steps | 20,000 |
| Physics steps per action | 1 |
| Maximum rotor velocity | 2,300 rad/s |
| Domain randomization | enabled |

All agents are commanded before the shared Gazebo world advances. The task
returns a joint observation, team reward, and shared termination state. Bundled
world files determine which `num_agents` values can actually be constructed;
check resource availability before changing the default.
