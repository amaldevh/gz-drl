---
title: Multi-UAV control
summary: Command multiple interacting quadrotors inside one synchronous world.
order: 50
tags: [multi-agent, geometric control]
---

# Multi-UAV control

Source example: `examples/multiuav_control/multiagent_geometric_control.py`.

The script demonstrates model/link-keyed states and commands for multiple UAVs
within one `DRLServer`. All model commands should be written before advancing
the shared world so their interaction belongs to the same physics interval.

```bash
python -m pip install '.[examples]'
python examples/multiuav_control/multiagent_geometric_control.py
```

For batched multi-agent RL with joint observations and team reward, use
{doc}`/environments/multiagent-formation-ll/index` instead.
