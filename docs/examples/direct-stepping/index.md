---
title: Synchronous stepping
summary: Advance one Gazebo world explicitly and inspect preserved state.
order: 10
tags: [beginner, DRLServer]
---

# Synchronous stepping

Source example: `examples/synchronous_simulation/synchronous_simulation.py`.

The script constructs a `gzdrl.DRLServer`, warms it up, and measures repeated
`run_N(1)` calls. It demonstrates the defining direct-server property: state is
preserved between calls, and simulation advances only when the application
requests it.

Run from an environment where the package is installed:

```bash
python examples/synchronous_simulation/synchronous_simulation.py
```

The printed frequency is a local diagnostic, not a benchmark result. It depends
on the machine, build flags, loaded SDF, rendering settings, and background
load.
