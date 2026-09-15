---
title: Downwash interaction
summary: Step two geometrically controlled quadrotors in the packaged downwash world.
order: 55
tags: [multi-UAV, downwash, direct server]
---

# Downwash interaction

Source example: `examples/downwash/downwash_sim.py`.

Run this source-checkout example against an installed GzDRL 1.0.0 package. It
uses `import gzdrl` and the packaged `world_downwash.sdf`; it does not require a
checkout-relative resource path.

```bash
python -m pip install '.[examples]'
python examples/downwash/downwash_sim.py
```

The script creates two `GeometricController` instances, converts their
collective thrust and body moments to rotor angular velocities, and advances a
shared `DRLServer` one physics step after writing both vehicles' commands. It
first holds two fixed references for 5,000 steps, then moves the second
vehicle's lateral reference for 10,000 steps. At completion, Matplotlib shows
position histories and the two three-dimensional paths.

Use a unique Gazebo partition if other simulations are running, and use a
graphical session (or select a noninteractive Matplotlib backend) for the final
plots. The one-millisecond sleeps are best-effort wall-clock pacing; they do
not guarantee a real-time factor. This is a qualitative interaction and
controller example, not a downwash benchmark or a reproduction of a paper
experiment.
