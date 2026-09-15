---
title: Controllers
summary: Common UAV controller interface and the bundled geometric, sliding-mode, and PID controllers.
order: 40
tags: [C++, Python, UAV]
---

# Controllers

`UAVController` defines force, moment, and combined thrust/moment operations.
The native bindings expose these implementations:

- `GeometricController`;
- `SlidingModeController`; and
- `PIDController`.

Controller parameters and tuned-gain maps are also bound. Task implementations
may install a controller on a model/link or call it directly while converting a
high-level policy action to rotor commands.

Controllers consume the 13-element state and state-derivative convention in
{doc}`/concepts/state-frames-units`. Gains, mass, inertia, and gravity must be
consistent with the model. When domain randomization changes physics, decide
explicitly whether controller parameters should change as well.
