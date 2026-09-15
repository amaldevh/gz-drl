---
title: Controller tuning
summary: Exercise and tune geometric, PID, and sliding-mode UAV controllers.
order: 40
tags: [controllers, UAV]
---

# Controller tuning

The `examples/controller_tuning` directory contains separate scripts for the
geometric controller, Python PID controller, and sliding-mode controller. Each
uses the installed `gzdrl` binding and a bundled Gazebo world.

Run one controller at a time:

```bash
python -m pip install '.[examples]'
python examples/controller_tuning/tune_geometric_controller.py
python examples/controller_tuning/tune_python_pid.py
python examples/controller_tuning/tune_smc_controller.py
```

Treat gains as model-specific. Confirm mass, inertia, gravity convention,
control rate, link name, and actuator mapping before carrying gains into a new
SDF or randomized task.
