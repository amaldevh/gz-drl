---
title: Vectorized hover
summary: Train a hover policy over multiple persistent Gazebo servers.
order: 20
tags: [Stable-Baselines3, AsyncDRLServerPool]
---

# Vectorized hover

Source example: `examples/rl/train_vectorized_hover_policy.py`.

This example uses the Python `HoverEnv`, `AsyncDRLServerPool`, and a
Stable-Baselines3 vector adapter. It is distinct from the native GazeboPool
hover task but demonstrates how model-level server vectorization fits an
existing RL library.

Install the optional RL dependencies before running:

```bash
python -m pip install '.[rl]'
python examples/rl/train_vectorized_hover_policy.py
```

The script uses 15 environments and trains for 500,000 transitions. It writes
checkpoints to `/tmp/hover_vec_models/` and TensorBoard logs to
`/tmp/hover_vec_tensorboard/`. Unlike the other Python hover training script,
this example applies `VecMonitor` without `VecNormalize`. Review its resource
use and output paths before a long run.
