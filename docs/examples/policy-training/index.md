---
title: Python hover training
summary: Train PPO using the Python hover environment and Stable-Baselines3.
order: 30
tags: [Stable-Baselines3, PPO]
---

# Train a hover policy

<span id="single-environment-policy-training"></span>

`examples/rl/train_hover_policy.py` connects the Python `HoverEnv` to
Stable-Baselines3's PPO implementation. The script creates 15 monitored
environments, normalizes observations and rewards with `VecNormalize`, and
trains for 500,000 transitions. It is separate from the native GazeboPool hover
task.

From the repository root:

```bash
python -m pip install '.[rl]'
python examples/rl/train_hover_policy.py
```

Training progress appears in the terminal. TensorBoard logs go to
`/tmp/hover_tensorboard/`, and checkpoints and the final policy go to
`/tmp/hover_models/`. Review the environment count and training budget in the
script before starting a long run.

The script saves policy weights but does not save the `VecNormalize` statistics.
To evaluate or resume a normalized policy, save those statistics alongside the
checkpoint, then restore and freeze them during evaluation. Do not expect a
GazeboPool evaluator to accept this policy solely because both tasks are named
hover; compare observation and action definitions first.

For native parallel server calls, continue with
{doc}`/examples/vectorized-hover/index`.
