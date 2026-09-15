---
title: SB3 inference and ONNX export
summary: Evaluate a saved Stable-Baselines3 policy and export a compatible feed-forward actor.
order: 35
tags: [Stable-Baselines3, evaluation, ONNX]
---

# SB3 inference and ONNX export

Source examples: `examples/rl/infer_sb3_policy.py` and
`examples/rl/sb3_policy_export.py`.

These source-checkout scripts use the installed `envs` and `gzdrl` packages.
Install the RL extra plus the example-only plotting and export dependencies:

```bash
python -m pip install '.[examples]'
```

## Evaluate a policy

```{warning}
The current `examples/rl/infer_sb3_policy.py` has a syntax error in its import
line (`import gzdrl.envs as envs as envpool`). Before using this script, replace
that line with `from gzdrl import envs as envpool`. The commands below describe
the evaluator after that correction.
```

Choose one of the IDs returned by `envs.list_all_envs()` and pass the model
created for that exact observation and action space:

```bash
python examples/rl/infer_sb3_policy.py \
  -algo ppo \
  -env_name GazeboPoolHoverEnv-v0 \
  -model_path /absolute/path/to/best_model.zip \
  -vecnormalize_path /absolute/path/to/vecnormalize.pkl \
  -logdir /tmp/gzdrl-evaluation
```

The evaluator uses one Gymnasium environment, requests
`domain_randomization=True`, requests deterministic policy actions, and
collects 100 complete episodes. It writes `eval_dr_rewards.npy` and
`eval_dr_histogram.png` below `-logdir`. PPO, SAC, TD3, DDPG, and A2C model
classes are accepted.

Domain-randomization application is task-specific. In particular, the current
hover reset path does not invoke its otherwise-defined randomization helpers,
so the command above does not randomize hover dynamics merely because the flag
was requested. Check {doc}`/guides/domain-randomization` and the selected task
implementation before interpreting the output.

Pass the matching `VecNormalize` state whenever normalization was used during
training. If `-vecnormalize_path` is omitted or does not name a file, the
script disables observation and reward normalization; it cannot reconstruct
training statistics. `-realtime` only adds best-effort sleep between policy
steps and does not establish a real-time guarantee. The `eval_dr_*` filenames
record the requested configuration, not proof that a particular task applied
randomization. These outputs are not, by themselves, a paper reproduction
protocol.

## Export the deterministic actor

The exporter accepts PPO, SAC, TD3, and DDPG checkpoints:

```bash
python examples/rl/sb3_policy_export.py \
  -algo ppo \
  -weights /absolute/path/to/best_model.zip \
  -output /tmp/policy.onnx
```

It exports the feed-forward actor on CPU at ONNX opset 13, with a fixed batch
dimension of one. The graph takes a vector observation named `observation` and
produces an action named `action`; for SAC it exports the deterministic mean.
A YAML file passed with `-config` may batch several exports, with `algo`,
`weights_path`, and `output_path` fields for each entry.

The exporter does not embed `VecNormalize` statistics or other environment
preprocessing. Apply the same preprocessing used in training, and compare
ONNX outputs numerically with the original SB3 policy before deployment. The
source extraction paths cover the listed feed-forward vector policies; do not
assume compatibility with recurrent, dictionary-observation, or image policies.
The optional C++ consumer is in `examples/real_time_inference`; its
example-local input and output dimensions must also match.
