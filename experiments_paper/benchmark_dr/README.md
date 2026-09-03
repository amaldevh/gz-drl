# Domain-randomization robustness experiment

> Research artifact: these training and evaluation programs support the paper.
> They are not part of the public GzDRL API or user documentation.

The launcher trains PPO on `GazeboPoolInvertedPendulumLLEnv-v0` under two
conditions: domain randomization disabled and enabled. It uses exactly seeds
`1` and `42`, 15 parallel environments, and `10e6` training steps per run.

Install GzDRL with the reinforcement-learning dependencies and run from the
repository root:

```bash
python -m pip install -e '.[rl]'
bash experiments_paper/benchmark_dr/run.sh
```

The two result roots are `nodr_inverted_pendulum` and
`dr_inverted_pendulum` in the directory from which the launcher is invoked.

`eval_dr.py` always enables domain randomization and uses 1,000 deterministic
evaluation episodes by default. For example:

```bash
python -m experiments_paper.benchmark_dr.eval_dr \
  -algo ppo \
  -env_name GazeboPoolInvertedPendulumLLEnv-v0 \
  -model_path /path/to/best_model.zip \
  -logdir /tmp/gzdrl_dr_eval
```

If training used `VecNormalize`, also pass
`-vecnormalize_path /path/to/vecnormalize.pkl`. Use `-episodes N` only to
override the evaluation count. `-realtime` adds wall-clock pacing for video
capture and is disabled for reported evaluation runs. Evaluation writes
`eval_dr_rewards.npy` and `eval_dr_histogram.png` to the selected log directory.
