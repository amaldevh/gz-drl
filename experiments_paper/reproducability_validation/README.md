# Training reproducibility validation

> Research artifact: this controlled study supports the paper and is not part
> of the public GzDRL API or user documentation.

This experiment separates implementation nondeterminism from ordinary PPO seed
variation on `GazeboPoolHoverEnv-v0`:

- `same_seed`: five isolated runs with seed `42`;
- `different_seed`: five isolated runs with seeds `1, 5, 25, 42, 125`.

The complete settings are in `config.json`. Domain randomization is disabled
for this specific determinism study. Environment construction, normalization,
episode horizon, environment count, batch size and thread count stay fixed.

Install GzDRL with its RL extra, then run from this directory:

```bash
python -m pip install -e '.[rl]'
cd experiments_paper/reproducability_validation
python run_experiment.py
```

Before training, the orchestrator compares two fresh eight-environment workers
using hashes of initial observations, a fixed 16-step transition, the initial
policy, one PPO update, and normalization state. A mismatch aborts and is
recorded in `results/determinism_check.json`.

Useful controls include:

```bash
python run_experiment.py --groups same_seed
python run_experiment.py --resume
python run_experiment.py --skip-training
```

Existing run data is not overwritten; use `--resume` or choose another
`--results-dir`. Every checkpoint is evaluated in a fresh environment process
over 20 deterministic episodes using frozen saved normalization statistics.

For a plumbing check only (not a scientific result):

```bash
python run_experiment.py \
  --results-dir /tmp/gzdrl_repro_smoke \
  --groups same_seed --same-seed-repetitions 2 \
  --total-timesteps 16 --checkpoint-interval 8 \
  --num-envs 1 --env-num-threads 1 --env-batch-size 1 \
  --episode-steps 16 --evaluation-episodes 2 \
  --ppo-n-steps 8 --ppo-batch-size 8 \
  --skip-determinism-check
```

Analysis outputs are placed below `results/analysis`, including per-run and
summary CSV files, plots, a LaTeX table, and a configuration/machine manifest.
