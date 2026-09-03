# PyBullet throughput benchmark

> Research artifact: this directory reproduces a paper experiment. It is not
> part of the public GzDRL Python API or the user documentation.

These scripts implement GzDRL-equivalent hover and multi-agent formation tasks
with `gym-pybullet-drones`. They are custom benchmark tasks, not the package's
stock hover environments.

Use a Python environment containing `gym-pybullet-drones`, Gymnasium, NumPy and
the editable GzDRL checkout:

```bash
python -m pip install -e .
python -m pip install gym-pybullet-drones
```

Run from the gzdrl repository root. No manual import-path changes are
needed.

## Smoke tests

```bash
python -m experiments_paper.benchmark_pybullet.benchmark_hover \
  --n-envs 4 --n-steps 100 --warmup 20 --physics-dt 0.001 \
  --seed 0 --out-file /tmp/pybullet_hover.csv

python -m experiments_paper.benchmark_pybullet.benchmark_multiagent \
  --n-drones 2 --n-envs 32 --n-steps 100 --warmup 20 \
  --physics-dt 0.001 --seed 0 \
  --out-file /tmp/pybullet_multiagent.csv
```

The default multiprocessing context is `spawn`; `--context forkserver` and
`--context fork` are also available. Each logical multi-agent environment uses
one PyBullet world, so its drones can physically interact.

## Paper sweeps

```bash
bash experiments_paper/benchmark_pybullet/run_hover_sweep.sh
bash experiments_paper/benchmark_pybullet/run_multiagent_sweep.sh
```

The launchers run exactly five trials with seeds `0, 1, 2, 3, 4`:

- hover: `1, 4, 8, 16, 32, 64, 128` environments, 1,000 warmup steps and
  10,000 measured vector steps;
- multi-agent: 32 environments with `2, 5, 10, 20` drones, 1,000 warmup steps
  and 10,000 measured vector steps.

Override `RESULTS_DIR`, `PHYSICS_DT`, `N_STEPS`, `WARMUP`, `CONTEXT`, or
`N_ENVS` (for the multi-agent sweep). Defaults are written below this directory
in `results/hover` and `results/multiagent`.

Hover CSV files contain
`n_envs,n_steps,fps,avg_cpu,avg_gpu`. Multi-agent files contain
`n_drones,n_steps,fps,avg_cpu,avg_gpu`.

## Interpretation

Normalized actions are mapped to the same 0--2300 rad/s rotor target; the
wrapper converts rad/s to the RPM interface expected by `BaseAviary`. The task
logic and FPS definition are aligned with the GzDRL benchmark, but vehicle
dynamics and contact physics are simulator-specific.
