# Aerial Gym throughput benchmark

> Research artifact: this directory reproduces a paper experiment. It is not
> part of the public GzDRL Python API or the user documentation.

These scripts implement GzDRL-equivalent hover and multi-agent formation tasks
on Aerial Gym. They use Aerial Gym's `base_quadrotor` asset and NVIDIA Isaac
Gym Preview 4. A CUDA-capable Aerial Gym installation is therefore required in
addition to an installed GzDRL checkout.

Run commands from the gzdrl repository root so the research modules are
available. Install the library first with `python -m pip install -e .`; no
manual import-path changes are needed.

## Smoke tests

```bash
python -m experiments_paper.benchmark_aerial_gym.benchmark_hover \
  --n-envs 4 --n-steps 100 --warmup 20 --physics-dt 0.001 \
  --seed 0 --out-file /tmp/aerialgym_hover.csv

python -m experiments_paper.benchmark_aerial_gym.benchmark_multiagent \
  --n-drones 2 --n-envs 32 --n-steps 100 --warmup 20 \
  --physics-dt 0.001 --seed 0 \
  --out-file /tmp/aerialgym_multiagent.csv
```

Both commands apply zero normalized motor actions while timing. Scene creation
and warmup are excluded. FPS is logical environment transitions per second;
the multi-agent value is not multiplied by the number of drones.

## Paper sweeps

```bash
bash experiments_paper/benchmark_aerial_gym/run_hover_sweep.sh
bash experiments_paper/benchmark_aerial_gym/run_multiagent_sweep.sh
```

The launchers run exactly five trials with seeds `0, 1, 2, 3, 4`:

- hover: `1, 4, 8, 16, 32, 64, 128` environments, 1,000 warmup steps and
  10,000 measured vector steps;
- multi-agent: 32 environments with `2, 5, 10, 20` drones, 1,000 warmup steps
  and 10,000 measured vector steps.

Override `RESULTS_DIR`, `PHYSICS_DT`, `N_STEPS`, `WARMUP`, or `N_ENVS` (for
the multi-agent sweep) as environment variables. Defaults are written below
this directory in `results/hover` and `results/multiagent`.

Hover CSV files contain
`n_envs,n_steps,fps,avg_cpu,avg_gpu`. Multi-agent files contain
`n_drones,n_steps,fps,avg_cpu,avg_gpu`.

## Interpretation

The scripts align action limits, observation layout, reset distributions,
reward and termination logic with the corresponding GzDRL benchmark as closely
as practical. They do not claim identical vehicle dynamics: Aerial Gym and
Gazebo use different assets, physics engines and aerodynamic models.
