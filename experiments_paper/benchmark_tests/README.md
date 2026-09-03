# Gazebo-EnvPool parameter sweep

> Research artifact: this hardware-tuning benchmark is not part of the public
> GzDRL API or user documentation.

Edit `sweep_config.yaml` to choose the environment, environment-count,
batch-size, worker-thread, and affinity ranges. With GzDRL installed, invoke
the benchmark script directly from this directory:

```bash
cd experiments_paper/benchmark_tests
python gazebo_envpool_bm.py -sweep -sweep_config sweep_config.yaml
```

The sweep writes `sweep_benchmark_results.pkl` and
`sweep_benchmark_results.csv` in the current directory. Generate the diagnostic
plots with:

```bash
python visualize_bm_results.py
```

For one point instead of a sweep:

```bash
python gazebo_envpool_bm.py \
  -env_name GazeboPoolHoverEnv-v0 -n_envs 32 -batch_size 16 \
  -n_steps 500 -num_threads 0
```
