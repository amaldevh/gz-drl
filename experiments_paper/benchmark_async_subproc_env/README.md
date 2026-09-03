# Vectorization backend benchmark

> Research artifact: this benchmark supports the paper and is not part of the
> public GzDRL API or user documentation.

This experiment compares four ways to step the Python hover task:

- GzDRL's native `AsyncDRLServerPool`;
- one direct `DRLServer` environment per thread;
- one direct environment per spawned process;
- serial stepping through the local `DummyVecEnv` implementation.

Install GzDRL (`python -m pip install -e .`) and run the complete sweep from the
repository root:

```bash
bash experiments_paper/benchmark_async_subproc_env/run.sh
```

The launcher runs five trials over `1, 4, 8, 16, 32, 64, 128` environments.
Each point uses 1,000 warmup steps and 10,000 measured vector steps. Results go
to `experiments_paper/benchmark_async_subproc_env/results` unless `RESULTS_DIR`
is set. The launcher refuses to append to an existing trial file; select a new
results directory before rerunning.

Run an individual native-pool point with:

```bash
python -m experiments_paper.benchmark_async_subproc_env.run_benchmark_async_drl \
  --n-envs 8 --warmup 1000 --n-steps 10000 \
  --out-file /tmp/asyncdrl.csv
```

Each current CSV uses
`n_envs,n_steps,fps,avg_cpu,avg_gpu,avg_ram`.
