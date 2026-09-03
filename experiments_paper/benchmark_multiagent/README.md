# GzDRL multi-agent scalability benchmark

> Research artifact: this benchmark supports the paper and is not part of the
> public GzDRL API or user documentation.

The benchmark measures `GazeboPoolMultiAgentFormationLLEnv-v0` while scaling
the number of drones in each logical environment.

Install GzDRL and run the full launcher from the repository root:

```bash
python -m pip install -e .
bash experiments_paper/benchmark_multiagent/run_multiagent.sh
```

It runs exactly five trials. Every trial covers `2, 5, 10, 20` drones with 32
logical environments, batch size 11, 1,000 warmup rounds, and 10,000 measured
receive/send steps. Results are written under
`experiments_paper/benchmark_multiagent/results` unless `RESULTS_DIR` is set.
The launcher refuses to append to an existing trial file.

An individual point can be run with:

```bash
python -m experiments_paper.benchmark_multiagent.run_ma_benchmark \
  --num-agents 5 --n-envs 32 --warmup 1000 --n-steps 10000 \
  --out-file /tmp/gzdrl_multiagent.csv
```

The current CSV schema is
`n_envs,n_agents,n_steps,fps,avg_cpu,avg_gpu,avg_ram`. FPS is computed from
logical environment transitions; it is not multiplied by drone count.
