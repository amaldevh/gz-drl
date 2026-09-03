# OmniDrones hover throughput benchmark

> Research artifact: this adapter supports the paper comparison. It is not a
> GzDRL library module or public user guide.

This bundle ports the `GazeboPoolHoverEnv-v0` task interface to OmniDrones for
a throughput comparison. It targets the external Isaac Sim 4.1/OmniDrones
environment used for the recorded experiment.

Copy the supplied task, configuration and runners into an OmniDrones checkout:

```bash
export OMNIDRONES_ROOT=/path/to/OmniDrones
cp omni_drones/envs/single/gazebo_pool_hover.py \
  "$OMNIDRONES_ROOT/omni_drones/envs/single/"
cp cfg/task/GazeboPoolHover.yaml "$OMNIDRONES_ROOT/cfg/task/"
cp scripts/run_omnidrones_benchmark.py \
  scripts/run_omnidrones_benchmark.sh "$OMNIDRONES_ROOT/scripts/"
```

Run a smoke test with the Python launcher supplied by Isaac Sim:

```bash
cd "$OMNIDRONES_ROOT"
/isaac-sim/python.sh scripts/run_omnidrones_benchmark.py \
  --repo-root "$OMNIDRONES_ROOT" --n-envs 4 --n-steps 100 --warmup 20 \
  --physics-dt 0.001 --seed 0 --step-api torchrl \
  --out-file /tmp/omnidrones_hover.csv
```

For the complete paper sweep:

```bash
OMNIDRONES_ROOT="$OMNIDRONES_ROOT" \
RESULTS_DIR="$OMNIDRONES_ROOT/benchmark_results/hover" \
bash "$OMNIDRONES_ROOT/scripts/run_omnidrones_benchmark.sh"
```

The shell runner performs exactly five trials over
`1, 4, 8, 16, 32, 64, 128` environments. Each trial uses seed `0`, 1,000
warmup vector steps and 10,000 measured vector steps. `STEP_API` selects the
public `torchrl` step path (default) or the lower-level `raw` path; use one
setting consistently for a comparison.

The current CSV schema is
`n_envs,n_steps,fps,avg_cpu,avg_gpu,avg_ram`. FPS is logical environment
transitions per second. Matching task logic does not imply identical Gazebo
and PhysX vehicle dynamics.
