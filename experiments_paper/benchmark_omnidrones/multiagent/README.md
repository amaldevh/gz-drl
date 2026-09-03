# OmniDrones multi-agent throughput benchmark

> Research artifact: this adapter supports the paper comparison. It is not a
> GzDRL library module or public user guide.

This bundle ports the low-level GzDRL formation task to OmniDrones. Each
logical environment contains a physically interacting swarm; FPS is not
multiplied by the number of drones.

Copy the supplied files into an external OmniDrones checkout:

```bash
export OMNIDRONES_ROOT=/path/to/OmniDrones
cp omni_drones/envs/single/gazebo_multiagent_formation_ll.py \
  "$OMNIDRONES_ROOT/omni_drones/envs/single/"
cp cfg/task/GazeboMultiAgentFormationLL.yaml "$OMNIDRONES_ROOT/cfg/task/"
cp scripts/run_omnidrones_multiagent_benchmark.py \
  scripts/run_omnidrones_multiagent_benchmark.sh \
  "$OMNIDRONES_ROOT/scripts/"
```

Run one point with Isaac Sim's Python launcher:

```bash
cd "$OMNIDRONES_ROOT"
/isaac-sim/python.sh scripts/run_omnidrones_multiagent_benchmark.py \
  --repo-root "$OMNIDRONES_ROOT" --n-drones 2 --n-envs 32 \
  --n-steps 100 --warmup 20 --physics-dt 0.001 --seed 0 \
  --out-file /tmp/omnidrones_multiagent.csv
```

For the complete paper sweep:

```bash
OMNIDRONES_ROOT="$OMNIDRONES_ROOT" \
RESULTS_DIR="$OMNIDRONES_ROOT/benchmark_results/multiagent" \
bash "$OMNIDRONES_ROOT/scripts/run_omnidrones_multiagent_benchmark.sh"
```

The runner performs exactly five trials with seeds `0, 1, 2, 3, 4`. Each
trial covers `2, 5, 10, 20` drones in 32 environments, with 1,000 warmup steps
and 10,000 measured vector steps. The current CSV schema is
`n_drones,n_steps,fps,avg_cpu,avg_gpu,avg_ram`.

The task adapter aligns action/observation dimensions, reward, termination and
formation geometry with the GzDRL benchmark. It is a workload and throughput
comparison, not a claim of identical Gazebo and PhysX dynamics.
