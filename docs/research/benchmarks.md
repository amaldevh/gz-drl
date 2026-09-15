# Benchmark interpretation

The manuscript evaluates GzDRL on two machines: an AMD Ryzen Threadripper 3960X
workstation and an AMD Ryzen 9 7940HS laptop. All reported throughput tests are
headless and use the task, physics timestep, warm-up, measurement count, and
environment-count sweep described in the paper.

## Evaluated comparisons

The single-agent study compares GazeboPool, `AsyncDRLServerPool`, Python
multiprocessing/threading/serial paths, ROS 2 serial/process vectorization,
PyBullet-Drones, OmniDrones, and Aerial Gym. The multi-agent study compares
GzDRL, PyBullet-Drones, OmniDrones, and Aerial Gym for 2, 5, 10, and 20
interacting UAVs in 32 environments.

For those measured configurations:

- GazeboPool had the highest measured single-agent throughput among evaluated
  frameworks on the workstation.
- On the laptop it was competitive with the evaluated GPU-accelerated
  alternatives.
- GPU simulation became more favorable as per-environment multi-agent workload
  increased; the crossover depended on hardware and agent count.

When comparing a new setup, record compiler flags, CPU topology,
GPU, simulator versions, rendering, task logic, batch size, thread count, and
physics engine: all affect the result.

## Reproducing a comparison

Use the research harnesses under `experiments_paper` only with their recorded
dependency environments. Before comparing results:

1. verify identical action/observation spaces and task logic;
2. fix the physics timestep and policy-to-physics step ratio;
3. run headless and record render/sensor settings;
4. record `num_envs`, `batch_size`, and native worker count;
5. separate warm-up from measured transitions;
6. repeat trials and report uncertainty; and
7. collect utilization at the same operating point as throughput.

Keep the harness configuration and raw measurements with the generated plots
and tables so the comparison can be inspected later.
