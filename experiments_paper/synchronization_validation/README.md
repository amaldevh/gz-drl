# Transition synchronization validation

> Research artifact: this experiment supports the paper and is not part of the
> public GzDRL API or user documentation.

The experiment applies one deterministic open-loop wrench sequence from
identical initial conditions to two backends:

- `gzdrl`: synchronous in-process `DRLServer` access;
- `ros2_gazebo`: an instrumented ROS 2 topic bridge around the same server,
  SDF world, state representation and wrench sequence.

The bridge records physics iterations, command receipt/application, latency,
acknowledgement, and the resulting 13-element state. The analyzer reports
action-to-physics latency, observation age, intervening physics updates,
application rate, and repeated-trajectory dispersion.

Install GzDRL. For the ROS backend, source the ROS 2 environment before running
so `rclpy` and `std_msgs` are importable. From the repository root:

```bash
python experiments_paper/synchronization_validation/run_all.py \
  --trials 100 --actions 200
```

Run one backend independently:

```bash
python experiments_paper/synchronization_validation/run_gzdrl.py \
  --trials 100 --actions 200

python experiments_paper/synchronization_validation/run_ros2.py \
  --trials 100 --actions 200
```

Regenerate analysis from existing raw CSV files with:

```bash
python experiments_paper/synchronization_validation/analyze_results.py
```

Outputs default to `experiments_paper/synchronization_validation/results` and
include raw trials, combined/per-trial summaries, figures,
`table_synchronization_validation.tex`, and `analysis_manifest.json`.

Gazebo Transport is not presented as a measured backend by these scripts. A
future runner can emit the same CSV schema under the `gazebo_transport` backend
name and the analyzer will include it automatically.
