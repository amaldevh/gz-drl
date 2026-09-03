# ONNX policy inference with ROS 2 SITL

This optional C++ example runs an exported policy through ONNX Runtime while a
GzDRL ROS 2 SITL server advances Gazebo. It uses the trajectory-tracking
processor bundled with the example.

The validated project stack is Ubuntu 24.04, Gazebo Jetty, and ROS 2 Jazzy.
Install GzDRL from a shell in which ROS 2 is sourced so the `rosrl_sitl` target
is built:

```bash
source /opt/ros/jazzy/setup.bash
python -m pip install '.[examples]'
```

Install an ONNX Runtime C/C++ distribution that provides
`onnxruntimeConfig.cmake`, then export an SB3 policy:

> **Model contract:** the bundled processor requires exactly 186 float
> observations and 6 float actions with the preprocessing and scaling in
> `include/trajectory_planning_processor.hh`. GzDRL 1.0.0 does not ship a
> compatible checkpoint or a maintained trainer for that legacy contract.
> The registered trajectory environments use different spaces. Supply a model
> trained against this exact processor; exporting an arbitrary Hover or
> registered trajectory checkpoint will not work.

```bash
python examples/rl/sb3_policy_export.py \
  -algo ppo \
  -weights /path/to/best_model.zip \
  -output /tmp/policy.onnx
```

Configure the example with explicit package directories instead of modifying
`PYTHONPATH` or referring to a checkout-local `_lib` directory:

```bash
cmake -S examples/real_time_inference \
  -B examples/real_time_inference/build \
  -DCMAKE_BUILD_TYPE=Release \
  -Dgzdrl_DIR="$(python -c 'import gzdrl; print(gzdrl.get_cmake_path())')" \
  -Donnxruntime_DIR=/absolute/path/to/onnxruntime/lib/cmake/onnxruntime
cmake --build examples/real_time_inference/build --parallel
```

Run the executable with explicit model and output paths:

```bash
examples/real_time_inference/build/gzdrl_onnx_inference \
  --model_path /tmp/policy.onnx \
  --envid 0 \
  --sdf_file "$(python -c 'import gzdrl; print(gzdrl.get_sdf_path("world_simple.sdf"))')" \
  --model_name quadrotor \
  --base_link_name quadrotor/base_link \
  --log_csv /tmp/inference_log.csv \
  --waypoints_yaml /tmp/waypoints.yaml \
  --duration_seconds 240
```

Do not work around a missing ONNX Runtime package by creating arbitrary
directories or copying headers. Point `onnxruntime_DIR` at a complete,
compatible installation instead.
