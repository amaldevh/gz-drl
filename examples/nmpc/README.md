# NMPC controller example

This optional example tracks a quadrotor trajectory with an acados SQP-RTI
nonlinear model-predictive controller. NMPC is an example integration, not a
built-in GzDRL controller.

## Prerequisites

- a working GzDRL 1.0.0 installation (`python -m pip install .` from the
  repository root);
- acados and its Python template package;
- CasADi, SciPy, Eigen, and the native dependencies used by GzDRL.

Set `ACADOS_SOURCE_DIR` to the acados install prefix containing `include/` and
`lib/`:

```bash
export ACADOS_SOURCE_DIR=/absolute/path/to/acados
```

## Generate and build

Run these commands from the gzdrl repository root:

```bash
python examples/nmpc/python/generate_acados_code.py

cmake -S examples/nmpc/cpp -B examples/nmpc/cpp/build \
  -DCMAKE_BUILD_TYPE=Release \
  -Dgzdrl_DIR="$(python -c 'import gzdrl; print(gzdrl.get_cmake_path())')"
cmake --build examples/nmpc/cpp/build --parallel
```

The generator writes solver code to `/tmp/c_generated_code_ocp`, matching the
path used by the example CMake project.

## Run and plot

```bash
export LD_LIBRARY_PATH="$ACADOS_SOURCE_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
examples/nmpc/cpp/build/nmpc_test \
  "$(python -c 'import gzdrl; print(gzdrl.get_sdf_path("world_simple.sdf"))')"
python examples/nmpc/python/plot_results.py \
  --csv examples/nmpc/cpp/build/trajectory_results.csv
```

The executable writes `trajectory_results.csv` and reports its tracking
metrics. The example has additional acados-specific requirements and is not
part of the base package smoke test.
