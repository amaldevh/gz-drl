# Build a direct control loop

<span id="drive-a-drlserver"></span>
<span id="reset-versus-respawn"></span>
<span id="choosing-a-command-api"></span>
<span id="utility-methods"></span>
<span id="state-access"></span>

`gzdrl.DRLServer` owns one Gazebo world and lets your application decide when
physics advances. Use it to test a controller, inspect vehicle dynamics, or
implement task logic in Python. This guide extends the {doc}`/getting-started/quickstart`
with repeated commands, state access, and resets.

## Command and step

Give the server a unique transport partition, a world file, and the names of
the models whose state it should track:

```python
import numpy as np
import gzdrl

server = gzdrl.DRLServer(
    "experiment-001",
    str(gzdrl.get_sdf_path("world_simple.sdf")),
    ["quadrotor"],
    False,
)
server.run_N(10)
server.update_control_states()

for _ in range(100):
    command = np.full(4, 900.0, dtype=np.float64)
    server.set_rotor_velocity_cmd("quadrotor", "quadrotor/base_link", command)
    server.run_once()
    server.update_control_states()
    state, state_dot = server.control_states["quadrotor"]["quadrotor/base_link"]
```

Each iteration applies the same rotor velocity command and reads the resulting
state. Replace the fixed command with a controller output to close the feedback
loop. Use one owner for stepping: `run_once()` and `run_N()` are not safe to call
concurrently on the same server.

## Choose the command and state format

Direct commands use physical units. Rotor velocity, single-rotor thrust,
collective thrust and body torque or rate, wrench, velocity, Ackermann, and
joint-position methods are available. The loaded model and its plugins must
support the command you choose. See {doc}`/concepts/control-abstractions`.

`control_states[model][link]` contains the controller-oriented state and its
derivative. Call `update_control_states()` before reading it after a step.
`state_info(model)` returns the helper system's cached link-state records in a
different, 19-element layout and does not require that conversion. The two
layouts are tabulated in {doc}`/concepts/state-frames-units`.

## Reset a vehicle

A pose reset takes a position in metres and an orientation expressed as Euler
angles in radians:

```python
server.reset_pos("quadrotor", np.array([0.0, 0.0, 1.0]), np.zeros(3))
server.update_control_states()
```

`reset_pos()` and `respawn_model()` each perform three internal physics steps
before returning. A respawn removes and recreates the model, which is needed
to apply changes to cached SDF properties such as mass and inertia.

```{warning}
Do not call `respawn_model()` with rendered sensors enabled. Recreating the
model can invalidate Gazebo rendering resources. Use `reset_pos()` for a pose
change in a simulation with sensors.
```

For changes to physical parameters, follow {doc}`domain-randomization`.
For parallel execution of these operations, use {doc}`async-pool`.
