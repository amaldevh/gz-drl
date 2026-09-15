# Run your first simulation

<span id="quickstart"></span>

This example loads a bundled quadrotor world, reads the vehicle state, applies
a rotor command, and advances physics once. It uses `DRLServer`, the interface
underneath GzDRL's RL environments. Complete {doc}`installation` and
{doc}`verification` first.

## Load the world and read the state

Save this code in `quickstart.py` and run it with the Python environment that
contains GzDRL:

```python
import numpy as np
import gzdrl

server = gzdrl.DRLServer(
    "quickstart",  # Gazebo transport partition for this simulation
    str(gzdrl.get_sdf_path("world_simple.sdf")),
    ["quadrotor"],
    False,  # rendered sensors disabled
)

server.run_N(10)
server.update_control_states()
state, state_dot = server.control_states["quadrotor"]["quadrotor/base_link"]
print("Initial position [m]", state[:3])

rotor_velocity = np.full(4, 900.0, dtype=np.float64)
server.set_rotor_velocity_cmd(
    "quadrotor", "quadrotor/base_link", rotor_velocity
)
server.run_once()
server.update_control_states()
state, state_dot = server.control_states["quadrotor"]["quadrotor/base_link"]
print("Position after one step [m]", state[:3])
```

The first ten iterations initialize the simulation. The command then requests
900 rad/s at each rotor. This is a simple actuation check, not a hover
controller. The output contains two three-dimensional positions in metres.

The essential order is **command → advance physics → refresh state → read**.
`control_states` is a cached property, so call `update_control_states()` after
stepping. Its 13-element state layout is described in
{doc}`/concepts/state-frames-units`.

## View a running simulation

`DRLServer` runs without a GUI. While a server with partition `quickstart` is
still alive, a second terminal can attach a Gazebo GUI:

```bash
GZ_PARTITION=quickstart gz sim -g
```

The short script above exits after one command; use an interactive Python
session or a longer control loop when attaching the GUI. Use a different
partition name for each independent simulation.

## Move to an RL environment

A direct server provides state and actuation. A training environment also
needs observations, rewards, and episode termination. GzDRL supplies those
through its registered GazeboPool tasks:

```python
from gzdrl import envs

print(envs.list_all_envs())
```

Choose a task in {doc}`/environments/index`, then follow
{doc}`/guides/gazebo-envpool` to collect transitions. For a custom controller or
world, continue with {doc}`/guides/direct-server`.
