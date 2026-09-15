# Run direct servers in parallel

<span id="parallel-direct-servers"></span>

`gzdrl.AsyncDRLServerPool` applies direct server operations across independent
Gazebo worlds using native workers. It is useful when your application already
implements observations and rewards in Python, or needs direct access to each
world's models. Complete {doc}`direct-server` before using the pool.

## Select environments and issue commands

This example initializes four worlds, sends one rotor command to each, and
advances all four simulations:

```python
import numpy as np
import gzdrl

pool = gzdrl.AsyncDRLServerPool(
    4,
    "parallel-",
    str(gzdrl.get_sdf_path("world_simple.sdf")),
    ["quadrotor"],
    False,
)
env_ids = [0, 1, 2, 3]

try:
    pool.run_N(env_ids, [10])
    commands = [np.full(4, 900.0) for _ in env_ids]
    pool.set_rotor_velocity_cmd(
        env_ids, ["quadrotor"], ["quadrotor/base_link"], commands
    )
    pool.run_once(env_ids)
    states = pool.state_info(env_ids, ["quadrotor"])
    print("Received states for", len(states), "environments")
finally:
    pool.close()
```

`[10]`, `["quadrotor"]`, and `["quadrotor/base_link"]` each broadcast one value
to the selected environments. `commands` supplies one vector per environment.
Check individual overloads in {doc}`/reference/python-api` before assuming a
parameter accepts broadcasting. Returned results follow the requested ID order.

## Understand completion

The bound pool operations dispatch work to C++ and collect the results before
returning to Python. Many release the Python global interpreter lock (GIL)
while waiting. Although an `AsyncToken` type is also bound, these calls are not
Python awaitables.

Close the pool after the final operation so its workers can shut down. For
an RL integration using this interface, see
{doc}`/examples/vectorized-hover/index`. If you want C++ task processing and
observation batches as well, use {doc}`gazebo-envpool`.
