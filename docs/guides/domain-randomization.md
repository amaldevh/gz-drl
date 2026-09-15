# Randomize vehicle dynamics

<span id="domain-randomization"></span>

Domain randomization varies simulated dynamics between episodes so a policy
experiences more than one model. GzDRL exposes mass, inertia, rotor parameters,
and actuator response settings. This guide shows how to change a direct
server's model and explains when the bundled tasks apply randomization.

## Change mass and inertia

Mass and inertia setters update the cached SDF model. A respawn applies those
values to the simulated entity. Use this sequence with a server constructed
with rendered sensors disabled:

```{warning}
Do not respawn a model while rendered sensors are enabled. Recreating the model
can invalidate Gazebo rendering resources. A pose-only reset with `reset_pos()`
does not apply cached mass or inertia changes.
```

Starting from the server in {doc}`direct-server`:

```python
import numpy as np

mass = server.get_mass("quadrotor", "quadrotor/base_link")
inertia = server.get_inertia("quadrotor", "quadrotor/base_link")
position = np.array([0.0, 0.0, 1.0])
orientation = np.zeros(3)

server.set_mass("quadrotor", "quadrotor/base_link", mass * 1.05)
server.set_inertia("quadrotor", "quadrotor/base_link", inertia * 1.05)
server.respawn_model("quadrotor", position, orientation)
server.update_control_states()
```

This applies a five-percent increase and resets the pose. `respawn_model()`
performs three stabilization iterations before returning. Store nominal
parameters separately when sampling repeated episodes, so successive scale
factors do not accumulate unintentionally.

Rotor dynamics are available through `get_rotor_parameters()` and
`set_rotor_parameters()` when the model uses the multirotor plugin. SRT, CTBR,
and CTBT commands also expose rate-limiter time constants. Consult the
{doc}`/reference/python-api` for their parameter types and application behavior.

## Randomization in bundled tasks

The task implementation determines whether a flag changes physical parameters
and how often it does so:

| Environment | Default | Application point when enabled |
|---|---|---|
| Hover | Disabled | Reset does not call its randomization helpers |
| Inverted pendulum LL | Disabled | Every reset |
| Payload transport | Enabled | Episode counter divisible by 10 |
| Payload transport LL | Enabled | Episode counter divisible by 10 |
| Multi-agent formation LL | Enabled | Episode counter divisible by 10 |
| Trajectory tracking | Disabled | Episode counter divisible by 20 |
| Trajectory tracking LL | Disabled | Episode counter divisible by 20 |

The divisible-counter checks include the initial counter value. In particular,
setting `domain_randomization=True` on hover does not activate the currently
unused helper functions.

Record the flag, sampling ranges, seed, reset cadence, and controller
parameters with the experiment. These settings are needed to interpret an
evaluation under changed dynamics; see {doc}`/research/reproducibility`.
