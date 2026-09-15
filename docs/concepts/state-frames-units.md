# State, coordinate frames, and units

<span id="state-frames-and-units"></span>
<span id="naming"></span>
<span id="action-normalization"></span>
<span id="task-observations"></span>
<span id="accessing-sensor-data"></span>

A controller needs both the state values and their coordinate conventions.
GzDRL provides a 13-element control state and a separate 19-element Gazebo link
state. This page distinguishes their layouts so you can connect observations,
controllers, and actuator commands consistently.

## Direct server state

After advancing physics, refresh the control-state cache and select a model
and link:

```python
server.update_control_states()
state, state_dot = server.control_states[model_name][link_name]
```

`state` has the following layout:

| Slice | Quantity | Unit / convention |
|---|---|---|
| `0:3` | Position | Metres, world frame |
| `3:6` | Linear velocity | Metres/second, world frame |
| `6:10` | Orientation quaternion | Scalar-first `(w, x, y, z)` |
| `10:13` | Angular velocity | Radians/second, body frame |

`state_dot` has the same number of elements: linear velocity, linear
acceleration, quaternion derivative, and angular acceleration occupy the
corresponding slices.

`state_info(model_name)` returns link records directly from the helper system's
state cache. Each 19-element record uses a different layout:

| Slice | Quantity | Unit / convention |
|---|---|---|
| `0:3` | Position | Metres, world frame |
| `3:7` | Orientation quaternion | Scalar-first `(w, x, y, z)` |
| `7:10` | Linear velocity | Metres/second, world frame |
| `10:13` | Angular velocity | Radians/second, body frame |
| `13:16` | Linear acceleration | Metres/second², world frame |
| `16:19` | Angular acceleration | Radians/second², body frame |

Use one layout consistently when passing data to a controller. In particular,
the quaternion begins at index 6 in the control state and index 3 in the
link-state record.

## Model and link names

Names must match the loaded SDF. The bundled quadrotor examples use model
`quadrotor` and link `quadrotor/base_link`; the slash is part of the link name.
Nested model lookup uses scoped names such as `outer_model::inner_model`.
Copy the model and link identifiers from the SDF or task spec rather than
shortening them.

## Policy actions and physical commands

GazeboPool tasks declare actions in `[-1, 1]`. Their processors convert these
values to a task-specific control input:

- hover and the rotor-velocity tasks map four values to rotor speeds;
- inverted pendulum LL maps four values to rotor thrusts;
- modular payload transport maps three values to a force in the world frame;
- modular trajectory tracking uses its configured action scale and bias;
- formation control supplies four values per agent.

Direct `DRLServer` command methods use physical units. For example,
`set_rotor_velocity_cmd()` receives radians per second and does not apply a
GazeboPool processor's normalization. See {doc}`control-abstractions` before
connecting a policy to a different command interface.

## Task observations and sensor data

An RL observation can combine state features, tracking references, and history.
It is not necessarily either raw state layout above. Task pages in
{doc}`/environments/index` describe their observation composition and default
shapes.

Camera and LiDAR arrays use separate discovery and retrieval methods. See
{doc}`/guides/sensors` for sensor names, callbacks, and reset restrictions.
