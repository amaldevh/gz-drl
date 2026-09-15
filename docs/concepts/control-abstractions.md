# Control abstractions

A policy can command rotor motion directly or supply a reference to a controller.
GzDRL supports both approaches through the interfaces below. The task processor
and loaded model determine which command is used; choosing a different task can
therefore change the physical meaning of the policy output.

| Abstraction | Command | Typical interface |
|---|---|---|
| Rotor angular velocity (rad/s) | one angular velocity per rotor | `set_rotor_velocity_cmd` |
| Single-rotor thrust (SRT) | thrust per rotor | `set_srt_cmd` |
| Collective thrust/body torque (CTBT) | thrust plus body moments | `set_ctbt_cmd` |
| Collective thrust/body rate (CTBR) | thrust plus rate references | `set_ctbr_cmd` |
| Wrench | force and torque at a link | `set_wrench` |
| Kinematic velocity | linear or angular velocity (both in body frame) | `set_velocity_cmd`, `set_angular_velocity_cmd` |
| High-level guidance | desired state / residual force | integrated controller methods and task processors |

The rotor plugin simulates rotor forces, torques, and first-order actuator responses.
The binding also provides access to `RotorParameters`, allowing the server to query or update model-specific rotor parameters during domain randomization (if `MultiRotorPlugin` is present in the SDF).
First-order time constants for other abstractions can be set using `DRLServer.set_{abstraction}_rate_limiter_time_constants` for abstractions in `{CTBT, CTBR, SRT}`.


Bundled controller classes include geometric, sliding-mode, and PID controllers.
`UAVController` defines the standard force and moment interface. Example implementations include an NMPC controller using acados, integrated with `DRLServer`.

```{warning}
A normalized policy action is not interchangeable with a direct physical command.
Review the environment documentation and processor before reusing a policy with a different control abstraction.
```

The paper's task study compares end-to-end four-rotor actions with modular three-dimensional force actions for trajectory tracking and payload transport.
The study reports learning behavior for the evaluated configurations.
