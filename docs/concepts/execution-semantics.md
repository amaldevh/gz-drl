# From action to observation

<span id="execution-semantics"></span>
<span id="state-capture-and-commands"></span>

An RL environment step must associate an action with the state it produces.
GzDRL makes this sequence explicit: the task applies a command, advances Gazebo
by a chosen number of physics iterations, and reads the resulting state. This
page follows one transition and explains how it determines the policy rate.

## The simulation loop

```text
Policy or application
    |
    | action
    v
Task processor -- physical command or controller reference --> DRLServer
                                                                 |
                                    repeat K times: command, advance, read
                                                                 |
                                                                 v
                                                        Gazebo physics
                                                                 |
                                                       PostUpdate state
                                                                 |
                                                                 v
Policy or application <-- observation, reward, episode status -- Task processor
```

GazeboPool tasks implement this loop in C++. Applications using `DRLServer`
directly provide their own action processing, reward, and episode logic.
A controller is optional: a rotor action can be applied directly, while a
force or state reference may require a controller to compute rotor commands.

Within a GazeboPool transition:

1. The processor converts the policy action to the task's control input.
2. The task writes a command through `DRLServer`. Tasks with an integrated
   controller may recompute this command from the latest state.
3. `run_once()` advances Gazebo by one physics iteration.
4. GzDRL captures link state during Gazebo's `PostUpdate` phase, after the
   physics update. The task refreshes its control-state cache.
5. The task repeats the command and physics loop for
   `physics_steps_per_control` iterations.
6. The processor computes the observation and reward from the resulting state;
   the environment also supplies episode status to the Python adapter.

The core server writes commands through Gazebo's entity-component system
(ECS), the data structure that stores simulation entities and their properties.
ROS messaging is not part of this transition path.

## Physics and policy rates

If the physics timestep is $\Delta t$ and a policy action spans $K$ iterations,
then one transition advances $K\Delta t$ of simulated time. The trajectory
tracking tasks default to $\Delta t=1$ ms and $K=10$: physics runs at 1 kHz in
simulation time, and the policy supplies a new action at 100 Hz.

Wall-clock throughput measures how quickly the computer completes those
transitions. It does not change the simulated control rate. Keep the SDF
physics timestep and any task `physics_dt` setting consistent when modifying a
world.

## Resets and sensors

Resets use a separate request path. `reset_pos()` and `respawn_model()` each
advance three internal physics iterations to apply the change. A task may then
perform additional initialization before producing its first observation.

Rendered sensors use Gazebo's rendering pipeline and may update on a different
schedule from the state cache. A completed physics step does not establish
that a new camera or LiDAR frame is available. See {doc}`/guides/sensors` for
sensor access and {doc}`determinism` for the scope of reproducibility results.
