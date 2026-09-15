# Parallel simulation

<span id="vectorization"></span>
<span id="asyncdrlserverpool"></span>
<span id="gazebopool"></span>
<span id="multi-robot-environment"></span>

Parallel environments collect several independent trajectories at once.
GzDRL supports two ways to distribute that work: parallel calls to the direct
server API, and native GazeboPool tasks that also compute RL transitions.
Choose according to where your task logic lives.

## Parallel direct servers

`AsyncDRLServerPool` owns a fixed set of `DRLServer` instances, each with a
worker. Calls identify the target environments with `env_ids`. Results follow
the requested ID order, and many arguments accept one value to broadcast to all
selected environments.

Use this interface when Python defines the task or when you need model-level
operations such as pose resets, physical commands, and sensor access. The
Python bindings collect the native futures before returning; the name
“Async” describes the underlying dispatch, not a Python `asyncio` interface.
See {doc}`/guides/async-pool` for a complete example.

## Native GazeboPool tasks

GazeboPool keeps the task, processor, and Gazebo server in C++. An
EnvPool-derived scheduler sends work to a configurable number of threads:

```text
Python policy -- actions and environment IDs --> action queue
                                                    |
                                            workers take requests
                                                    |
                                                    v
                                      persistent task/server instances
                                                    |
                                           completed transitions
                                                    |
                                                    v
Python policy <-- NumPy batch and environment IDs -- state buffers
```

Three settings describe the pool:

| Setting | Meaning |
|---|---|
| `num_envs` | Number of persistent environments |
| `num_threads` | Number of native workers |
| `batch_size` | Number of completed transitions returned by `recv()` |

Workers take requests from the queue; they are not permanently assigned to one
environment. With a batch smaller than the environment count, each receive may
contain a different set of environments. Send actions with the returned
`info["env_id"]` values to preserve the association between a state and its
action. The {doc}`/guides/gazebo-envpool` shows both synchronous and asynchronous
loops.

The Python boundary exposes NumPy views of C++ state buffers. Copy observations
that must be retained independently of the pool's buffer reuse, such as data
stored for later analysis.

## Several robots in one environment

Independent environments and interacting robots are different dimensions of an
experiment. A formation environment contains several quadrotors in one world;
the task commands all of them before advancing their shared physics. It
returns a joint observation, a team reward, and shared episode status.

Increasing `num_envs` creates more independent worlds. Increasing the formation
task's `num_agents` changes the number of robots and the observation and action
shapes within each world.
