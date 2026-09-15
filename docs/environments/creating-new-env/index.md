---
catalog: false
---

# Create a GazeboPool environment

<span id="creating-new-gazebopool-envs"></span>
<span id="specs"></span>

A native GazeboPool environment combines three parts: a **spec** that defines
configuration and spaces, a **processor** that transforms actions and states,
and a **task** that owns the simulation and episode lifecycle. This guide
follows the working hover implementation so you can adapt a complete task to a
new aerial robotics problem.

You will need a working C++ build, an SDF world with known model and link names,
and a definition of the desired action, observation, reward, and termination
conditions. First review {doc}`/concepts/execution-semantics` to understand
where these operations occur in a transition.

## Start from the hover task

Use these files together as the implementation reference:

| Part | Source | Responsibility |
|---|---|---|
| Spec | `envs/env_specs/hover_spec.hh` | Defaults and tensor dimensions |
| Processor | `envs/processors/hover_processor.hh` | Action scaling, observations, history, reward |
| Task | `envs/gazebo_envpool/gazebo_hover.hh` | Construction, reset, stepping, completion |
| Binding | `envs/gazebo_envpool/gazebo_envpool_envs.cc` | Export the spec and task to Python |

Create corresponding files for your task and rename the classes and header
guards. Keep the existing hover task available as a working reference during
development.

## Define configuration and spaces

A spec derives from `GazeboSpec`. Its `DefaultConfig()` combines
`GazeboSpec::BaseGazeboConfig()` with task fields, including the required
`max_steps_per_episode`. Python callers can override these values through
keyword arguments to `envs.make_spec()` and the pool constructors.

The hover spec defines its observation and action spaces as follows:

```{literalinclude} ../../../envs/env_specs/hover_spec.hh
:language: cpp
:start-at:     template <typename Config>
:end-before: using HoverEnvSpec
```

The leading `-1` in the action spec is EnvPool's variable player dimension.
For a single-agent task, the Python action batch has shape
`(num_envs, action_dimension)`. The observation spec describes one environment;
the returned array adds the batch dimension.

Make every shape-dependent setting part of the spec. For example, hover uses
`14 + 4 * action_history_size` observation values. A processor must write
exactly the layout that its spec allocates.

## Implement the processor

Derive the processor from `GazeboProcessor<EnvSpec<YourSpec>>`. The hover
processor demonstrates these hooks:

| Hook | Input | Result |
|---|---|---|
| `ProcessAction` | Normalized policy action | Physical command stored in `processed_action_` |
| `ProcessObservation` | Current and previous state maps | Observation and any declared `info` fields |
| `ComputeReward` | States and policy action | Reward for each tracked state key |

State maps use a key formed by concatenating the model and link names. Reuse
the same key in the task and processor. Actions and state buffers have fixed
shapes established by the spec; initialize history buffers during reset and
write every declared field.

## Construct and step the task

Derive the task from `GazeboEnvpool<YourSpec, YourProcessor>`. The base class
already provides `drl_server_`, `processor_`, model/link maps, current and
previous state maps, rewards, and episode counters. Initialize those members
rather than declaring separate copies.

During construction, acquire `AcquireConstructionLock()` before constructing
Gazebo objects. Use `UniqueEnvid()` for the transport partition and the logical
`envid` passed to the constructor for per-environment random seeds. Separate
pools in the same process should use non-overlapping `gz_partition_offset`
ranges. This keeps transport identifiers distinct without coupling random
streams to thread scheduling.

The hover task's `Step()` implements the complete action-to-observation path:

```{literalinclude} ../../../envs/gazebo_envpool/gazebo_hover.hh
:language: cpp
:start-at:         void Step(const Action &action) override
:end-before:         /** @brief Update the randomized parameters */
```

For a task with an integrated controller, compute its command inside the
physics loop if the controller should run at the physics rate. After the final
iteration, compute reward and write the observation into the buffer returned
by `Allocate()`. The hover `WriteObs()` method also assigns the `reward` field.

Implement `IsDone()` for episode length and task failure or success conditions.
The EnvPool lifecycle calls it when publishing state. Implement `Reset()` to
reset model state, counters, history, references, and rewards, then write the
initial observation. If you use physical randomization, document when reset
applies it and whether it requires a respawn. See
{doc}`/guides/domain-randomization`.

## Register the task

Include the new task header in `envs/gazebo_envpool/gazebo_envpool_envs.cc`, bring
its type into scope, and add the binding beside the existing tasks. For classes
named `MyTaskEnv` and `MyTaskEnvSpec`, the binding line is:

```cpp
MAKE_PY_GAZEBO_ENVPOOL(m, MyTaskEnv, MyTaskEnvSpec)
```

Rebuild from the repository root:

```bash
python -m pip install .
```

The installed `gzdrl.envs` registry discovers exported spec/pool pairs. Inspect
the result before starting a simulation:

```python
from gzdrl import envs

print(envs.list_all_envs())
spec = envs.make_spec("GazeboPoolMyTaskEnv-v0", num_envs=1)
print(spec.config)
print(spec.gymnasium_observation_space)
print(spec.gymnasium_action_space)
```

For an independently compiled pybind11 module exporting the same binding
pattern, import the module and call `envs.env_loader(module)` to register it.
Use the installed {doc}`/reference/cmake` for downstream headers and libraries.

## Check the first rollout

Start with one environment and one worker. Check reset observations, finite
state and reward values, action scaling, episode completion, and a second reset.
Then compare repeated runs using the same seed before increasing the worker
count. Exercise configuration changes that affect the tensor shapes.

Add a page using {doc}`/contributing/new-environment` when the task is ready.
The documentation build checks that every registered native task has a matching
catalog entry.
