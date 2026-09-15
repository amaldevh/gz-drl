# GzDRL

<span id="architecture-at-a-glance"></span>

GzDRL connects reinforcement learning algorithms to Gazebo for aerial robotics
research. It provides explicit control over simulation stepping, parallel
environments, and quadrotor tasks for hover, trajectory tracking, payload
transport, and formation control.

A policy sends an action, GzDRL advances the requested physics iterations, and
the environment returns an observation and reward. The same server interface
also supports controller development and experiments with custom robot models.

These guides are for researchers and developers who use Python and have a
working knowledge of reinforcement learning. C++ is needed when implementing
native GazeboPool tasks or extending the simulator interface.

## Get started

Follow {doc}`getting-started/installation` to build GzDRL, then
{doc}`getting-started/quickstart` to command a quadrotor and read its state.
Continue with {doc}`guides/gazebo-envpool` to collect batched RL transitions.

## Choose an interface

| Your task | Interface | Start here |
|---|---|---|
| Control one world or develop a controller | `gzdrl.DRLServer` | {doc}`guides/direct-server` |
| Run direct server operations across independent worlds | `gzdrl.AsyncDRLServerPool` | {doc}`guides/async-pool` |
| Train with native batched environments | GazeboPool through `gzdrl.envs` | {doc}`guides/gazebo-envpool` |
| Connect ROS nodes to a simulation | `gzdrl.sitl.RosDRLServer` | {doc}`guides/ros-sitl` |

GazeboPool combines a task definition with a server for each environment. Its
scheduler distributes work across C++ threads and returns batches to Python.
The {doc}`concepts/execution-semantics` guide follows the simulation loop;
{doc}`concepts/vectorization` explains how it extends to multiple environments.

## Develop an experiment

The {doc}`environment catalog <environments/index>` describes the seven registered tasks, including
action meanings, observation shapes, and reset settings. The
{doc}`examples section <examples/index>` connects those interfaces to controller and training
scripts. To implement a new task, follow
{doc}`environments/creating-new-env/index`.

For method names and configuration fields, use the {doc}`reference/index`.
The {doc}`research/index` links the software to the GzDRL paper and its
reproducibility and throughput experiments.

```{toctree}
:hidden:
:maxdepth: 2

getting-started/index
concepts/index
guides/index
environments/index
examples/index
modules/index
reference/index
research/index
contributing/index
```
