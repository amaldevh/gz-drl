# Paper and implementation

<span id="paper-and-implementation-map"></span>
<span id="paper-to-source-map"></span>
<span id="qualification-boundaries"></span>

[GzDRL: Reproducible and Scalable Deep Reinforcement Learning with Gazebo](https://arxiv.org/abs/2609.13243)
by Amal Dev Haridevan, Junjie Kang, and Jinjun Shan introduces the framework
and evaluates its use in aerial robotics. The paper is available as an arXiv
preprint; use {doc}`citation` when citing it.

## Find the implementation

| Topic | Source components |
|---|---|
| Controlled simulation stepping | `DRLServer`, `DRLHelperSystem`, command and run methods |
| Parallel direct server calls | `AsyncDRLServerPool` |
| Native batched tasks | GazeboPool and the EnvPool-derived scheduler |
| Task definitions | `env_specs`, `processors`, `gazebo_envpool` |
| Control interfaces | Controller classes and rotor/command APIs |
| Physical randomization | Mass, inertia, rotor parameter, and respawn APIs |
| Sensors | Sensor interface and server sensor methods |
| ROS integration | `gzdrl.sitl.RosDRLServer` |

The {doc}`/modules/index` includes the project architecture figure. The
{doc}`/concepts/execution-semantics` explains one action-to-observation cycle.

## Read the experimental results

Synchronization and reproducibility concern the controlled state transition
path. Throughput comparisons depend on the evaluated workloads and hardware.
The training study fixes hardware and stochastic inputs, while the hardware
study evaluates a particular trained policy on QDrone2.

Use {doc}`benchmarks` for the conditions behind throughput comparisons and
{doc}`reproducibility` for experiment settings and source harnesses.
