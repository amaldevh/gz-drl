# Environments

GzDRL includes seven native GazeboPool tasks for quadrotor control. Start with
hover to check a training integration, use the trajectory and payload pairs to
compare controller-assisted and direct rotor policies, or choose formation
control for several interacting vehicles in one world.

“LL” means low-level control: the policy supplies rotor commands. The modular
trajectory and payload tasks use a three-dimensional action with an integrated
controller. Their policies have different input and output spaces.

The shapes below describe **one environment**, before the outer batch
dimension. They are defaults; history lengths, rotation representations, and
agent counts can change them. Use `envs.make_spec()` as shown in
{doc}`/guides/gazebo-envpool` to inspect your selected configuration.

```{include} ../_generated/environments-catalog.md
```

## Extend the task set

Follow the hover implementation through its spec, processor, and stepping loop:

```{toctree}
:maxdepth: 1

creating-new-env/index
```
