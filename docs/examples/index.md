# Examples

The repository includes scripts for controller development, RL training,
evaluation, and ROS integration. Run commands from the `gz-drl` repository root
with GzDRL installed in the active Python environment. The scripts remain in the
source checkout; they are not installed as part of the wheel.

Start with synchronous stepping to inspect the simulator, or Python hover
training to connect it to Stable-Baselines3. The vectorized hover example uses
`AsyncDRLServerPool`; native GazeboPool stepping is covered in
{doc}`/guides/gazebo-envpool`.

```{include} ../_generated/examples-catalog.md
```
