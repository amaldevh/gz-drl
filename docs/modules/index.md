# Architecture and components

<span id="modules"></span>

The core server provides the simulation operations used by the other modules.
GazeboPool adds task processing and batched execution; the asynchronous server
pool exposes parallel direct operations. Controllers, sensors, and the optional
ROS bridge serve the workflows that need them.

```{figure} ../_static/system_architecture.svg
:alt: GazeboPool sends actions through worker threads to task processors and DRLServer, advances Gazebo, and returns observation batches to Python.
:class: architecture-figure
:width: 100%

The project architecture figure shows the scheduler above and one transition
below. The “No ROS/gz-transport” label refers to action and state exchange in
the direct stepping path. Gazebo transport is still used by facilities such as
the GUI, and the optional ROS bridge has its own messaging loop.
```

The simpler flow in {doc}`/concepts/execution-semantics` follows one transition
without the scheduler. Use the component pages below to locate the code that
implements each part.

```{include} ../_generated/modules-catalog.md
```
