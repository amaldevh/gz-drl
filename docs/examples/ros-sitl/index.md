---
title: ROS 2 bridge
summary: Publish and subscribe around a real-time RosDRLServer.
order: 60
tags: [ROS 2, Jazzy, SITL]
---

# ROS 2 bridge

Source example: `examples/ros/ros_server.py`.

This example connects the simulation to ROS using GzDRL's direct bridge.
Build with ROS detected as described in {doc}`/guides/ros-sitl`, then source
Jazzy and run from the repository root:

```bash
source /opt/ros/jazzy/setup.bash
python examples/ros/ros_server.py
```

The example starts simulation, prints the subscribed and published topic maps,
spins the executor asynchronously, and obtains the underlying direct server.
Its source warns about data races: do not step that server independently while
the bridge owns the real-time loop.
