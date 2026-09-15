---
title: SITL and ROS bridge
summary: Real-time ROS integration plus a deployment-oriented low-pass filter.
order: 60
tags: [ROS 2, SITL, Python]
---

# SITL and ROS bridge

The `gzdrl.sitl` package always exposes `SecondOrderLPFilter3d`. When CMake detects ROS
1 or ROS 2, it also builds `RosDRLServer`, which owns a core server, publishes
simulation data, subscribes to commands, and runs at a requested real-time
factor.

The current Dockerfile uses ROS 2 Jazzy, Ubuntu 24.04, and Gazebo Jetty.
See {doc}`/guides/ros-sitl` for build selection and runtime ownership.

`RosDRLServer` is a direct GzDRL ROS bridge, not a `ros_gz` integration.


Keep training and deployment boundaries explicit: GazeboPool is the batched
middleware-free learning path; `RosDRLServer` is the real-time integration path, suitable for `SITL`.
