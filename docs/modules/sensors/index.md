---
title: Sensor interface
summary: Optional Gazebo camera, depth, LiDAR, contact, and recording access.
order: 50
tags: [Gazebo, rendering, callbacks]
---

# Sensor interface

The sensor library integrates Gazebo camera, depth-camera, RGB-D,
segmentation, thermal, wide-angle, LiDAR, GPU LiDAR, and bounding-box sensor
components selected by CMake. `DRLServer` exposes discovery, frame retrieval,
callback, and contact APIs.

`VideoRecordingManager` backs camera recording controls in the Python binding.
Sensor support is opt-in at server construction because rendering and data
movement change the workload.

Part of the sensor/plugin source is derived from Gazebo and retains upstream
Apache-2.0 notices.

Rendered sensor output is outside the paper's state-transition determinism
claim. See {doc}`/guides/sensors`.
