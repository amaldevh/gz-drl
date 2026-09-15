---
title: Async server pool
summary: Parallel model-level operations over persistent DRLServer instances.
order: 20
tags: [C++, Python, threading]
---

# Async server pool

`AsyncDRLServerPool` owns multiple core servers. The C++ pool schedules work on
native workers; `pybind11` methods receive explicit `env_ids`, release the GIL
while waiting, and preserve result order relative to the request IDs.

Its surface mirrors most direct-server operations, including stepping, reset,
commands, state access, model randomization, sensors, and camera recording.
Broadcast rules let many calls accept one model/link/value for all requested
environments or one per environment.

`AsyncToken` is bound as a native future handle, but the current pool operations
collect their futures before returning to Python. Refer to specific method
docstrings for verifying blocking behavior, as some methods may require GIL hold.

This is Python-facing vectorization of the low-level server API. It is distinct
from GazeboPool, which also owns task processing and state-buffer scheduling.
