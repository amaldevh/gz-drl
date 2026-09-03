# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Amal Dev Haridevan

import gzdrl as grl
import time 

if __name__ == "__main__":
    serv = grl.DRLServer("1", "world_simple.sdf", ["quadrotor"], False)
    serv.run_N(100)  # warm up
    ts = time.perf_counter()
    for _ in range(1000):
        serv.run_N(1)
        # do something with the state of the sim
        # serv....
        # state is preserved between steps/runs
    te = time.perf_counter()
    dt = te - ts
    freq = 1000 / dt 
    print(f"Frequency: {freq} Hz")
