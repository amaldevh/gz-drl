# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Amal Dev Haridevan

import gzdrl as grl
import numpy as np
import time 

def timer(func, iters):
    ts = time.perf_counter()
    for _ in range(iters):
        func()
    te = time.perf_counter()
    dt = te - ts
    freq = iters / dt 
    return {"dt": dt, "freq": freq}

def test_server_no_sensor():
    serv = grl.DRLServer("no_sensor", "world_racing.sdf", ["quadrotor"], False)
    serv.run_N(100)  # warm up

    print("Testing server step performance without sensor retrieval:")
    results = timer(lambda: serv.run_N(1), 1000)
    print(f" Frequency: {results['freq']} Hz")
    # Test controller step performance 
    gain_map = grl.GAIN_MAP()
    param_map = grl.PARAMETER_MAP()
    geom_gains = gain_map["qdrone2"]["geometric_controller"]
    geom_params = param_map["qdrone2"]["geometric_controller"]
    make_controller = lambda :  grl.GeometricController(geom_gains["kp"], geom_gains["kd"],
        geom_gains["kp_att"], geom_gains["kd_att"],
        geom_params.max_accel,
        geom_params.gravity_vec,
        geom_params.mass,
        geom_params.inertia)
    controller = make_controller()
    serv.set_controller("quadrotor", 
                        "quadrotor/base_link",
                        controller
                        )
    serv.reset_pos("quadrotor", [0,0,0.15], [0,0,0])
    print("Testing server + controller step performance (individual python steps):")
    desired_state = np.zeros((13))
    desired_state[:3] =[ 1.0, 2.0, 1.5]
    desired_state[6] = 1.0  # w
    results = timer(lambda: serv.control_with_rotor_velocity("quadrotor",
                                                              "quadrotor/base_link",
                                                              desired_state, 1), 1000)
    print(f" Frequency: {results['freq']} Hz")

    print("Testing server + controller step performance (c++ delegated steps):")
    results = timer(lambda: serv.control_with_rotor_velocity("quadrotor",
                                                              "quadrotor/base_link",
                                                              desired_state, 1000), 1)
    print(f" Frequency: {results['freq']*1000} Hz")

def test_server_with_sensor():
    serv = grl.DRLServer("with_sensor", "world_racing.sdf", ["quadrotor"], True)
    serv.run_N(100)  # warm up
    cam_names = serv.camera_sensor_names()
    assert cam_names, "world_racing.sdf must expose at least one camera sensor"
    serv.run_N(100)  # warm up

    print("Testing server step performance with sensors enabled:")
    results = timer(lambda: serv.run_N(1), 1000)
    print(f" Frequency: {results['freq']} Hz")

if __name__ == "__main__":
    test_server_no_sensor()
    test_server_with_sensor()
