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
    n_envs = 5
    serv = grl.AsyncDRLServerPool(n_envs, "no_sensor", "world_racing.sdf", ["quadrotor"], False)
    envids = list(range(n_envs))
    serv.run_N(envids, [100])  # warm up

    print("Testing server step performance without sensor retrieval:")
    results = timer(lambda: serv.run_N(envids, [1]), 1000)
    print(f"Frequency: {results['freq']*n_envs} Hz")
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
    controllers = [make_controller() for _ in envids]
    serv.set_controller(envids, ["quadrotor"], 
                        ["quadrotor/base_link"],
                        controllers
                        )
    serv.reset_pos(envids, ["quadrotor"], [[0,0,0.15]], [[0,0,0]])
    print("Testing server + controller step performance (individual python steps):")
    desired_state = np.zeros((13))
    desired_state[:3] =[ 1.0, 2.0, 1.5]
    desired_state[6] = 1.0  # w
    results = timer(lambda: serv.control_with_rotor_velocity(envids, ["quadrotor"],
                                                              ["quadrotor/base_link"],
                                                              [desired_state], [1]), 1000)
    print(f"Frequency: {results['freq']*n_envs} Hz")

    print("Testing server + controller step performance (c++ delegated steps):")
    results = timer(lambda: serv.control_with_rotor_velocity(envids, ["quadrotor"],
                                                              ["quadrotor/base_link"],
                                                              [desired_state], [1000]), 1)
    print(f"Frequency: {results['freq']*1000*n_envs} Hz")


def test_server_with_sensor():
    n_envs = 5
    serv = grl.AsyncDRLServerPool(n_envs, "with_sensor", "world_racing.sdf", ["quadrotor"], True)
    envids = list(range(n_envs))
    serv.run_N(envids, [100])  # warm up
    cam_names = serv.camera_sensor_names([envids[0]])
    assert cam_names and cam_names[0], (
        "world_racing.sdf must expose at least one camera sensor"
    )
    serv.run_N(envids, [100])  # warm up

    print("Testing server step performance with sensors enabled:")
    results = timer(lambda: serv.run_N(envids, [1]), 1000)
    print(f"Frequency: {results['freq']*n_envs} Hz")
if __name__ == "__main__":
    test_server_no_sensor()
    test_server_with_sensor()

