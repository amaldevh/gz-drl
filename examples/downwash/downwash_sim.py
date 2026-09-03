# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Amal Dev Haridevan

import gzdrl as grl
import time
import numpy as np
import matplotlib.pyplot as plt 
import tqdm 

if __name__ == "__main__":
    serv = grl.DRLServer( "0", 
                    "world_downwash.sdf",
                    ["quadrotor1", "quadrotor2"],
                    False)
    serv.run_N(10)
    gain_map = grl.GAIN_MAP()
    param_map = grl.PARAMETER_MAP()
    geom_gains = gain_map["qdrone2"]["geometric_controller"]
    geom_params = param_map["qdrone2"]["geometric_controller"]
    controller1 = grl.GeometricController(geom_gains["kp"], geom_gains["kd"],
            geom_gains["kp_att"], geom_gains["kd_att"],
            geom_params.max_accel,
            geom_params.gravity_vec,
            geom_params.mass,
            geom_params.inertia)
    serv.set_controller("quadrotor1", "quadrotor/base_link", controller1)
    controller2 = grl.GeometricController(geom_gains["kp"], geom_gains["kd"],
            geom_gains["kp_att"], geom_gains["kd_att"],
            geom_params.max_accel,
            geom_params.gravity_vec,
            geom_params.mass,
            geom_params.inertia)
    serv.set_controller("quadrotor2", "quadrotor/base_link", controller2)
    drone1_mapping_func = serv.get_thrust_moment_to_rotor_velocity_mapping_function("quadrotor1")
    drone2_mapping_func = serv.get_thrust_moment_to_rotor_velocity_mapping_function("quadrotor2")
    drone1_desired_state = [0, 0, 1.0, 0, 0, 0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    drone2_desired_state = [0, 0.5, 1.25, 0, 0, 0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    rtf_cb = lambda : time.sleep(1e-3)
    drone1_states = []
    drone2_states = []

    for i in range(5000):
        ts = time.time()
        serv.update_control_states()
        s1, sdot1 = serv.control_states["quadrotor1"]["quadrotor/base_link"]
        s2, sdot2 = serv.control_states["quadrotor2"]["quadrotor/base_link"]
        c1 = controller1.calculate_thrust_moments(s1, sdot1, drone1_desired_state)
        c2 = controller2.calculate_thrust_moments(s2, sdot2, drone2_desired_state)
        r1 = drone1_mapping_func(c1)
        r2 = drone2_mapping_func(c2)
        serv.set_rotor_velocity_cmd("quadrotor1", "quadrotor/base_link", r1)
        serv.set_rotor_velocity_cmd("quadrotor2", "quadrotor/base_link", r2)
        serv.run_N(1)
        drone1_states.append(s1.copy())
        drone2_states.append(s2.copy())
        time.sleep(max(0, 1e-3- (time.time() - ts)))
    
    drone2_desired_state_fnc = lambda step: [0, 0.5- 2.0*np.sin(step*2*np.pi/5000), 1.25, 0, 0, 0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    for i in range(10000):
        ts = time.time()
        serv.update_control_states()
        s1, sdot1 = serv.control_states["quadrotor1"]["quadrotor/base_link"]
        s2, sdot2 = serv.control_states["quadrotor2"]["quadrotor/base_link"]
        c1 = controller1.calculate_thrust_moments(s1, sdot1, drone1_desired_state)
        c2 = controller2.calculate_thrust_moments(s2, sdot2, drone2_desired_state_fnc(i))
        r1 = drone1_mapping_func(c1)
        r2 = drone2_mapping_func(c2)
        serv.set_rotor_velocity_cmd("quadrotor1", "quadrotor/base_link", r1)
        serv.set_rotor_velocity_cmd("quadrotor2", "quadrotor/base_link", r2)
        serv.run_N(1)
        drone1_states.append(s1.copy())
        drone2_states.append(s2.copy())
        time.sleep(max(0, 1e-3- (time.time() - ts)))
    
    drone1_states = np.array(drone1_states)
    drone2_states = np.array(drone2_states)
    max_lim = max(drone1_states[:, :3].max(), drone2_states[:, :3].max()) + 0.5
    min_lim = min(drone1_states[:, :3].min(), drone2_states[:, :3].min()) - 0.5

    t = np.arange(drone1_states.shape[0])*1e-3
    fig, axs = plt.subplots(3, 2)
    axs[0][0].plot(t, drone1_states[:, 0], label="drone1 x")
    axs[1][0].plot(t, drone1_states[:, 1], label="drone1 y")
    axs[2][0].plot(t, drone1_states[:, 2], label="drone1 z")
    axs[0][1].plot(t, drone2_states[:, 0], label="drone2 x")
    axs[1][1].plot(t, drone2_states[:, 1], label="drone2 y")
    axs[2][1].plot(t, drone2_states[:, 2], label="drone2 z")
    plt.xlabel("time (s)")
    plt.ylabel("position (m)")
    plt.legend()
    plt.title("Drone positions under downwash effect")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(drone1_states[:, 0], drone1_states[:, 1], drone1_states[:, 2], label="drone1 traj")
    ax.plot(drone2_states[:, 0], drone2_states[:, 1], drone2_states[:, 2], label="drone2 traj")
    
    ax.set_xlim([min_lim, max_lim])
    ax.set_ylim([min_lim, max_lim])
    ax.set_zlim([min_lim, max_lim])
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("3D Trajectories under downwash effect")
    ax.legend()
    plt.show()
