# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Amal Dev Haridevan

import gzdrl as grl
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.spatial.transform import Rotation

def test_trajectory_tracking(server: grl.DRLServer, 
                             controller: grl.UAVController,
                             uav_model: str = "quadrotor",
                             uav_canonical_link: str = "quadrotor/base_link",
                             trajectory_function: callable = None) -> tuple[np.ndarray, np.ndarray]:
    """ Test the trajectory tracking performance of the given controller on the server.
     Args:
        server (grl.DRLServer): The DRL server instance.
        controller (grl.UAVController): The UAV controller to be tested.
        uav_model (str): The name of the UAV model in the simulation.
        uav_canonical_link (str): The canonical link of the UAV model to which MultiRotorPlugin is attached.
        trajectory_function (callable): A function that defines the desired trajectory. It should take time t as input and return desired position and velocity.
     Returns:
        np.ndarray: Recorded states during the test.
        np.ndarray: Desired states during the test.
    """
    des_state = np.zeros((13))
    des_state[6] = 1.0  # set desired quaternion w component to
    if (trajectory_function is None):
        x = lambda t:  np.cos(2*np.pi/4.0 * t) * 2.5
        y = lambda t: np.sin(3*np.pi/4.0 * t) * 2.5
        z = lambda t: 2.0 + 0.5 * np.sin (1*np.pi/4.0 * t)
        pos = lambda t: np.array([x(t), y(t), z(t)])
        vx = lambda t: -np.sin(2*np.pi/4.0 * t) * (2*np.pi/4.0) * 2.5
        vy = lambda t:  np.cos(3*np.pi/4.0 * t) * (3*np.pi/4.0) * 2.5
        vz = lambda t: 0.5 * np.cos(1*np.pi/4.0 * t) * (1*np.pi/4.0)
        vel = lambda t: np.array([vx(t), vy(t), vz(t)])
        trajectory_function = lambda t: np.concatenate((pos(t), vel(t)))
    server.reset_pos(uav_model, [0, 0, 0.15], [0, 0, 0])
    server.run_N(10)
    states = []
    desired_states = []
    print("Starting test...")
    mapping_func = server.get_thrust_moment_to_rotor_velocity_mapping_function(uav_model)
    ts = time.perf_counter()
    steps_per_point = int(1e-2 / server.step_size())
    for i in range(4000):
        t = i * 0.01
        des_state[:6] = trajectory_function(t)
        d_rad = 0.5*np.sin(2 * np.pi * t / 2.5)
        des_state[6] =  1.0 #np.cos(d_rad / 2.0)
        des_state[9] = 0.0 #np.sin(d_rad / 2.0)
        
        for _ in range(steps_per_point):
            server.update_control_states()
            state, state_dot = server.control_states[uav_model][uav_canonical_link]
            states.append(state.copy())
            desired_states.append(des_state.copy())
            TM = controller.calculate_thrust_moments(state,
                                                    state_dot,
                                                    des_state,)
            server.set_rotor_velocity_cmd(uav_model, 
                                          uav_canonical_link, 
                                          mapping_func(TM))
            server.run_N(1)
    te = time.perf_counter()
    print(f"Test duration: {te - ts:.2f}s")
    print("Averaged FPS:", 10000 / (te - ts))
    return np.array(states), np.array(desired_states)

def make_geometric_pd_controller(**kwargs):
    """ Create a Geometric PD Controller with specified parameters.
    Args:
        **kwargs: Keyword arguments for controller parameters.
            kp (np.ndarray): Proportional gain for position control.
            kd (np.ndarray): Derivative gain for position control.
            kp_att (np.ndarray): Proportional gain for attitude control.
            kd_att (np.ndarray): Derivative gain for attitude control.
            gravity (np.ndarray): Gravity vector.
            mass (float): Mass of the UAV.
            inertia (np.ndarray): Inertia matrix of the UAV.    """
    # Tuned gains for good trajectory tracking performance
    # Z gains are higher for fast altitude response
    # XY gains are moderate to avoid attitude oscillations
    gain_map = grl.GAIN_MAP()
    param_map = grl.PARAMETER_MAP()
    geom_gains = gain_map["qdrone2"]["geometric_controller"]
    geom_params = param_map["qdrone2"]["geometric_controller"]

    kp = kwargs.get("kp", geom_gains["kp"])
    kd = kwargs.get("kd", geom_gains["kd"])
    kp_att = kwargs.get("kp_att", geom_gains["kp_att"])
    kd_att = kwargs.get("kd_att", geom_gains["kd_att"])
    gravity = kwargs.get("gravity", geom_params.gravity_vec)
    mass = kwargs.get("mass",geom_params.mass)  
    inertia = kwargs.get("inertia", 
                        geom_params.inertia)
    # max_acc magnitude = 2.5, redistributed for higher z authority
    # x=y=1.06, z=2.0 → sqrt(1.06^2 + 1.06^2 + 2.0^2) ≈ 2.5
    max_acc = kwargs.get("max_acc", geom_params.max_accel)
    print("Controller parameters:")
    print("Kp:", kp)
    print("Kd:", kd)
    print("Kp_att:", kp_att)
    print("Kd_att:", kd_att)
    print("Gravity:", gravity)
    print("Mass:", mass)
    print("Inertia:", inertia)
    print("Max Acceleration:", max_acc)
    controller = grl.GeometricController(kp, kd, kp_att, kd_att,
                                        max_acc, gravity, mass, inertia)
    return controller


def plot_performace(states, desired_state):
    time = np.arange(states.shape[0]) * 0.001
    fig, axs = plt.subplots(3, 2, figsize=(8, 6))
    fig2, ax2 = plt.subplots(3, 2, figsize=(8, 6))
    labels = ['x', 'y', 'z']
    rpy_labels = ['Roll', 'Pitch', 'Yaw']
    rot_mats = Rotation.from_quat(states[:, 6:10], scalar_first=True)
    rpys = rot_mats.as_euler('xyz')
    rot_mats = rot_mats.as_matrix()
    rpys[:, 2] = np.arctan2(rot_mats[:, 1, 0], rot_mats[:, 0, 0])
    des_rot_mats = Rotation.from_quat(desired_state[:, 6:10], scalar_first=True)
    rpy_des = des_rot_mats.as_euler('xyz')
    des_rot_mats = des_rot_mats.as_matrix()
    rpy_des[:, 2] = np.arctan2(des_rot_mats[:, 1, 0], des_rot_mats[:, 0, 0])
    fig3, ax3 = plt.subplots(3, 2, figsize=(8, 6))
    for i in range(3):
        axs[i][0].plot(time, states[:, i], label='Actual ' + labels[i])
        axs[i][0].plot(time, desired_state[:, i], label='Desired ' + labels[i], linestyle='--')
        axs[i][0].set_xlabel('Time (s)')
        axs[i][0].set_ylabel(labels[i] + ' Position (m)')
        axs[i][0].legend()
        axs[i][0].grid()

        axs[i][1].plot(states[:, i], states[:, i+3], label='Phase ' + labels[i])
        axs[i][1].set_xlabel(labels[i] + ' Position (m)')
        axs[i][1].set_ylabel(labels[i] + ' Velocity (m/s)')
        axs[i][1].legend()
        axs[i][1].grid()

        ax2[i][0].plot(time, rpys[:, i], label=rpy_labels[i])
        ax2[i][0].plot(time, rpy_des[:, i], label='Desired ' + rpy_labels[i], linestyle='--')
        ax2[i][0].set_xlabel('Time (s)')
        ax2[i][0].set_ylabel(rpy_labels[i] + ' (rad)')
        ax2[i][0].legend()
        ax2[i][0].grid()
        
        ax2[i][1].plot(time, states[:, i+10], label=rpy_labels[i] + ' Rate')
        ax2[i][1].set_xlabel('Time (s)')
        ax2[i][1].set_ylabel(rpy_labels[i] + ' Rate (rad/s)')
        ax2[i][1].legend()
        ax2[i][1].grid()
        
        # plot vel and accel 
        ax3[i][0].plot(time, states[:, i+3], label=labels[i] + ' Velocity')
        ax3[i][0].set_xlabel('Time (s)')
        ax3[i][0].set_ylabel(labels[i] + ' Velocity (m/s)')
        ax3[i][0].legend()
        ax3[i][0].grid()
        ax3[i][1].plot(time[1:], np.diff(states[:, i+3]) / 0.001, label=labels[i] + ' Acceleration')
        ax3[i][1].set_xlabel('Time (s)')
        ax3[i][1].set_ylabel(labels[i] + ' Acceleration (m/s^2)')
        ax3[i][1].legend()
        ax3[i][1].grid()    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    server = grl.DRLServer("0", 
                           "world_simple.sdf", 
                           ["quadrotor"],
                           False)
    np.random.seed(int(time.time()))
    random_values = (np.random.uniform(-1, 1, size=(19,)) + 11)/11.0
    print("Randomized Physics Values:", random_values)
    controller= make_geometric_pd_controller(
        # kp = np.array([5.0, 5.0, 10.0])*random_values[0:3],
        #                        kd = np.array([5.0, 5.0, 5.0])*random_values[3:6],
        #                        kp_att = np.array([25.0, 25.0, 1.0])*random_values[6:9],
        #                        kd_att = np.array([1.2, 1.2, 0.2])*random_values[9:12],
        #                        mass = 1.53*random_values[12],
        #                        inertia =
        #                             np.array([[0.0147209*random_values[13], 0, 0], 
        #                           [0, 0.0169101*random_values[14], 0], 
        #                           [0, 0, 0.029448*random_values[15]]]),
        #                         max_acc =  np.array([20.0, 20.0, 35.0])*random_values[16:19]
                               )
    # Create trajectory function
    # should return desired position and velocity at time t
    # as [x , y, z, vx, vy, vz]
    # trajectory_function = lambda t: np.array([2.5 * np.cos(2 * np.pi / 4.0 * t),
    #                                               2.5 * np.sin(3 * np.pi / 4.0 * t),
    #                                               2.0 + 0.5 * np.sin(1 * np.pi / 4.0 * t),
    #                                               -2.5 * (2 * np.pi / 4.0) * np.sin(2 * np.pi / 4.0 * t),
    #                                               2.5 * (3 * np.pi / 4.0) * np.cos(3 * np.pi / 4.0 * t),
    #                                               0.5 * (1 * np.pi / 4.0) * np.cos(1 * np.pi / 4.0 * t)])
    A = 1.0
    B = 1.0
    C = 1.0

    c = 1 #2*np.pi/15.0
    b = 1 #c*2
    a = 0.5

    trajectory_function = lambda t: np.array([ A *np.sin(a*t),
                                                  B *np.sin(b*t),
                                                  1.5 + np.sin(c*t+ np.pi/2),
                                                  A*a*np.cos(a*t),
                                                   B*b*np.cos(b*t),
                                                 C*c*np.cos(c*t + np.pi/2)])
    
    # trajectory_function = lambda t: np.array([ 1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
    states, desired_state = test_trajectory_tracking(server, controller,
                                                     uav_model="quadrotor",
                                                     uav_canonical_link="quadrotor/base_link",
                                                     trajectory_function=trajectory_function)
    plot_performace(states, desired_state)
