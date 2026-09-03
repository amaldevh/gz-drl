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
    for i in range(4000):
        t = i * 0.01
        des_state[:6] = trajectory_function(t)
        for _ in range(10):
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

def make_smc_controller(**kwargs):
    """ Create a SMC Controller with specified parameters.
    Args:
        **kwargs: Keyword arguments for controller parameters.
            lambda_pos : numpy.ndarray
                Convergence rate for position error (3 elements).
            kappa_pos : numpy.ndarray
                Switching gain for position error (3 elements).
            lambda_att : numpy.ndarray
                Convergence rate for attitude error (3 elements).
            kappa_att : numpy.ndarray
                Switching gain for attitude error (3 elements).
            boundary_pos : numpy.ndarray
                Boundary layer thickness for position (3 elements).
            boundary_att : numpy.ndarray
                Boundary layer thickness for attitude (3 elements).
            max_lin_acc : numpy.ndarray
                Maximum allowable linear acceleration (3 elements).
            gravity : numpy.ndarray
                Gravity vector [gx, gy, gz] in m/s².
            mass : float  
                Total mass of the UAV in kg.
            inertia : numpy.ndarray
                Inertia matrix of the UAV (3x3).    """
    lambda_pos = kwargs.get("lambda_pos", np.array([2.5, 2.5, 3.0]))
    kappa_pos = kwargs.get("kappa_pos", np.array([4.0, 4.0, 5.0]))
    lambda_att = kwargs.get("lambda_att", np.array([8.0, 8.0, 5.0]))
    kappa_att = kwargs.get("kappa_att", np.array([0.8, 0.8, 0.4]))
    boundary_pos = kwargs.get("boundary_pos", np.array([0.5, 0.5, 0.5]))
    boundary_att = kwargs.get("boundary_att", np.array([0.2, 0.2, 0.2]))
    max_lin_acc = kwargs.get("max_lin_acc", np.array([10.0, 10.0, 10.0]))
    gravity = kwargs.get("gravity", np.array([0.0, 0.0, -7.5]))
    mass = kwargs.get("mass", 1.53)
    inertia = kwargs.get("inertia", 
                        np.array([[0.0147209, 0, 0], 
                                  [0, 0.0169101, 0], 
                                  [0, 0, 0.029448]]))
    print("Controller parameters:")
    print("lambda_pos:", lambda_pos)
    print("kappa_pos:", kappa_pos)
    print("lambda_att:", lambda_att)
    print("kappa_att:", kappa_att)
    print("boundary_pos:", boundary_pos)
    print("boundary_att:", boundary_att)
    print("Gravity:", gravity)
    print("Mass:", mass)
    print("Inertia:", inertia)
    print("Max Acceleration:", max_lin_acc)
    controller = grl.SlidingModeController(lambda_pos, kappa_pos, lambda_att, kappa_att,
                                  boundary_pos, boundary_att, max_lin_acc, gravity, mass, inertia)   
    return controller


def plot_performace(states, desired_state):
    time = np.arange(states.shape[0]) * 0.001
    fig, axs = plt.subplots(3, 2, figsize=(8, 6))
    fig2, ax2 = plt.subplots(3, 1, figsize=(8, 3))
    labels = ['x', 'y', 'z']
    rpy_labels = ['Roll', 'Pitch', 'Yaw']
    rpys = Rotation.from_quat(states[:, 6:10], scalar_first=True).as_euler('xyz')
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

        ax2[i].plot(time, rpys[:, i], label=rpy_labels[i])
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    server = grl.DRLServer("0", 
                           "world_simple.sdf", 
                           ["quadrotor"],
                           False)
    # Final tuning - push Z damping a bit more
    gains = grl.GAIN_MAP()["qdrone2"]["sliding_mode_controller"]
    params = grl.PARAMETER_MAP()["qdrone2"]["sliding_mode_controller"]
    controller= make_smc_controller(
                                lambda_pos=gains["lambda_pos"],
                                kappa_pos=gains["kappa_pos"],
                                lambda_att=gains["lambda_att"],
                                kappa_att=gains["kappa_att"],
                                boundary_pos=gains["boundary_pos"],
                                boundary_att=gains["boundary_att"],
                                max_acc=params.max_accel,
                                gravity=params.gravity_vec,
                                 mass = params.mass,
                                 inertia=params.inertia
     )
    # Create trajectory function
    # should return desired position and velocity at time t
    # as [x , y, z, vx, vy, vz]
    # trajectory_function = lambda t: np.array([2.5 * np.cos(2 * np.pi / 4.0 * t),
    #                                               2.5 * np.sin(2 * np.pi / 4.0 * t),
    #                                               2.0 + 0.5 * np.sin(1 * np.pi / 4.0 * t),
    #                                               -2.5 * (2 * np.pi / 4.0) * np.sin(2 * np.pi / 4.0 * t),
    #                                               2.5 * (2 * np.pi / 4.0) * np.cos(2 * np.pi / 4.0 * t),
    #                                               0.5 * (1 * np.pi / 4.0) * np.cos(1 * np.pi / 4.0 * t)])
    A = 1.0
    B = 1.0
    
    a = 2*np.pi/15.0
    b = a*2
    trajectory_function = lambda t: np.array([ A *np.cos(a*t),
                                                  B *np.sin(b*t),
                                                  1.0,
                                                  -A*a*np.sin(a*t),
                                                   B*b*np.cos(b*t),
                                                 0.0])
    # trajectory_function = lambda t: np.array([ 1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
    states, desired_state = test_trajectory_tracking(server, controller,
                                                     uav_model="quadrotor",
                                                     uav_canonical_link="quadrotor/base_link",
                                                     trajectory_function=trajectory_function)
    plot_performace(states, desired_state)
