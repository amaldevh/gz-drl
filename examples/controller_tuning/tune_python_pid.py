# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Amal Dev Haridevan

import gzdrl as grl
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.spatial.transform import Rotation 

class PIDController(grl.UAVController):
    """ A simple PID controller for UAV trajectory tracking. 
    """
    def __init__(self, kp: np.ndarray, ki: np.ndarray, kd: np.ndarray,
                 kp_att: np.ndarray , kd_att: np.ndarray, 
                 kp_att_rate: np.ndarray, kd_att_rate: np.ndarray,
                 mass: float, gravity: np.ndarray,
                 dt: float = 0.01):
        """ Initialize the PID controller with given parameters.    
        Args:
            kp (np.ndarray): Proportional gain for position control.
            ki (np.ndarray): Integral gain for position control.
            kd (np.ndarray): Derivative gain for position control.
            kp_att (np.ndarray): Proportional gain for attitude control.
            kd_att (np.ndarray): Derivative gain for attitude control.
            kp_att_rate (np.ndarray): Proportional gain for attitude rate control.
            kd_att_rate (np.ndarray): Derivative gain for attitude rate control.
            mass (float): Mass of the UAV.
            gravity (np.ndarray): Gravity vector.
            dt (float): Time step for integration.
        """
        super().__init__()
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.kp_att = kp_att
        self.kd_att = kd_att
        self.kp_att_rate = kp_att_rate
        self.kd_att_rate = kd_att_rate
        self.mass = mass
        self.gravity = gravity
        self.dt = dt
        self._g = np.linalg.norm(gravity)
        self.integral_error = np.zeros(3)

    def calculate_force(self, state: np.ndarray,
                        state_dot: np.ndarray,
                        desired_state: np.ndarray) -> np.ndarray:
        """ Calculate the required force based on current and desired states.
        Args:
            state (np.ndarray): Current state of the UAV.
            state_dot (np.ndarray): Current derivative of the state.
            desired_state (np.ndarray): Desired state of the UAV.
        Returns:
            np.ndarray: Calculated force vector.
        """
        pos_error = desired_state[:3] - state[:3]
        vel_error = desired_state[3:6] - state_dot[:3]
        self.integral_error += pos_error * self.dt
        self.integral_error = np.clip(self.integral_error, -2, 2)
        force = (self.kp * pos_error +
                 self.ki * self.integral_error +
                 self.kd * vel_error -
                 self.mass * self.gravity)
        return force
    
    def calculate_moments(self, state: np.ndarray,
                          state_dot: np.ndarray,
                          desired_state: np.ndarray,
                          force: np.ndarray) -> np.ndarray:
        """ Calculate the required moments based on current and desired states.
        Args:
            state (np.ndarray): Current state of the UAV.
            state_dot (np.ndarray): Current derivative of the state.
            desired_state (np.ndarray): Desired state of the UAV.
            force (np.ndarray): Calculated force vector.
        Returns:
            np.ndarray: Calculated moment vector.
        """
        rpy = Rotation.from_quat(state[6:10], scalar_first=True).as_euler('xyz')
        des_rpy = Rotation.from_quat(desired_state[6:10], scalar_first=True).as_euler('xyz')
        roll, pitch, yaw = rpy
        if (np.cos(pitch) == 0.0):
            pitch += 1e-6  # avoid singularity
        omega = state[10:13]
        omega_des = state_dot[10:13]
        alpha = state_dot[10:13]
        roll_des = -force[1]/(self.mass*self._g)
        pitch_des = force[0]/(self.mass*self._g)
        yaw_des = des_rpy[2]
        rpy_error = np.array([roll_des - roll,
                              pitch_des - pitch,
                              yaw_des - yaw])
        omega_error = - (omega - omega_des)
        omega_cmd = self.kp_att * rpy_error + self.kd_att * omega_error
        omega_cmd = np.clip(omega_cmd, -5.0, 5.0)
        omega_error = omega_cmd - omega
        moments = (self.kp_att_rate * omega_error - self.kd_att_rate * alpha)
        return moments #np.clip(moments, -3.0, 3.0)
    
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
    for i in range(1000):
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
    controller = PIDController(kp = np.array([0.4712389*5.5,  0.4712389*5.5, 31.5]),
                               kd = np.array([0.41887902*6,  0.41887902*6, 23  ])    ,
                               ki = np.array([0.01, 0.01, 6. ])     ,
                               kp_att = np.array([14.0, 14.0, 0.1])  ,
                               kd_att = np.array([0.1, 0.1, 0.05]),
                               kp_att_rate= np.array([0.1876, 0.1876, 0.05]),
                               kd_att_rate= np.array([0.0032, 0.0032, 0.0001]),
                               mass = 1.53,
                               gravity=np.array([0.0, 0.0, -9.82]),
                               dt=0.001
                               )
    # Create trajectory function
    # should return desired position and velocity at time t
    # as [x , y, z, vx, vy, vz]
    trajectory_function = lambda t: np.array([2.5 * np.cos(2 * np.pi / 4.0 * t),
                                                  2.5 * np.sin(3 * np.pi / 4.0 * t),
                                                  2.0 + 0.5 * np.sin(1 * np.pi / 4.0 * t),
                                                  -2.5 * (2 * np.pi / 4.0) * np.sin(2 * np.pi / 4.0 * t),
                                                  2.5 * (3 * np.pi / 4.0) * np.cos(3 * np.pi / 4.0 * t),
                                                  0.5 * (1 * np.pi / 4.0) * np.cos(1 * np.pi / 4.0 * t)])
    

    states, desired_state = test_trajectory_tracking(server, controller,
                                                     uav_model="quadrotor",
                                                     uav_canonical_link="quadrotor/base_link",
                                                     trajectory_function=trajectory_function)
    plot_performace(states, desired_state)
