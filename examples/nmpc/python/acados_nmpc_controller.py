# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Amal Dev Haridevan

import acados_template as at   
import numpy as np
import casadi as ca
from scipy.spatial.transform import Rotation
import scipy
import gzdrl as grl
import matplotlib.pyplot as plt
import time
from generate_acados_code import acados_ocp_solver, quadrotor_dynamics_ctbt


class NMPCController(grl.UAVController):
    """ NMPC Controller using acados for quadrotor trajectory tracking.
    Args:
        ocp_solver (at.AcadosOcpSolver): The acados OCP solver instance.
        N_horizon (int): The prediction horizon length.
    """
    def __init__(self, control_type, ocp_solver: at.AcadosOcpSolver, N_horizon: int,
                 mass: float , gravity: np.ndarray, inertia: np.ndarray):
        """ NMPC Controller using acados for quadrotor trajectory tracking.
        Args:
            ocp_solver (at.AcadosOcpSolver): The acados OCP solver instance.
            N_horizon (int): The prediction horizon length.
            mass (float): Mass of the quadrotor.
            gravity (np.ndarray): Gravity vector.
            inertia (np.ndarray): Inertia tensor
        """
        super().__init__()
        if (control_type not in ["ctbr", "ctbt"]):
            raise RuntimeError("control type not supported")
        self.ocp_solver = ocp_solver
        self.N_horizon = N_horizon
        self.mass = mass
        self.gravity = gravity
        self.u_ref = np.array([mass * np.linalg.norm(gravity), 0.0, 0.0, 0.0])
        self.u_opt = self.u_ref.copy()
        self.control_type = control_type
        self.kp_omega = np.array([ 0.1876, 0.1744, 0.12790])
        self.kd_omega = np.array([0.0032, 0.0030, 0.0120])
        self.inertia = inertia
        self._uopt_w_running_mean = np.zeros((3,))
        self._u_opt_N = 0
        self._moment_running_mean = np.zeros((3,))
        self._max_mom = np.zeros((3,))
        self.max_omeg = np.zeros((3,))

    def calculate_force(self,  state: np.ndarray, 
                                 state_dot:  np.ndarray, 
                                 desired_state: np.ndarray) -> np.ndarray:
        """ Calculate thrust and moments using NMPC.
        Args:
            state (np.ndarray): Current state of the UAV.
            state_dot (np.ndarray): Current state derivative of the UAV.
            desired_state (np.ndarray): Desired state of the UAV.
        Returns:
            np.ndarray: Calculated thrust and moments.
        """
        # Set initial state constraint
        # self.ocp_solver.set(0, "x", state)
        # # Set reference trajectory
        des_state_ = desired_state[:10] if self.control_type == "ctbr" else desired_state
        state_ = state[:10] if self.control_type == "ctbr" else state
        for i in range(self.N_horizon):
            self.ocp_solver.set(i, "yref", np.concatenate((des_state_, self.u_opt)))
        self.ocp_solver.set(self.N_horizon, "yref", des_state_)
        # status = self.ocp_solver.solve()
        self.u_opt = self.ocp_solver.solve_for_x0(x0_bar =state[:10] if self.control_type == "ctbr" else state, fail_on_nonzero_status = True)
        # convert to inertial frame force
        force_body = np.array([0.0, 0.0, self.u_opt[0]])
        quat = state[6:10]
        R = Rotation.from_quat(quat, scalar_first=True).as_matrix()
        force_inertial = R @ force_body
        # print("Thrust:", self.u_opt[0], "Moments:", self.u_opt[1:4], "Force_inertial:", force_inertial)
        return force_inertial
    
    def calculate_moments(self,  state: np.ndarray, 
                                 state_dot:  np.ndarray, 
                                 desired_state: np.ndarray,
                                 force: np.ndarray) -> np.ndarray:
        """ Calculate moments using NMPC.
        Args:
            state (np.ndarray): Current state of the UAV.
            state_dot (np.ndarray): Current state derivative of the UAV.
            desired_state (np.ndarray): Desired state of the UAV.
            force (np.ndarray): Calculated force on the UAV.
        Returns:
            np.ndarray: Calculated moments.
        """
        if self.control_type == "ctbr":
            moments = self.kp_omega * (self.u_opt[1:4] - state[10:13]) + self.kd_omega * (0.0 - state_dot[10:13]) + np.cross(state[10:13], 
                                                              self.inertia@state[10:13] )
            self._uopt_w_running_mean = (self._uopt_w_running_mean * self._u_opt_N + self.u_opt[1:4]) / (self._u_opt_N + 1)
            self._moment_running_mean = (self._moment_running_mean * self._u_opt_N + moments) / (self._u_opt_N + 1)
            self._u_opt_N += 1
            self._max_mom = np.maximum(self._max_mom, np.abs(moments))
            self.max_omeg = np.maximum(self.max_omeg, self.u_opt[1:4])
            # print(self.u_opt[1:4].round(2),
            #        " Running mean:", self._uopt_w_running_mean.round(2),
            #          " Moments:", moments.round(2), " Moment mean:", self._moment_running_mean.round(2),
            #          " Max moments:", self._max_mom.round(2), " Max omeg:", self.max_omeg.round(2))
        else:
            moments = self.u_opt[1:4]
        return np.clip(moments, np.array([-1.0915, -0.8984, -0.0984]),
                       np.array([1.0915, 0.8984, 0.0984]))

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
        if i%100 == 0:
            print("Iteration:", i)
        t = i * 0.01
        des_state[:6] = trajectory_function(t)
        for _ in range(10):
            server.update_control_states()
            state, state_dot = server.control_states[uav_model][uav_canonical_link]
            states.append(state.copy())
            desired_states.append(des_state.copy())
            
            if controller.control_type == "ctbr":
                force = controller.calculate_force(state, state_dot, des_state)
                bodyrates = scipy.spatial.transform.Rotation.from_quat(state[6:10], scalar_first=True).as_matrix() @ controller.u_opt[1:]
                server.set_wrench(uav_model, uav_canonical_link,force, np.array((0.0,0.0,0.0)))
                server.set_angular_velocity_cmd(uav_model, uav_canonical_link, bodyrates)
            else:
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
    # plt.show()
    return np.array(states), np.array(desired_states)


def plot_performace(states, desired_state):
    time = np.arange(states.shape[0]) * 0.001
    fig, axs = plt.subplots(3, 2, figsize=(8, 6))
    fig2, ax2 = plt.subplots(3, 1, figsize=(8, 3))
    fig3, ax3 = plt.subplots(3, 1, figsize=(8, 3))
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

        ax3[i].plot(time, states[:, i+3], label=labels[i] + ' Velocity')
        ax3[i].plot(time, desired_state[:, i+3], label='Desired ' + labels[i] + ' Velocity', linestyle='--')
        ax3[i].set_xlabel('Time (s)')
        ax3[i].set_ylabel(labels[i] + ' Velocity (m/s)')
        ax3[i].legend()
        ax3[i].grid()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    gravity = np.array([0.0, 0.0, -9.81])
    mass = 1.53
    inertia = np.array([[0.0147209, 0, 0], 
                                  [0, 0.0169101, 0], 
                                  [0, 0, 0.029448]])
    control_type = "ctbt"
    ocp_solver, N_horizon = acados_ocp_solver(control_type, mass, gravity, inertia)
    controller = NMPCController(control_type, ocp_solver, N_horizon, mass, gravity, inertia)
    server = grl.DRLServer("0", "world_simple.sdf", ["quadrotor"], False)
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

    states, desired_states = test_trajectory_tracking(server, controller,
                                                      uav_model="quadrotor",
                                                     uav_canonical_link="quadrotor/base_link" ,
                                                     trajectory_function=trajectory_function)
    plot_performace(states, desired_states)
