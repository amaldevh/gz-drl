#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Amal Dev Haridevan

"""
Generate acados C code for the quadrotor NMPC controller.
This script creates the OCP solver code that will be used by the C++ implementation.
"""
import acados_template as at   
import numpy as np
import casadi as ca
import scipy
import os

def quadrotor_dynamics_ctbt():
    """ Defines the quadrotor dynamics using CasADi for Collective thrust and body torque."""
    # Define the state and control variables
    x = ca.SX.sym('x', 13)  # State: [position(3), velocity(3), quaternion(4), angular_velocity(3)]
    u = ca.SX.sym('u', 4)   # Control: [thrust, moment_x, moment_y, moment_z]

    # Parameters
    mass = ca.SX.sym('mass')
    gravity = ca.SX.sym('gravity', 3)
    I = ca.SX.sym('I', 3, 3)  # Inertia matrix

    # Create the dynamics equations
    pos = x[0:3]
    vel = x[3:6]
    quat = x[6:10]
    quat_n = quat/ca.norm_2(quat)  # Normalize quaternion
    omega = x[10:13]
    thrust = u[0]
    moments = u[1:4]
    R = ca.SX(3, 3)
    
    def quat_prod(q1, q2):
        qw1 = q1[0]
        qw2 = q2[0]
        qv1 = q1[1:]
        qv2 = q2[1:]
        qw3 = qw1*qw2 - ca.dot(qv1, qv2)
        qv3 = qw1*qv2 + qw2*qv1 + ca.cross(qv1, qv2)
        return ca.vertcat(qw3, qv3)


    # Rotation matrix from quaternion
    R[0, 0] = 1 - 2*(quat_n[2]**2 + quat_n[3]**2)
    R[0, 1] = 2*(quat_n[1]*quat_n[2] - quat_n[0]*quat_n[3]) 
    R[0, 2] = 2*(quat_n[1]*quat_n[3] + quat_n[0]*quat_n[2])
    R[1, 0] = 2*(quat_n[1]*quat_n[2] + quat_n[0]*quat_n[3]) 
    R[1, 1] = 1 - 2*(quat_n[1]**2 + quat_n[3]**2)   
    R[1, 2] = 2*(quat_n[2]*quat_n[3] - quat_n[0]*quat_n[1])
    R[2, 0] = 2*(quat_n[1]*quat_n[3] - quat_n[0]*quat_n[2]) 
    R[2, 1] = 2*(quat_n[2]*quat_n[3] + quat_n[0]*quat_n[1])
    R[2, 2] = 1 - 2*(quat_n[1]**2 + quat_n[2]**2)
    acc = (1/mass) * (R @ ca.vertcat(0, 0, thrust)) + gravity
    omega_dot = ca.inv(I) @ (moments - ca.cross(omega, I @ omega))
    # quaternion_dot = 0.5 * ca.vertcat(
    #     -quat[1]*omega[0] - quat[2]*omega[1] - quat[3]*omega[2],
    #      quat[0]*omega[0] + quat[2]*omega[2] - quat[3]*omega[1],
    #      quat[0]*omega[1] - quat[1]*omega[2] + quat[3]*omega[0],
    #      quat[0]*omega[2] + quat[1]*omega[1] - quat[2]*omega[0]
    # )
    quaternion_dot = 0.5*quat_prod(quat, ca.vertcat(0, omega))

    x_dot = ca.vertcat(vel, acc, quaternion_dot, omega_dot)
    return ca.Function('quadrotor_dynamics_ctbt', [x, u, mass, gravity, I], [x_dot])

def quadrotor_dynamics_ctbr():
    """ Defines the quadrotor dynamics using CasADi for Collective thrust and body rates."""
    # Define the state and control variables
    x = ca.SX.sym('x', 10)  # State: [position(3), velocity(3), quaternion(4)]
    u = ca.SX.sym('u', 4)   # Control: [thrust, bodyrate_x, bodyrate_y, bodyrate_z]

    # Parameters
    mass = ca.SX.sym('mass')
    gravity = ca.SX.sym('gravity', 3)

    # Create the dynamics equations
    pos = x[0:3]
    vel = x[3:6]
    quat = x[6:10]
    quat_n = quat/ca.norm_2(quat)  # Normalize quaternion
    thrust = u[0]
    bodyrates = u[1:4]
    R = ca.SX(3, 3)
    
    def quat_prod(q1, q2):
        qw1 = q1[0]
        qw2 = q2[0]
        qv1 = q1[1:]
        qv2 = q2[1:]
        qw3 = qw1*qw2 - ca.dot(qv1, qv2)
        qv3 = qw1*qv2 + qw2*qv1 + ca.cross(qv1, qv2)
        return ca.vertcat(qw3, qv3)


    # Rotation matrix from quaternion
    R[0, 0] = 1 - 2*(quat_n[2]**2 + quat_n[3]**2)
    R[0, 1] = 2*(quat_n[1]*quat_n[2] - quat_n[0]*quat_n[3]) 
    R[0, 2] = 2*(quat_n[1]*quat_n[3] + quat_n[0]*quat_n[2])
    R[1, 0] = 2*(quat_n[1]*quat_n[2] + quat_n[0]*quat_n[3]) 
    R[1, 1] = 1 - 2*(quat_n[1]**2 + quat_n[3]**2)   
    R[1, 2] = 2*(quat_n[2]*quat_n[3] - quat_n[0]*quat_n[1])
    R[2, 0] = 2*(quat_n[1]*quat_n[3] - quat_n[0]*quat_n[2]) 
    R[2, 1] = 2*(quat_n[2]*quat_n[3] + quat_n[0]*quat_n[1])
    R[2, 2] = 1 - 2*(quat_n[1]**2 + quat_n[2]**2)
    acc = (1/mass) * (R @ ca.vertcat(0, 0, thrust)) + gravity
    quaternion_dot = 0.5*quat_prod(quat, ca.vertcat(0, bodyrates))

    x_dot = ca.vertcat(vel, acc, quaternion_dot)
    return ca.Function('quadrotor_dynamics_ctbr', [x, u, mass, gravity], [x_dot])

def acados_ocp_solver(control_type, mass: float, gravity: np.ndarray, I: np.ndarray = None, 
                      output_dir: str = '/tmp/c_generated_code_ocp'):
    """ Creates an acados OCP Solver for a quadrotor dynamics.
    Args:
        control_type (str): Type of control input ("ctbt" for collective thrust and body torques, "ctbr" for collective thrust and body rates).
        mass (float): Mass of the quadrotor.
        gravity (np.ndarray): Gravity vector.
        I (np.ndarray): Inertia matrix.
        output_dir (str): Directory to export the generated C code.
    """
    if (control_type not in ["ctbt", "ctbr"]):
        raise ValueError("Invalid control type. Must be 'ctbt' or 'ctbr'.")
    
    ocp = at.AcadosOcp()
    model = ocp.model
    model.name = 'quadrotor'
    # Set states and control syms
    model.x = ca.SX.sym('x', 13) if control_type == "ctbt" else ca.SX.sym('x', 10)
    model.u = ca.SX.sym('u', 4) 
    
    # Get dynamics expression
    # dm = quadrotor_dynamics_ctbt() if control_type == "ctbt" else quadrotor_dynamics_ctbr()
    # model.f_expl_expr = dm(model.x, model.u, mass, gravity, I) if control_type == "ctbt" else dm(model.x, model.u, mass, gravity)
    if control_type == "ctbt":
        p_mass    = ca.SX.sym('mass')
        p_gravity = ca.SX.sym('gravity', 3)
        p_I       = ca.SX.sym('I', 9)          # flattened row-major 3x3
        model.p   = ca.vertcat(p_mass, p_gravity, p_I)
        dm = quadrotor_dynamics_ctbt()
        model.f_expl_expr = dm(model.x, model.u, p_mass, p_gravity, ca.reshape(p_I, 3, 3))
        # initial parameter values used for code generation / precomputation
        ocp.parameter_values = np.concatenate([[mass], gravity, I.flatten(order='C')])
    else:
        p_mass    = ca.SX.sym('mass')
        p_gravity = ca.SX.sym('gravity', 3)
        model.p   = ca.vertcat(p_mass, p_gravity)
        dm = quadrotor_dynamics_ctbr()
        model.f_expl_expr = dm(model.x, model.u, p_mass, p_gravity)
        ocp.parameter_values = np.concatenate([[mass], gravity])
        
    # Set cost parameters
    nx = model.x.rows()
    nu = model.u.rows()
    ny = nx + nu
    ny_e = nx

    # set cost
    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.cost.cost_type_e = 'NONLINEAR_LS'

    # Tune this weight matrix to get desired performance
    # For State tune Q, for control tune R
    # Intermediate weights: Increased Q from stable set, kept R high for stability
    Q_mat = np.diag([30.0,30.0,30.0, 15.0, 15.0, 15.0, 
                       0.5/16*16, 0.5/16*16, 0.5/16*16,
                         0.5/16*16, 0.1/2*2, 0.1/2*2, 0.1/2*2]) if control_type == "ctbt" else np.diag([30.0,30.0,30.0, 15.0, 15.0, 15.0, 
                       0.5/16*16, 0.5/16*16, 0.5/16*16,
                         0.5/16*16])
    R_mat = np.diag([0.05/8, 0.05/16, 0.05/16, 0.05/16]) if control_type == "ctbt" else np.diag([0.05/8, 0.1/2*2, 0.1/2*2, 0.1/2*2])
    ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)
    ocp.cost.W_e = Q_mat

    # Cost expressions (using Q and R)
    # For terminal cost, we don't have control input
    ocp.model.cost_y_expr = ca.vertcat(model.x, model.u)
    ocp.model.cost_y_expr_e = model.x
    ocp.cost.yref = np.zeros((ny, ))
    ocp.cost.yref_e = np.zeros((ny_e, ))

    # set constraints
    ocp.constraints.lbu = np.array([0.0, -1.0915, -0.8984, -0.0984]) if control_type == "ctbt" else np.array([0.0, -3.0, -3.0, -3.0])
    ocp.constraints.ubu = np.array([+20.0, 1.0915, 0.8984, 0.0984]) if control_type == "ctbt" else np.array([+20.0, 3.0, 3.0, 3.0])
    # If using solve_for_x0, need to set initial condition constraints
    ocp.constraints.x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                   1.0, 0.0, 0.0, 0.0,
                                   0.0, 0.0, 0.0]) if control_type == "ctbt" else np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      1.0, 0.0, 0.0, 0.0])
    # Indices where control bounds are applied (here all controls)
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])

    # set prediction horizon
    ocp.solver_options.N_horizon = 10
    ocp.solver_options.tf = 1000e-3/5  # 0.2 seconds horizon
    ocp.solver_options.qp_solver_iter_max = 50
    # set ocp options
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.qp_tol = 1e-8
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'
    ocp.solver_options.globalization = 'MERIT_BACKTRACKING'

    ocp.code_export_directory = output_dir
    ocp_solver = at.AcadosOcpSolver(ocp)
    
    print(f" Acados C code generated successfully in: {output_dir}")
    print(f"  - Model name: {model.name}")
    print(f"  - State dimension: {nx}")
    print(f"  - Control dimension: {nu}")
    print(f"  - Prediction horizon: {ocp.solver_options.N_horizon}")
    
    return ocp_solver, ocp.solver_options.N_horizon

if __name__ == "__main__":
    # Quadrotor parameters (match the C++ implementation)
    gravity = np.array([0.0, 0.0, -9.82])
    mass = 1.53
    inertia = np.array([[0.0147209, 0, 0], 
                        [0, 0.0169101, 0], 
                        [0, 0, 0.029448]])
    
    print("=" * 60)
    print("Generating acados C code for NMPC controller")
    print("=" * 60)
    
    # Generate the code
    output_dir = '/tmp/c_generated_code_ocp'
    ocp_solver, N_horizon = acados_ocp_solver(control_type="ctbt", mass=mass, gravity=gravity, I=inertia, output_dir=output_dir)

    print("\n" + "=" * 60)
    print("Code generation complete!")
    print("=" * 60)
    print(f"\nGenerated files can be found in: {output_dir}")
    print("\nYou can now build the C++ NMPC controller.")
