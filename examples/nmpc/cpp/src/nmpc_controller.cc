// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Amal Dev Haridevan

#include "nmpc_controller.hh"
#include <iostream>

NMPCController::NMPCController(int control_type, double mass,
                               const Eigen::Vector3d& gravity,
                               const Eigen::Matrix3d& inertia)
    : control_type_(control_type),
      mass_(mass),
      gravity_(gravity),
      inertia_(inertia),
      u_opt_(Eigen::Vector4d::Zero()) {
  
  // sanity check for NX, to prevent running CTBT model against CTBR 
  if (control_type_ == NMPCController::CTBR){
    if (QUADROTOR_NX != 10){
      throw std::runtime_error("Using CTBR, but the state dim is not 10");
    }
    if (QUADROTOR_NP != 4){
      throw std::runtime_error("Using CTBR, but param dim is not 4");
    }
  }else if (control_type_ == NMPCController::CTBT){
    if (QUADROTOR_NX != 13){
      throw std::runtime_error("Using CTBT, but the state dim is not 13");
    }
    if (QUADROTOR_NP != 13){
      throw std::runtime_error("Using CTBT, but param dim is not 13");
    }
  }
  else {
    throw std::runtime_error("Unsupported control type");
  }
  // Create acados solver capsule
  acados_ocp_capsule_ = quadrotor_acados_create_capsule();
  
  // Create the solver
  int status = quadrotor_acados_create(acados_ocp_capsule_);
  if (status != 0) {
    std::cerr << "Failed to create acados solver, status: " << status << std::endl;
    throw std::runtime_error("Failed to create acados solver");
  }
  
  // Get prediction horizon
  N_horizon_ = QUADROTOR_N;
  
  // Calculate hover thrust reference
  u_ref_ << mass_ * gravity_.norm(), 0.0, 0.0, 0.0;
  // set vehicle params in the solver
  set_vehicle_params(mass_, gravity_, inertia_);
  std::cout << "NMPC Controller initialized with:" << std::endl;
  std::cout << "  Mass: " << mass_ << " kg" << std::endl;
  std::cout << "  Gravity: " << gravity_.transpose() << " m/s^2" << std::endl;
  std::cout << "  Horizon: " << N_horizon_ << " steps" << std::endl;
  std::cout << "  Hover thrust: " << u_ref_(0) << " N" << std::endl;
}

NMPCController::~NMPCController() {
  // Free acados solver
  if (acados_ocp_capsule_ != nullptr) {
    int status = quadrotor_acados_free(acados_ocp_capsule_);
    if (status != 0) {
      std::cerr << "Failed to free acados solver, status: " << status << std::endl;
    }
    status = quadrotor_acados_free_capsule(acados_ocp_capsule_);
    if (status != 0) {
      std::cerr << "Failed to free acados capsule, status: " << status << std::endl;
    }
  }
}

Eigen::Vector3d NMPCController::calculate_force(const X& state,
                                               const X& state_dot,
                                               const X& desired) {
  // Get the nlp configuration
  ocp_nlp_config* nlp_config = quadrotor_acados_get_nlp_config(acados_ocp_capsule_);
  ocp_nlp_dims* nlp_dims = quadrotor_acados_get_nlp_dims(acados_ocp_capsule_);
  ocp_nlp_in* nlp_in = quadrotor_acados_get_nlp_in(acados_ocp_capsule_);
  ocp_nlp_out* nlp_out = quadrotor_acados_get_nlp_out(acados_ocp_capsule_);
  ocp_nlp_solver* nlp_solver = quadrotor_acados_get_nlp_solver(acados_ocp_capsule_);
  
  // Prepare state vector (need to convert Eigen to array)
  // this safe against both CTBR and CTBT states. Because state is organized as (pos, vel, quat, [omega opt])
  double x0[QUADROTOR_NX];
  for (int i = 0; i < QUADROTOR_NX; i++) {
    x0[i] = state(i);
  }
  
  // Set initial state constraint (equivalent to solve_for_x0 in Python)
  ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, 0, "lbx", x0);
  ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, 0, "ubx", x0);
  
  // Prepare reference (state + control)
  double yref[QUADROTOR_NY];
  for (int i = 0; i < QUADROTOR_NX; i++) {
    yref[i] = desired(i);
  }
  // Add control reference
  for (int i = 0; i < QUADROTOR_NU; i++) {
    yref[QUADROTOR_NX + i] = u_ref_(i);
  }
  
  // Set reference trajectory for all stages
  for (int i = 0; i < N_horizon_; i++) {
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "yref", yref);
  }
  
  // Set terminal reference (only state, no control)
  double yref_e[QUADROTOR_NYN];
  for (int i = 0; i < QUADROTOR_NX; i++) {
    yref_e[i] = desired(i);
  }
  ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N_horizon_, "yref", yref_e);
  
  // Solve OCP
  int status = quadrotor_acados_solve(acados_ocp_capsule_);
  if (status != 0) {
    std::cerr << "NMPC solver failed with status: " << status << std::endl;
  }
  
  // Get optimal control at first stage
  double u0[QUADROTOR_NU];
  ocp_nlp_out_get(nlp_config, nlp_dims, nlp_out, 0, "u", u0);
  
  // Store optimal control
  for (int i = 0; i < QUADROTOR_NU; i++) {
    u_opt_(i) = u0[i];
  }
  
  // Convert body frame thrust to inertial frame force
  Eigen::Vector3d force_body(0.0, 0.0, u_opt_(0));
  Eigen::Quaterniond q(state(6), state(7), state(8), state(9));
  q.normalize();
  Eigen::Vector3d force_inertial = q* force_body;
  
  return force_inertial;
}

Eigen::Vector3d NMPCController::calculate_moments(const X& state,
                                                  const X& state_dot,
                                                  const X& desired,
                                                  const Eigen::Vector3d& control_thrust) {
  // Return the moments from the optimal solution
  if (control_type_ == NMPCController::CTBR){
      throw std::runtime_error("calculate_moments should not be called for CTBR controller");
  }
  return u_opt_.tail<3>();
}


Eigen::Vector3d NMPCController::calculate_bodyrates(const X& state,
                                                  const X& state_dot,
                                                  const X& desired,
                                                  const Eigen::Vector3d& control_thrust) {
  // Return the bodyrates from the optimal solution
  if (control_type_ == NMPCController::CTBT){
      throw std::runtime_error("calculate_bodyrates should not be called for CTBT controller");
  }
  return u_opt_.tail<3>();
}


Eigen::Vector4d NMPCController::calculate_thrust_bodyrates(const X& state,
                                                  const X& state_dot,
                                                  const X& desired) {
  // Return the bodyrates from the optimal solution
  if (control_type_ == NMPCController::CTBT){
      throw std::runtime_error("calculate_bodyrates should not be called for CTBT controller");
  }
    Eigen::Vector3d force = calculate_force(state, state_dot, desired);
    Eigen::Vector3d bodyrates = calculate_bodyrates(state, state_dot, desired, force);
    Eigen::Quaterniond q(state[6], state[7], state[8], state[9]);
    q.normalize();
    Eigen::Vector4d thrust_bodyrates;
    thrust_bodyrates << (q.inverse()*force)[2], bodyrates[0], bodyrates[1], bodyrates[2];
    return thrust_bodyrates;
}


void NMPCController::set_vehicle_params(double mass,
                                        const Eigen::Vector3d& gravity,
                                        const Eigen::Matrix3d& inertia) {
    mass_    = mass;
    gravity_ = gravity;
    inertia_ = inertia;
    u_ref_   << mass_ * gravity_.norm(), 0.0, 0.0, 0.0;

    if (control_type_ == CTBT) {
        // [mass(1), gravity(3), I_row_major(9)] = 13 params
        double p[13];
        p[0] = mass_;
        p[1] = gravity_(0); p[2] = gravity_(1); p[3] = gravity_(2);
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
                p[4 + r*3 + c] = inertia_(r, c);
        for (int i = 0; i <= N_horizon_; i++)
            quadrotor_acados_update_params(acados_ocp_capsule_, i, p, 13);
    } else {
        // [mass(1), gravity(3)] = 4 params
        double p[4] = {mass_, gravity_(0), gravity_(1), gravity_(2)};
        for (int i = 0; i <= N_horizon_; i++)
            quadrotor_acados_update_params(acados_ocp_capsule_, i, p, 4);
    }
}