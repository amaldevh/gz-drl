// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Amal Dev Haridevan

#ifndef NMPC_CONTROLLER_HH_
#define NMPC_CONTROLLER_HH_

#include "controllers/uav_controllers.hh"
#include "acados_solver_quadrotor.h"

using namespace uav_controllers;

/** @class NMPCController
 * @brief Nonlinear Model Predictive Controller using acados for UAV trajectory tracking
 */
class NMPCController final : public UAVController {
  public:
    inline static const int CTBT = 0;
    inline static const int CTBR = 1;

  public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
   * @brief Construct a new NMPCController object
   * @param mass Mass of the quadrotor (kg)
   * @param gravity Gravity vector (m/s^2)
   * @param inertia Inertia matrix (kg*m^2)
   */
  NMPCController(int control_type, double mass, 
                 const Eigen::Vector3d& gravity,
                 const Eigen::Matrix3d& inertia);
  
  ~NMPCController() override;

  /**
   * @brief Calculate thrust force using NMPC
   * @param state Current state of the UAV [pos(3), vel(3), quat(4), omega(3)]
   * @param state_dot Current state derivative
   * @param desired Desired state
   * @return Eigen::Vector3d Calculated force in inertial frame
   */
  Eigen::Vector3d calculate_force(const X& state,
                                 const X& state_dot,
                                 const X& desired) override;

  /**
   * @brief Calculate control moments using NMPC
   * @param state Current state of the UAV
   * @param state_dot Current state derivative
   * @param desired Desired state
   * @param control_thrust Calculated thrust force
   * @return Eigen::Vector3d Calculated moments in body frame
   */
  Eigen::Vector3d calculate_moments(const X& state,
                                    const X& state_dot,
                                    const X& desired,
                                    const Eigen::Vector3d& control_thrust) override;

  /**
   * @brief Factory method to create NMPC controller for qdrone2
   * @return std::shared_ptr<UAVController> Shared pointer to controller
   */
  static std::shared_ptr<NMPCController> tuned_nmpc() {
    int control_type;
    if (QUADROTOR_NX == 13){
      control_type=0;
    }
    else if (QUADROTOR_NX == 10){
      control_type=1;
    }
    else {
      throw std::runtime_error("Unsupported state dim");
    }
    Eigen::Vector3d g(0.0, 0.0, -9.82);
    Eigen::Matrix3d I;
    I << 0.0147209, 0, 0,
         0, 0.0169101, 0,
         0, 0, 0.029448;
    const double m = 1.54;
    return std::make_shared<NMPCController>(control_type, m, g, I);
  }

  /**
    * @brief Calculate the bodyrates, only usable if control type is CTBR
    * @param state Current state of the UAV
    * @param state_dot Current state derivative
    * @param desired Desired state
    * @param control_thrust Calculated thrust force
    * @return Eigen::Vector3d Calculated bodyrates in body frame
   */
  Eigen::Vector3d calculate_bodyrates(const X& state,
                                    const X& state_dot,
                                    const X& desired,
                                    const Eigen::Vector3d& control_thrust);
  
  
                                    /**
    * @brief Calculate the ctbr only usable if control type is CTBR
    * @param state Current state of the UAV
    * @param state_dot Current state derivative
    * @param desired Desired state
    * @param control_thrust Calculated thrust force
    * @return Eigen::Vector3d Calculated bodyrates in body frame
   */
  Eigen::Vector4d calculate_thrust_bodyrates(const X& state,
                                    const X& state_dot,
                                    const X& desired);


  /** @brief Set vehicle parameters
   * @param mass Mass of the quadrotor (kg)
   * @param gravity Gravity vector (m/s^2)
   * @param inertia Inertia matrix (kg*m^2)
   */
  void set_vehicle_params(double mass, const Eigen::Vector3d& gravity,
                        const Eigen::Matrix3d& inertia);
private:
  // Acados solver capsule
  quadrotor_solver_capsule* acados_ocp_capsule_;
  
  // Vehicle parameters
  double mass_;
  Eigen::Vector3d gravity_;
  Eigen::Matrix3d inertia_;
  
  // Reference control input (hover thrust)
  Eigen::Vector4d u_ref_;
  
  // Optimal control solution
  Eigen::Vector4d u_opt_;
  
  // Prediction horizon
  int N_horizon_;

  // control_type
  int control_type_;
};

#endif  // NMPC_CONTROLLER_HH_
