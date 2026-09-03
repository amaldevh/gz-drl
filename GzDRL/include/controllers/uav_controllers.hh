// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Amal Dev Haridevan

#pragma once
#ifndef UAV_CONTROLLER_
#define UAV_CONTROLLER_
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <memory>
#include <stdexcept>
#include <cmath>
#include <iostream>

class UAVController;

namespace uav_controllers
{

  using U = Eigen::Matrix<double, Eigen::Dynamic, 1>; // [T, Mx, My, Mz] or N motor cmds
  using X = Eigen::Matrix<double, 13, 1>;             // [p(3) v(3) q(4: w x y z) w(3)]
  using Gain = Eigen::Matrix<double, 3, 1>;
  using QuatD = Eigen::Matrix<double, 4, 1>;
  using U_f = Eigen::Matrix<float, 4, 1>;
  using X_f = Eigen::Matrix<float, 13, 1>;
  using Gain_f = Eigen::Matrix<float, 3, 1>;
  static const Eigen::Vector3d eps = Eigen::Vector3d::Ones() * 1e-12;

  /**
   * @brief Obtain the derivative of quaternion,
   * defined as the quaternion product : 0.5 * q* (0, w)
   *
   * @param q quaternion
   * @param omega angular velocity
   * @return Eigen::Vector4d Quaternion derivative
   */
  EIGEN_STRONG_INLINE QuatD quaternion_derivative(const Eigen::Quaterniond &q,
                                                  const Eigen::Vector3d &omega) noexcept;

#ifndef UAV_CONTROLLER_ENABLE_LOG
  inline void log_X(const X &) {}
  inline void log_U(const U &) {}
  inline void log_Mat(const Eigen::Matrix3d &) {}
#else
  inline void log_X(const X &v) { std::cout << v.transpose() << "\n"; }
  inline void log_U(const U &v) { std::cout << v.transpose() << "\n"; }
  inline void log_Mat(const Eigen::Matrix3d &m) { std::cout << m << "\n"; }
#endif

}; // namespace uav_controllers

EIGEN_STRONG_INLINE uav_controllers::QuatD uav_controllers::quaternion_derivative(const Eigen::Quaterniond &q,
                                                                                  const Eigen::Vector3d &w) noexcept
{
  // qdot = 0.5 * [ -qx -qy -qz; qw qz -qy; -qz qw qx; qy -qx qw ] * w
  Eigen::Matrix<double, 4, 3> M;
  M << -q.x(), -q.y(), -q.z(),
      q.w(), q.z(), -q.y(),
      -q.z(), q.w(), q.x(),
      q.y(), -q.x(), q.w();
  return 0.5 * M * w;
}

/**
 * @brief A base UAVController interface. This defines
 * two abstract method calculate_force and calculate_moments, which
 * is the most common control abstraction at Collective Thrust and Body Torques.
 * Controllers inherting UAVController can be directly used within
 * the DRLServer. For example you can use DRLServer::set_controller(model_name, link_name, UAVController)
 * and then invoke DRLServer::control_with_wrench or DRLServer::control_with_rotor_velocity,
 * which will automatically invoke the controller that was set
 *
 */
class UAVController
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  /**
   * @brief Construct a new UAVController object
   *
   */
  UAVController() = default;
  /**
   * @brief Destroy the UAVController object
   *
   */
  virtual ~UAVController() = default;

  /**
   * @brief calculates the 3D Force control
   * @param state The current state
   * @param state_dot The current state derivative
   * @param desired The desired state
   * @return Eigen::Vector3d  The Force command
   */
  virtual Eigen::Vector3d calculate_force(const uav_controllers::X &state,
                                          const uav_controllers::X &state_dot,
                                          const uav_controllers::X &desired) = 0;
  /**
   * @brief calculates the 3D Moment control
   * @param state The current state
   * @param state_dot The current state derivative
   * @param desired The desired state
   * @param control_thrust The control thrust command
   * @return Eigen::Vector3d  The Moments command
   */
  virtual Eigen::Vector3d calculate_moments(const uav_controllers::X &state,
                                            const uav_controllers::X &state_dot,
                                            const uav_controllers::X &desired,
                                            const Eigen::Vector3d &control_thrust) = 0;

  /**
   * @brief Computes the CTBT cmd. This depends on the implementation
   * of the calculate_force and calculate_moments (both are pure virtual)
   *
   * @param state current state
   * @param state_dot current state derivative
   * @param desired desired state
   * @return Eigen::Vector4d  the CTBT cmd
   */
  Eigen::Vector4d calculate_thrust_moments(const uav_controllers::X &state,
                                           const uav_controllers::X &state_dot,
                                           const uav_controllers::X &desired)
  {
    Eigen::Vector3d force = calculate_force(state, state_dot, desired);
    Eigen::Vector3d moments = calculate_moments(state, state_dot, desired, force);
    Eigen::Quaterniond q(state[6], state[7], state[8], state[9]);
    q.normalize();
    Eigen::Vector4d thrust_moments;
    thrust_moments << (q.inverse() * force)[2], moments[0], moments[1], moments[2];
    return thrust_moments;
  };
};

#endif // UAV_CONTROLLER_
