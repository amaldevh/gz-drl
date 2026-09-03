// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Amal Dev Haridevan

#ifndef GEOMETRIC_CONTROLLER_HH_
#define GEOMETRIC_CONTROLLER_HH_

#include "controllers/uav_controllers.hh"
using namespace uav_controllers;

/**
 * @brief Geometric controller implementation for UAVs
 */
class GeometricController final : public UAVController
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  /**
   * @brief Construct a new Geometric Controller object
   *
   * @param p_gain Position gain
   * @param d_gain Derivative gain
   * @param att_p_gain P gain for attidue (Inner-loop)
   * @param att_d_gain D gain for attitude (Inner-loop)
   * @param max_lin_acc Maximum linear acceleration
   * @param gravity Gravity vector
   * @param mass Mass of model
   * @param inertia Inertia of the model
   */
  GeometricController(Gain p_gain,
                      Gain d_gain,
                      Gain att_p_gain,
                      Gain att_d_gain,
                      Eigen::Vector3d max_lin_acc,
                      Eigen::Vector3d gravity,
                      double mass,
                      Eigen::Matrix3d inertia);
  ~GeometricController() override = default;

  Eigen::Vector3d calculate_force(const X &state,
                                  const X &state_dot,
                                  const X &desired) override;

  Eigen::Vector3d calculate_moments(const X &state,
                                    const X &state_dot,
                                    const X &desired,
                                    const Eigen::Vector3d &control_thrust) override;

private:
  /**
   * @brief Skew symmetric matrix operator
   *
   * @param v vector
   * @return Eigen::Matrix3d The skew symmetric matrix associated with the vector
   */
  static EIGEN_STRONG_INLINE Eigen::Matrix3d skew(const Eigen::Vector3d &v) noexcept
  {
    Eigen::Matrix3d S;
    S << 0.0, -v.z(), v.y(),
        v.z(), 0.0, -v.x(),
        -v.y(), v.x(), 0.0;
    return S;
  }

  // Gains
  Gain p_gain_;     ///< P gain
  Gain d_gain_;     ///< D gain
  Gain att_p_gain_; ///< Inner-loop P gain
  Gain att_d_gain_; ///< Inner-loop D gain

  Gain stab_out_hi_; ///< Final cmd saturation high
  Gain stab_out_lo_; ///< Final cmd saturation low

  // Vehicle params
  Eigen::Vector3d max_lin_acc_;                                 ///< maximum linear acceleration
  Eigen::Vector3d gravity_;                                     ///< gravity
  double mass_{0.0};                                            ///< mass
  Eigen::Matrix3d inertia_;                                     ///< Inertia
  Eigen::Matrix3d latest_rotmat_ = Eigen::Matrix3d::Identity(); ///< Latest rotation matrix, stored for convenience
};

#endif // GEOMETRIC_CONTROLLER_HH_