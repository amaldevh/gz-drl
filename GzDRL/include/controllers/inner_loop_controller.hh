// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Amal Dev Haridevan

#ifndef INNER_LOOP_CONTROLLER_HH
#define INNER_LOOP_CONTROLLER_HH

#include "controllers/uav_controllers.hh"

using namespace uav_controllers;

/**
 * @brief An inner loop controller that takes in Fx-Fy-Fz yaw-rate
 * and computes CTBR commands.
 */
class InnerLoopController final : public UAVController
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    /**
     * @brief Construct a new Inner Loop Controller object
     *
     * @param kp_angle P gain for angle cmd
     * @param kd_angle D gain for angle cmd
     * @param kp_angular_rate P gain for Omega cmd
     * @param kd_angular_rate D gain for Omgea cmd
     * @param mass Mass of model
     * @param inertia Inertia of model
     * @param gravity  Gravity vector
     */
    InnerLoopController(const Gain &kp_angle,
                        const Gain &kd_angle,
                        const Gain &kp_angular_rate,
                        const Gain &kd_angular_rate,
                        const double mass,
                        const Eigen::Matrix3d &inertia,
                        const Eigen::Vector3d &gravity);

    /** @brief Destructor */
    ~InnerLoopController() = default;

    /** @brief computes force command */
    Eigen::Vector3d calculate_force(const X &state,
                                    const X &state_dot,
                                    const X &desired) override;
    /** @brief computes moment command */
    Eigen::Vector3d calculate_moments(const X &state,
                                      const X &state_dot,
                                      const X &desired,
                                      const Eigen::Vector3d &control_thrust) override;

private:
    Gain kp_angle_;        ///< P gain for angle
    Gain kd_angle_;        ///< D gain for angle
    Gain kp_angular_rate_; ///< P gain for omega
    Gain kd_angular_rate_; ///< D gain for omega
    // To store latest rotation matrix
    Eigen::Matrix3d latest_rotmat_; ///< most rceent rotation matrix, stored for convenience
    // Saturation params
    Gain angle_cmd_max_;        ///< Maximum angle cmd
    Gain angular_rate_cmd_max_; ///< Maximum omega cmd
    Gain final_cmd_max_;        ///< Maximum torque
    // Helper function to compute desired angles from Fx, Fy, Fz, yaw_rate
    Eigen::Matrix3d force_to_thrust_angles; ///< Simple mapping matrix to convert force to angle cmd
    Eigen::Matrix3d inertia_;               ///< Inertia matrix of model
    double mass_;                           ///< mass of model
    Eigen::Vector3d gravity_;               ///< gravity vector
    double g_;                              ///< gravity magnitude
};
#endif // INNER_LOOP_CONTROLLER_HH