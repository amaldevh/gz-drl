// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Amal Dev Haridevan

#ifndef SMC_CONTROLLER_HH
#define SMC_CONTROLLER_HH

#include "controllers/uav_controllers.hh"
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>

/**
 * @brief Sliding Mode Controller implementation for UAV
 *
 */
class SlidingModeController final : public UAVController
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using X = uav_controllers::X;
    using U = uav_controllers::U;

    /**
     * @brief Construct a new Sliding Mode Controller object
     *
     * @param lambda_pos Lambda gain for position
     * @param kappa_pos  Kappa gain for position, a.k.a switching gain
     * @param lambda_att Lambda gain for attitude
     * @param kappa_att  Kappa gain for attitude, a.k.a switching gain
     * @param boundary_pos Boundary vector for position,
     * @param boundary_att Boundary vector for attitude
     * @param max_lin_acc  Maximum linear acceleration
     * @param gravity  Gravity vector
     * @param mass Mass of model
     * @param inertia Inertia of the model
     */
    SlidingModeController(Eigen::Vector3d lambda_pos,
                          Eigen::Vector3d kappa_pos, // Switching gain position
                          Eigen::Vector3d lambda_att,
                          Eigen::Vector3d kappa_att,    // Switching gain attitude
                          Eigen::Vector3d boundary_pos, // Boundary layer (phi)
                          Eigen::Vector3d boundary_att,
                          Eigen::Vector3d max_lin_acc,
                          Eigen::Vector3d gravity,
                          double mass,
                          Eigen::Matrix3d inertia);

    Eigen::Vector3d calculate_force(const X &x, const X &xdot, const X &xref) override;
    Eigen::Vector3d calculate_moments(const X &x, const X &xdot, const X &xref,
                                      const Eigen::Vector3d &force) override;

private:
    // Gains
    Eigen::Vector3d lambda_pos_, kappa_pos_, boundary_pos_; ///< lambda, kappa, and boundary gains for position
    Eigen::Vector3d lambda_att_, kappa_att_, boundary_att_; ///< lambda, kappa, and boundary gains for attitude

    // System constraints
    Eigen::Vector3d max_lin_acc_; ///< maximum linear acceleration
    Eigen::Vector3d gravity_;     ///< gravity vector
    double mass_;                 ///< mass of model
    Eigen::Matrix3d inertia_;     ///< inertia of model

    /**
     * @brief Saturation function for determining if within boundary
     *
     * @param s current state
     * @param boundary boundary condition
     * @return Eigen::Vector3d saturated state
     */
    Eigen::Vector3d saturation(const Eigen::Vector3d &s, const Eigen::Vector3d &boundary);
    /**
     * @brief Get the skew-symmetric map of a vector
     *
     * @param v vector
     * @return Eigen::Matrix3d  skew symmetric map
     */
    Eigen::Matrix3d skew(const Eigen::Vector3d &v);
};

#endif