// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Amal Dev Haridevan

#ifndef PID_CONTROLLER_HH
#define PID_CONTROLLER_HH

#include "uav_controllers.hh"
#include <Eigen/Dense>
#include "second_order_lp_filter.hh"

using namespace uav_controllers;

class PIDController final : public UAVController
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    /**
     * @brief Construct a new PIDController object
     *
     * @param kp Position gain (proportional term)
     * @param kd Derivative gain
     * @param ki Integral term
     * @param kp_att Attitude P term
     * @param kd_att Attitude D term
     * @param kp_omega Omega P term (Inner-loop)
     * @param kd_omega Omgea D term (Inner-loop)
     * @param omega_n Second order rate filter natural frequency (for filtering outer-loop)
     * @param zeta  Second order rate filter damping ratio (for filtering outer-loop)
     * @param dt    Seconf order rate filter time step (for filtering outer-loop)
     * @param integral_sat_limit Saturation limit for integral term of outer-loop
     * @param stabilization_command_sat_limit Saturation limit for inner-loop
     * @param angle_cmd_limit Maximum attitude cmd such that |(kp*e_p + kd*derivative(e_p) + ki*integral(e_p)) |< angle_cmd_limit
     * @param omega_cmd_limit Maximum omega cmd such that |kp_att*e_a + kd_att*derivative(e_a)| < omega_cmd_limit
     * @param max_torque    Maximum torque such that |kp_omega * e_w + kd_omega*derivative(e_w)| < max_torque
     * @param mass mass of the model
     * @param gravity_mag Gravity magnitude
     */
    PIDController(const Eigen::Vector4d &kp,                              // Position Gain
                  const Eigen::Vector4d &kd,                              // Derivative Gain
                  const Eigen::Vector4d &ki,                              // Integral Gain
                  const Eigen::Vector3d &kp_att,                          // Attitude P gain
                  const Eigen::Vector3d &kd_att,                          // Attitude D gain
                  const Eigen::Vector3d &kp_omega,                        // Angular rate P gain
                  const Eigen::Vector3d &kd_omega,                        // Angular rate D gain
                  const double &omega_n,                                  // w_n for second order low pass filter (for outer loop commands)
                  const double &zeta,                                     // zeta for second order low pass filyter (for outer loop commands)
                  const double &dt,                                       // sampling time (for integral and filter)
                  const Eigen::Vector4d &integral_sat_limit,              // Saturation for integral term [post gain multiplication]
                  const Eigen::Vector4d &stabilization_command_sat_limit, // Saturation limit for stabilization command
                  const Eigen::Vector3d &angle_cmd_limit,                 // maximum angle commands from outer-loop
                  const Eigen::Vector3d &omega_cmd_limit,                 // maximum omega cmd from pd controller for attitude
                  const Eigen::Vector3d &max_torque,
                  const double &mass,
                  const double &gravity_mag); // Maximum torque commands

    Eigen::Vector3d calculate_force(const X &state,
                                    const X &state_dot,
                                    const X &desired_state) override;

    Eigen::Vector3d calculate_moments(const X &state,
                                      const X &state_dot,
                                      const X &desired_state,
                                      const Eigen::Vector3d &control_thrust) override;

private:
    const Eigen::Vector4d kp_;                                           ///< Position Gain
    const Eigen::Vector4d kd_;                                           ///< Derivative Gain
    const Eigen::Vector4d ki_;                                           ///< Integral Gain
    const Eigen::Vector3d kp_att_;                                       ///< Attitude P gain
    const Eigen::Vector3d kd_att_;                                       ///< Attitude D gain
    const Eigen::Vector3d kp_omega_;                                     ///< Angular rate P gain
    const Eigen::Vector3d kd_omega_;                                     ///< Angular rate D gain
    const double omega_n_;                                               ///< w_n for second order low pass filter (for outer loop commands)
    const double zeta_;                                                  ///< zeta for second order low pass filyter (for outer loop commands)
    const double dt_;                                                    ///< sampling time (for integral and filter)
    const Eigen::Vector4d integral_sat_limit_;                           ///< Saturation for integral term [post gain multiplication]
    const Eigen::Vector4d stabilization_command_sat_limit_;              ///< Saturation limit for stabilization command
    const Eigen::Vector3d angle_cmd_limit_;                              ///< maximum angle commands from outer-loop
    const Eigen::Vector3d omega_cmd_limit_;                              ///< maximum omega cmd from pd controller for attitude
    const Eigen::Vector3d max_torque_;                                   ///< Maximum torque commands
    std::unique_ptr<sitl::SecondOrderLPFilter<2, double>> filter_;       ///< filter for outer-loop
    std::unique_ptr<sitl::SecondOrderLPFilter<3, double>> filter_alpha_; ///< alpha filter
    Eigen::Vector4d integral_error_;                                     ///< integral term error
    Eigen::Vector4d inner_loop_cmd_;                                     ///< Store inertial force generated during calculate_force
    const double mass_;                                                  ///< mass of model
    const double gravity_mag_;                                           ///< gravity magnitude
    Eigen::Matrix4d mapping_matrix_;                                     ///< simple mapping matrix
};
#endif