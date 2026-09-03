// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Amal Dev Haridevan

#include "controllers/pid_controller.hh"
#include <cmath>

PIDController::PIDController(const Eigen::Vector4d& kp,     // Position Gain
                    const Eigen::Vector4d& kd,          // Derivative Gain
                    const Eigen::Vector4d& ki,          // Integral Gain
                    const Eigen::Vector3d& kp_att,      // Attitude P gain
                    const Eigen::Vector3d& kd_att,      // Attitude D gain
                    const Eigen::Vector3d& kp_omega,    // Angular rate P gain
                    const Eigen::Vector3d& kd_omega,    // Angular rate D gain
                    const double& omega_n,               // w_n for second order low pass filter (for outer loop commands)
                    const double& zeta,                  // zeta for second order low pass filyter (for outer loop commands)
                    const double& dt,                    // sampling time (for integral and filter)
                    const Eigen::Vector4d& integral_sat_limit,  // Saturation for integral term [post gain multiplication]
                    const Eigen::Vector4d& stabilization_command_sat_limit, // Saturation limit for stabilization command
                    const Eigen::Vector3d& angle_cmd_limit,     // maximum angle commands from outer-loop
                    const Eigen::Vector3d& omega_cmd_limit,     // maximum omega cmd from pd controller for attitude
                    const Eigen::Vector3d& max_torque,
                    const double& mass,
                    const double& gravity_mag):
                    kp_(kp), kd_(kd), ki_(ki),
                    kp_att_(kp_att), kd_att_(kd_att), 
                    kp_omega_(kp_omega), kd_omega_(kd_omega), 
                    omega_n_(omega_n), zeta_(zeta), dt_(dt),
                    integral_sat_limit_(integral_sat_limit), stabilization_command_sat_limit_(stabilization_command_sat_limit), angle_cmd_limit_(angle_cmd_limit),
                    omega_cmd_limit_(omega_cmd_limit), max_torque_(max_torque),
                     mass_(mass), gravity_mag_(gravity_mag){        // Maximum torque commands
                    integral_error_.setZero();
                    inner_loop_cmd_.setZero();
                    mapping_matrix_ <<0.0, 0.0, 1.0, 0.0, 
                                          0.0, -1.0, 0.0, 0.0,
                                          1.0, 0.0, 0.0, 0.0,
                                          0.0, 0.0, 0.0, 1.0  ;
                    filter_ = std::make_unique<sitl::SecondOrderLPFilter<2, double>>(omega_n, zeta, dt);
                    filter_alpha_ = std::make_unique<sitl::SecondOrderLPFilter<3, double>>(omega_n, zeta, dt);
                    }
    
    Eigen::Vector3d PIDController::calculate_force(const X& state, 
                                    const X& state_dot,
                                    const X& desired_state){
                const auto& pos = state.segment<3>(0);
                const auto& vel = state.segment<3>(3);
                const auto& pos_d = desired_state.segment<3>(0);
                const auto& vel_d = desired_state.segment<3>(3);

                const Eigen::Quaterniond q(state(6), state(7), state(8), state(9));
                const Eigen::Matrix3d R = q.toRotationMatrix();
                Eigen::Quaterniond q_d(desired_state(6), desired_state(7), desired_state(8), desired_state(9));
                q_d.normalize();
                const Eigen::Matrix3d R_d = q_d.toRotationMatrix();

                // Extract Euler angles (ZYX intrinsic) using atan2 to avoid ambiguity
                // ypr = [yaw, pitch, roll]
                // This gives unique angles for pitch in [-pi/2, pi/2]
                auto extractYPR = [](const Eigen::Matrix3d& R_mat) -> Eigen::Vector3d {
                    Eigen::Vector3d ypr;
                    // pitch (rotation about Y)
                    double sp = -R_mat(2, 0);
                    sp = std::max(-1.0, std::min(1.0, sp)); // clamp for numerical stability
                    ypr(1) = std::asin(sp);
                    
                    double cp = std::cos(ypr(1));
                    if (std::abs(cp) > 1e-6) {
                        // yaw (rotation about Z)
                        ypr(0) = std::atan2(R_mat(1, 0), R_mat(0, 0));
                        // roll (rotation about X)
                        ypr(2) = std::atan2(R_mat(2, 1), R_mat(2, 2));
                    } else {
                        // Gimbal lock: pitch = ±90°
                        ypr(0) = std::atan2(-R_mat(0, 1), R_mat(1, 1));
                        ypr(2) = 0.0;
                    }
                    return ypr;
                };
                
                Eigen::Vector3d ypr = extractYPR(R);
                Eigen::Vector3d ypr_d = extractYPR(R_d);

                const double& yaw_rate_d = desired_state(12);
                // Compute yaw rate (euler angles rate)
                double sp = std::sin(ypr(1));
                double cp = std::cos(ypr(1));
                double sr = std::sin(ypr(2));
                double cr = std::cos(ypr(2));
                
                double yaw_rate = 0.0;
                // Check for singularity at pitch = +/- 90 degrees
                // (cp will be close to zero, causing division by zero with tan or sec)
                if (std::abs(cp) < 1e-6) {
                    std::cerr << "Warning: Pitch angle is near singularity (+/- 90 degrees). Euler rates are ill-defined." << std::endl;
                   
                }
                else{
                // Transformation matrix T (or W^-1)
                Eigen::Matrix3d T;
                T << 1, sr * sp / cp, cr * sp / cp,
                    0, cr, -sr,
                    0, sr / cp, cr / cp;
                
                // An alternative way to write T for clarity using tan/sec formulas
                // T << 1, sr * std::tan(pitch), cr * std::tan(pitch),
                //      0, cr, -sr,
                //      0, sr * (1.0 / cp), cr * (1.0 / cp);

                // Calculate the Euler rates
                Eigen::Vector3d euler_rates = T * state.segment<3>(10);
                yaw_rate = euler_rates(2);
                }
                
                // Create AngleAxis objects for each rotation
                Eigen::AngleAxisd roll(0.0, Eigen::Vector3d::UnitX());
                Eigen::AngleAxisd pitch(0.0, Eigen::Vector3d::UnitY());
                Eigen::AngleAxisd yaw(ypr(0), Eigen::Vector3d::UnitZ());
                
                Eigen::Quaterniond  q_h = yaw*roll*pitch;
                Eigen::Matrix3d R_h = q_h.toRotationMatrix();
                Eigen::Vector3d hf_pos = R_h.transpose()*pos;
                Eigen::Vector3d hf_vel = R_h.transpose()*vel;
                Eigen::Vector3d hf_pos_d = R_h.transpose()*pos_d;
                Eigen::Vector3d hf_vel_d = R_h.transpose()*vel_d;
                
                Eigen::Vector4d ref(hf_pos_d(0), hf_pos_d(1), hf_pos_d(2), ypr_d(0));
                Eigen::Vector4d curr(hf_pos(0), hf_pos(1), hf_pos(2), ypr(0));
                Eigen::Vector4d ref_v(hf_vel_d(0), hf_vel_d(1), hf_vel_d(2), yaw_rate_d);
                Eigen::Vector4d curr_v(hf_vel(0), hf_vel(1), hf_vel(2), yaw_rate);
                Eigen::Vector4d hf_err = ref - curr;
                Eigen::Vector4d vel_err = ref_v - curr_v;
                integral_error_ += hf_err*dt_;
                
                Eigen::Vector4d cmd =  kp_.cwiseProduct(hf_err) + kd_.cwiseProduct(vel_err) + (ki_.cwiseProduct(integral_error_)).cwiseMin(integral_sat_limit_).cwiseMax(-integral_sat_limit_);
                cmd.segment<2>(0) = filter_->Update(cmd.segment<2>(0));
                cmd = cmd.cwiseMin(stabilization_command_sat_limit_).cwiseMax(-stabilization_command_sat_limit_);
                cmd = mapping_matrix_*cmd;
                cmd(0) += std::abs(mass_*gravity_mag_);
                cmd(0) = std::min(std::max(cmd(0), 0.0 ), 58.0);
                inner_loop_cmd_ = cmd;
                Eigen::Vector3d thrust_b = Eigen::Vector3d(0.0, 0.0, cmd(0));
                return R*thrust_b; // Inertial frame force, aligned along body-frame z-axis

}
    
    Eigen::Vector3d PIDController::calculate_moments(const X& state, 
                                    const X& state_dot,
                                    const X& desired_state,
                                    const Eigen::Vector3d&  control_thrust){
                                
                Eigen::Vector3d angle_cmd = inner_loop_cmd_.segment<3>(1);
                angle_cmd = angle_cmd.cwiseMax(-angle_cmd_limit_).cwiseMin(angle_cmd_limit_);
                const Eigen::Quaterniond q(state(6), state(7), state(8), state(9));
                const Eigen::Matrix3d R = q.toRotationMatrix();
                
                // Extract Euler angles using atan2 to avoid ambiguity (same as in calculate_force)
                double sp = -R(2, 0);
                sp = std::max(-1.0, std::min(1.0, sp));
                double pitch = std::asin(sp);
                double roll = 0.0;
                double cp = std::cos(pitch);
                if (std::abs(cp) > 1e-6) {
                    roll = std::atan2(R(2, 1), R(2, 2));
                }
                
                Eigen::Vector3d rpy(roll, pitch, 0.0);
                Eigen::Vector3d angle_err = angle_cmd - rpy;
                Eigen::Vector3d omega = state.segment<3>(10);
                Eigen::Vector3d alpha = state_dot.segment<3>(10);
                alpha = filter_alpha_->Update(alpha);

                Eigen::Vector3d omega_d = kp_att_.cwiseProduct(angle_err)  - kd_att_.cwiseProduct(omega);
                omega_d = omega_d.cwiseMax(-omega_cmd_limit_).cwiseMin(omega_cmd_limit_);
                Eigen::Vector3d omega_err = omega_d - omega;
                Eigen::Vector3d torque = kp_omega_.cwiseProduct(omega_err) - kd_omega_.cwiseProduct(alpha);
                torque = torque.cwiseMax(-max_torque_).cwiseMin(max_torque_);
                return torque;
}
    