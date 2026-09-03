// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Amal Dev Haridevan

#include "controllers/inner_loop_controller.hh"
#include "print_utils.hh"

using namespace uav_controllers;

inline double wrap_angle(double a) {
    while (a > M_PI)  a -= 2.0*M_PI;
    while (a <= -M_PI) a += 2.0*M_PI;
    return a;
}


InnerLoopController::InnerLoopController(const Gain& kp_angle,
                const Gain& kd_angle,
                const Gain& kp_angular_rate,
                const Gain& kd_angular_rate,
                const double mass,
                const Eigen::Matrix3d& inertia,
                const Eigen::Vector3d& gravity) :
                    kp_angle_(kp_angle),
                    kd_angle_(kd_angle),
                    kp_angular_rate_(kp_angular_rate),
                    kd_angular_rate_(kd_angular_rate),
                    mass_(mass),
                    inertia_(inertia),
                    gravity_(gravity), g_(gravity.norm())
                {
                // set the force cmd to angle conversion matrix
                // The intuition behind this matrix is:
                // Fz -> thrust , assigned to V[0]
                // Fx -> pitch angle (positive Fx -> nose down -> positive pitch), assigned to V[2]
                // Fy -> roll angle (positive Fy -> left wing down -> negative roll), assigned to V[1]
                // yaw_rate -> yaw rate, assigned to V[3]
                force_to_thrust_angles<<0, 0, 1, 0,
                                0, -1, 0, 0,
                                1, 0, 0, 0;
                // set the saturation limits
                angle_cmd_max_ << 0.7853981633974483, 0.7853981633974483, 1.7453292519943295; // 45 deg roll/pitch, 100 deg/s yaw rate
                angular_rate_cmd_max_ << 10.47197551, 10.47197551, 10.47197551; // 600 deg/s roll/pitch/yaw rate
                final_cmd_max_ << 1.0915,0.8984, 0.0984; // The moments saturation limits
            }

Eigen::Vector3d InnerLoopController::calculate_force(const X& state,
            const X& state_dot,
            const X& desired){ 
            // For inner loop, the force command is directly from outer loop
            throw std::runtime_error("InnerLoopController::calculate_force not implemented. Force command should come directly from outer loop.");
            }

Eigen::Vector3d InnerLoopController::calculate_moments(const X& state,
            const X& state_dot,
            const X& desired,
            const Eigen::Vector3d&  control_thrust){ 
            Eigen::Vector3d cmd_vec = force_to_thrust_angles * control_thrust;
            // Extract desired angles and yaw rate
            Eigen::Vector3d attitude_cmd;
            attitude_cmd << cmd_vec(1), cmd_vec(2), 0.0; // yaw is not controlled here
            // Update throttle
            double throttle = cmd_vec(0) + mass_ * g_;
            // clamp throottle between 0 and 60.0
            throttle = std::clamp(throttle, 0.0, 60.0);
            // Clamp the attitude commands
            attitude_cmd = attitude_cmd.cwiseMax(-angle_cmd_max_).cwiseMin(angle_cmd_max_);
            // determine current attitude and angular rates
            Eigen::Quaterniond q(state(6), state(7), state(8), state(9));
            q.normalize();
            Eigen::Matrix3d R = q.toRotationMatrix();
            latest_rotmat_ = R;
            Eigen::Vector3d w = state.segment<3>(10);
            Eigen::Vector3d w_d = desired.segment<3>(10);
            Eigen::Vector3d alpha = state_dot.segment<3>(10);
            // Extract current roll, pitch, yaw from R
            double roll = std::atan2(R(2,1), R(2,2));
            double pitch = std::asin(-R(2,0));
            double yaw = wrap_angle(std::atan2(R(1,0), R(0,0)));
            Eigen::Vector3d euler_angles;
            euler_angles << roll, pitch, yaw; // yaw is not controlled here
            // Compute attitude errors
            Eigen::Vector3d attitude_error = attitude_cmd - euler_angles ;
            // Desired angular rates from attitude PD control
            Eigen::Vector3d angular_rate_cmd = attitude_error.cwiseProduct(kp_angle_)
                                                + (w_d - w).cwiseProduct(kd_angle_);
            // Clamp angular rate commands
            angular_rate_cmd = angular_rate_cmd.cwiseMax(-angular_rate_cmd_max_).cwiseMin(angular_rate_cmd_max_);
            // Compute angular rate errors
            Eigen::Vector3d angular_rate_error = angular_rate_cmd - w;
            // Compute moments from angular rate PD control
            Eigen::Vector3d moments_cmd = angular_rate_error.cwiseProduct(kp_angular_rate_)
                                                - alpha.cwiseProduct(kd_angular_rate_);
            // Clamp final moments command
            moments_cmd = moments_cmd.cwiseMax(-final_cmd_max_).cwiseMin(final_cmd_max_);
            // Construct final command
            Eigen::Vector3d u_cmd;
            u_cmd << moments_cmd(0), moments_cmd(1), moments_cmd(2);
            return u_cmd;   
            }
