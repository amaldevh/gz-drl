// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Amal Dev Haridevan

#include "controllers/smc_controller.hh"



SlidingModeController::SlidingModeController(Eigen::Vector3d lambda_pos,
                                             Eigen::Vector3d kappa_pos,
                                             Eigen::Vector3d lambda_att,
                                             Eigen::Vector3d kappa_att,
                                             Eigen::Vector3d boundary_pos,
                                             Eigen::Vector3d boundary_att,
                                             Eigen::Vector3d max_lin_acc,
                                             Eigen::Vector3d gravity,
                                             double mass,
                                             Eigen::Matrix3d inertia)
  : lambda_pos_(std::move(lambda_pos)), kappa_pos_(std::move(kappa_pos)),
    lambda_att_(std::move(lambda_att)), kappa_att_(std::move(kappa_att)),
    boundary_pos_(std::move(boundary_pos)), boundary_att_(std::move(boundary_att)),
    max_lin_acc_(std::move(max_lin_acc)), gravity_(std::move(gravity)),
    mass_(mass), inertia_(std::move(inertia)) {}

// Helper: Vectorized Saturation function (replaces signum to reduce chattering)
// returns val/boundary clamped between -1 and 1
Eigen::Vector3d SlidingModeController::saturation(const Eigen::Vector3d& s, const Eigen::Vector3d& boundary) {
    Eigen::Vector3d out;
    for (int i = 0; i < 3; ++i) {
        double val = s(i) / (boundary(i) + 1e-6); // Avoid div by zero
        out(i) = std::clamp(val, -1.0, 1.0);
    }
    return out;
}

// Helper: Skew symmetric matrix
Eigen::Matrix3d SlidingModeController::skew(const Eigen::Vector3d& v) {
    Eigen::Matrix3d m;
    m << 0, -v(2), v(1),
         v(2), 0, -v(0),
         -v(1), v(0), 0;
    return m;
}

Eigen::Vector3d SlidingModeController::calculate_force(const X& x, const X& xdot, const X& xref) {
    // Unpack state
    const Eigen::Vector3d p = x.segment<3>(0);
    const Eigen::Vector3d v = x.segment<3>(3);
    
    const Eigen::Vector3d p_d = xref.segment<3>(0);
    const Eigen::Vector3d v_d = xref.segment<3>(3);

    // Position and velocity errors
    Eigen::Vector3d e_p = p - p_d;
    Eigen::Vector3d e_v = v - v_d;

    // Sliding surface: s = e_v + lambda * e_p
    Eigen::Vector3d s_pos = e_v + e_p.cwiseProduct(lambda_pos_);
    
    // Saturation function for robustness
    Eigen::Vector3d sat_term = saturation(s_pos, boundary_pos_);
    
    // Control law (exactly matching geometric controller structure):
    // a_cmd = (Kp*e_p + Kd*e_v + robust_term)/m
    // where Kp = lambda*kappa, Kd = kappa
    Eigen::Vector3d Kp = lambda_pos_.cwiseProduct(kappa_pos_);
    Eigen::Vector3d Kd = kappa_pos_;
    
    Eigen::Vector3d a_cmd = e_p.cwiseProduct(Kp) + e_v.cwiseProduct(Kd) + sat_term.cwiseProduct(kappa_pos_);
    a_cmd /= mass_;
    
    // Clamp and add gravity (exactly like geometric controller)
    a_cmd = a_cmd.cwiseMax(-max_lin_acc_).cwiseMin(max_lin_acc_);
    a_cmd += gravity_;
    
    return -mass_ * a_cmd;
}
Eigen::Vector3d SlidingModeController::calculate_moments(const X& x, 
    const X& xdot, const X& xref, const Eigen::Vector3d& force) {

    // Unpack state
    Eigen::Quaterniond q(x(6), x(7), x(8), x(9));
    q.normalize();
    const Eigen::Matrix3d R = q.toRotationMatrix();
    const Eigen::Vector3d w = x.segment<3>(10);

    Eigen::Quaterniond qd(xref(6), xref(7), xref(8), xref(9));
    qd.normalize();
    const Eigen::Vector3d w_d = xref.segment<3>(10);
    
    // Desired attitude from thrust vector
    Eigen::Vector3d b3d = force.normalized();
    
    const double yaw = std::atan2(2.0 * (qd.w()*qd.z() + qd.x()*qd.y()),
                                  1.0 - 2.0 * (qd.y()*qd.y() + qd.z()*qd.z()));
    Eigen::Vector3d b1d(std::cos(yaw), std::sin(yaw), 0.0);

    if ((b1d.cross(b3d)).squaredNorm() < 1e-12) {
        b1d = R.col(1);
    }

    Eigen::Vector3d b2d = b3d.cross(b1d).normalized();
    Eigen::Matrix3d Rd;
    Rd.col(0) = b2d.cross(b3d);
    Rd.col(1) = b2d;
    Rd.col(2) = b3d;

    // Attitude errors (matching geometric controller)
    Eigen::Matrix3d RtRd = R.transpose() * Rd;
    Eigen::Matrix3d Rdtr = Rd.transpose() * R;
    Eigen::Matrix3d errm = Rdtr - RtRd;
    Eigen::Vector3d eR(errm(2,1), errm(0,2), errm(1,0));
    eR = 0.5 * eR;
    
    Eigen::Vector3d eW = w - RtRd * w_d;

    // Sliding surface
    Eigen::Vector3d s_att = eW + eR.cwiseProduct(lambda_att_);
    Eigen::Vector3d sat_term = saturation(s_att, boundary_att_);

    // Control moments (matching geometric controller structure)
    // Kp_att = lambda*kappa, Kd_att = kappa
    Eigen::Vector3d Kp_att = lambda_att_.cwiseProduct(kappa_att_);
    
    Eigen::Vector3d M = - eR.cwiseProduct(Kp_att)
                        - eW.cwiseProduct(kappa_att_)
                        - sat_term.cwiseProduct(kappa_att_)
                        + w.cross(inertia_ * w)
                        - inertia_ * (skew(w) * RtRd * w_d);

    return M;
}