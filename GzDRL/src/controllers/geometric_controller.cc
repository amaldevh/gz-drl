// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Amal Dev Haridevan

#include "controllers/geometric_controller.hh"

using namespace uav_controllers;

// ---------- GeometricController ----------
GeometricController::GeometricController(Gain p_gain,
                                         Gain d_gain,
                                         Gain att_p_gain,
                                         Gain att_d_gain,
                                         Eigen::Vector3d max_lin_acc,
                                         Eigen::Vector3d gravity,
                                         double mass,
                                         Eigen::Matrix3d inertia)
  : p_gain_(p_gain), d_gain_(d_gain),
    att_p_gain_(att_p_gain), att_d_gain_(att_d_gain),
    stab_out_hi_(1.0915, 1.0915, 0.5),
    stab_out_lo_(-1.0915,-1.0915,-0.5),
    max_lin_acc_(std::move(max_lin_acc)),
    gravity_(std::move(gravity)),
    mass_(mass), inertia_(std::move(inertia)){}

Eigen::Vector3d GeometricController::calculate_force(const X& x,
                                                    const X& xdot,
                                                    const X& xref)
{
  // Unpack state (map avoids temporaries)
  const Eigen::Vector3d p   = x.segment<3>(0);
  const Eigen::Vector3d v   = x.segment<3>(3);
  Eigen::Quaterniond q(x(6), x(7), x(8), x(9));
  q.normalize();
  const Eigen::Matrix3d R   = q.toRotationMatrix();
  const Eigen::Vector3d w   = x.segment<3>(10);
  const Eigen::Vector3d a   = xdot.segment<3>(3);
  const Eigen::Vector3d wdot= xdot.segment<3>(10);

  const Eigen::Vector3d p_d = xref.segment<3>(0);
  const Eigen::Vector3d v_d = xref.segment<3>(3);
  Eigen::Quaterniond qd(xref(6), xref(7), xref(8), xref(9));
  qd.normalize();
  const Eigen::Matrix3d Rd = qd.toRotationMatrix();
  const Eigen::Vector3d w_d = xref.segment<3>(10);

  // Position PD -> desired accel in world frame
  Eigen::Vector3d a_cmd = (p - p_d).cwiseProduct(p_gain_) + (v - v_d).cwiseProduct(d_gain_);
  a_cmd /= mass_;
  // clamp |a_cmd|
  a_cmd = a_cmd.cwiseMax(-max_lin_acc_).cwiseMin(max_lin_acc_);
  a_cmd += gravity_;
  return -mass_ * a_cmd; // desired force
}
Eigen::Vector3d GeometricController::calculate_moments(const X& x,
                                                       const X& xdot,
                                                       const X& xref,
                                                       const Eigen::Vector3d&  control_thrust){
  // Unpack state (map avoids temporaries)
  const Eigen::Vector3d p   = x.segment<3>(0);
  const Eigen::Vector3d v   = x.segment<3>(3);
  Eigen::Quaterniond q(x(6), x(7), x(8), x(9));
  q.normalize();
  const Eigen::Matrix3d R   = q.toRotationMatrix();
  Eigen::Quaterniond qd(xref(6), xref(7), xref(8), xref(9));
  qd.normalize();
  const Eigen::Vector3d w_d = xref.segment<3>(10);
  const Eigen::Vector3d w   = x.segment<3>(10);
  // Desired body z-axis
  Eigen::Vector3d b3d = control_thrust; 
  const double n_b3 = b3d.norm();
  if (n_b3 > 1e-9) b3d /= n_b3; else b3d = Eigen::Vector3d::UnitZ();

  // Desired yaw from qd (fast path using quaternion -> yaw)
  const double yaw = std::atan2(2.0 * (qd.w()*qd.z() + qd.x()*qd.y()),
                                1.0 - 2.0 * (qd.y()*qd.y() + qd.z()*qd.z()));
  Eigen::Vector3d b1d(std::cos(yaw), std::sin(yaw), 0.0);

  // Ensure b1d not collinear with b3d
  if ((b1d.cross(b3d)).squaredNorm() < 1e-12) {
    b1d = R.col(1); // fallback to current b2
    if ((b1d.cross(b3d)).squaredNorm() < 1e-12)
      b1d = R.col(2); // fallback to current b3
  }

  // Orthonormal desired frame
  Eigen::Vector3d b2d = b3d.cross(b1d).normalized();
  Eigen::Matrix3d Rd_b;
  Rd_b.col(0) = b2d.cross(b3d);
  Rd_b.col(1) = b2d;
  Rd_b.col(2) = b3d;

  // Attitude error eR = 0.5 vee(RdᵀR - RᵀRd)
  const Eigen::Matrix3d RtRd = R.transpose() * Rd_b;
  const Eigen::Matrix3d Rdtr = Rd_b.transpose() * R;
  const Eigen::Matrix3d errm = Rdtr - RtRd;
  const Eigen::Vector3d eR(errm(2,1), errm(0,2), errm(1,0));

  // Body-rate error eW = w - RᵀRd w_d
  const Eigen::Vector3d eW = w - RtRd * w_d;

  // Moments (with simple coriolis term)
  Eigen::Vector3d M = - eR.cwiseProduct(att_p_gain_)
                      - eW.cwiseProduct(att_d_gain_)
                      + w.cross(inertia_ * w)
                      - inertia_ * (skew(w) * RtRd * w_d);
  // Clip moments if desired (kept conservative)
  M = M.cwiseMax(stab_out_lo_).cwiseMin(stab_out_hi_);

  return M;
}



