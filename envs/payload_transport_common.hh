// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Amal Dev Haridevan

#ifndef PAYLOAD_TRANSPORT_COMMON_HH_
#define PAYLOAD_TRANSPORT_COMMON_HH_

#include "common_utils.hh"

#include <algorithm>
#include <cmath>

namespace payload_transport_common
{
inline constexpr double kMaximumPosition = 7.0;
inline constexpr double kMaximumTilt = 1.57;
inline constexpr double kMaximumYaw = 1.57;
inline constexpr float kFailurePenalty = 50.0f;

inline bool IsFailureState(const Statef &state) noexcept
{
    const Eigen::Vector3f position = state.head<3>();
    if (!position.allFinite() ||
        (position.cwiseAbs().array() > kMaximumPosition).any())
    {
        return true;
    }

    Eigen::Quaternionf quaternion(
        state(6), state(7), state(8), state(9));
    const float quaternion_norm = quaternion.norm();
    if (!std::isfinite(quaternion_norm) || quaternion_norm <= 1.0e-6f)
    {
        return true;
    }

    const Eigen::Matrix3f rotation =
        quaternion.normalized().toRotationMatrix();
    const float tilt = std::acos(std::clamp(
        rotation.col(2).dot(Eigen::Vector3f::UnitZ()), -1.0f, 1.0f));
    const float yaw = std::atan2(rotation(1, 0), rotation(0, 0));
    return !std::isfinite(tilt) || !std::isfinite(yaw) ||
           tilt > kMaximumTilt || std::abs(yaw) > kMaximumYaw;
}
} // namespace payload_transport_common

#endif // PAYLOAD_TRANSPORT_COMMON_HH_
