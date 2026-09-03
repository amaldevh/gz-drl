// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Amal Dev Haridevan

#ifndef TRAJECTORY_TRACKING_LL_PROCESSOR_HH_
#define TRAJECTORY_TRACKING_LL_PROCESSOR_HH_

#include "processors/trajectory_tracking_processor.hh"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

/** Trajectory-tracking processor for direct four-rotor-velocity control. */
template <typename EnvSpec>
class TrajectoryTrackingLLProcessor
    : public TrajectoryTrackingProcessor<EnvSpec, 4>
{
public:
    using Base = TrajectoryTrackingProcessor<EnvSpec, 4>;

    TrajectoryTrackingLLProcessor(
        std::string state_key,
        bool use_rotation_matrix,
        int state_history_len,
        int future_waypoint_num,
        float minimum_rotor_velocity,
        float maximum_rotor_velocity,
        int physics_steps_per_control,
        float physics_dt,
        float horizontal_amplitude_min,
        float horizontal_amplitude_max,
        float vertical_amplitude_min,
        float vertical_amplitude_max,
        float height_offset_min,
        float height_offset_max,
        float trajectory_speed_min,
        float trajectory_speed_max,
        float maximum_normal_acceleration,
        float vertical_position_weight,
        float vertical_velocity_weight,
        float fixed_zero_yaw_probability,
        int trajectory_sampling_attempts,
        std::uint32_t trajectory_seed)
        : Base(
              std::move(state_key),
              use_rotation_matrix,
              state_history_len,
              future_waypoint_num,
              RotorScale(minimum_rotor_velocity, maximum_rotor_velocity),
              RotorBias(minimum_rotor_velocity, maximum_rotor_velocity),
              physics_steps_per_control,
              physics_dt,
              horizontal_amplitude_min,
              horizontal_amplitude_max,
              vertical_amplitude_min,
              vertical_amplitude_max,
              height_offset_min,
              height_offset_max,
              trajectory_speed_min,
              trajectory_speed_max,
              maximum_normal_acceleration,
              vertical_position_weight,
              vertical_velocity_weight,
              fixed_zero_yaw_probability,
              trajectory_sampling_attempts,
              trajectory_seed)
    {
    }

private:
    static void ValidateRotorBounds(float minimum, float maximum)
    {
        if (!std::isfinite(minimum) || !std::isfinite(maximum) ||
            minimum < 0.0f || maximum <= minimum)
        {
            throw std::invalid_argument(
                "Rotor velocity bounds must be finite and satisfy "
                "0 <= minimum < maximum.");
        }
    }

    static std::vector<float> RotorScale(float minimum, float maximum)
    {
        ValidateRotorBounds(minimum, maximum);
        return std::vector<float>(4, 0.5f * (maximum - minimum));
    }

    static std::vector<float> RotorBias(float minimum, float maximum)
    {
        ValidateRotorBounds(minimum, maximum);
        return std::vector<float>(4, 0.5f * (maximum + minimum));
    }
};

#endif // TRAJECTORY_TRACKING_LL_PROCESSOR_HH_
