// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Amal Dev Haridevan

#ifndef COMMON_TRAJECTORY_TRACKING_REFERENCE_HH
#define COMMON_TRAJECTORY_TRACKING_REFERENCE_HH

#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Dense>

/** Continuous Lissajous reference used by the trajectory-tracking tasks.
 *
 * The returned state has the Gazebo UAV layout
 * [position(3), velocity(3), quaternion(wxyz), angular_velocity(3)].
 */
class TrajectoryTrackingReference
{
public:
    TrajectoryTrackingReference(
        std::vector<float> amplitudes,
        std::vector<float> frequencies,
        std::vector<float> phases,
        float height_offset,
        float control_dt,
        bool fixed_zero_yaw = false)
        : amplitudes_(std::move(amplitudes)),
          frequencies_(std::move(frequencies)),
          phases_(std::move(phases)),
          height_offset_(height_offset),
          control_dt_(control_dt),
          fixed_zero_yaw_(fixed_zero_yaw)
    {
        Validate();
    }

    void UpdateCurve(
        const std::vector<float> &amplitudes,
        const std::vector<float> &frequencies,
        const std::vector<float> &phases,
        float height_offset,
        bool fixed_zero_yaw)
    {
        amplitudes_ = amplitudes;
        frequencies_ = frequencies;
        phases_ = phases;
        height_offset_ = height_offset;
        fixed_zero_yaw_ = fixed_zero_yaw;
        Validate();
    }

    Eigen::Matrix<float, 13, 1> State(float time) const
    {
        Eigen::Matrix<float, 13, 1> reference;
        for (int axis = 0; axis < 3; ++axis)
        {
            const float angle =
                frequencies_[axis] * time + phases_[axis];
            reference(axis) = amplitudes_[axis] * std::sin(angle);
            reference(3 + axis) = amplitudes_[axis] *
                                  frequencies_[axis] * std::cos(angle);
        }
        reference(2) += height_offset_;

        const float yaw = fixed_zero_yaw_
                              ? 0.0f
                              : std::atan2(reference(4), reference(3));
        const float half_yaw = 0.5f * yaw;
        reference(6) = std::cos(half_yaw);
        reference(7) = 0.0f;
        reference(8) = 0.0f;
        reference(9) = std::sin(half_yaw);
        reference.template segment<3>(10).setZero();
        return reference;
    }

    Eigen::MatrixXf Errors(
        const Eigen::Matrix<float, 13, 1> &state,
        float time,
        int horizon,
        bool include_rotation) const
    {
        const int feature_count = include_rotation ? 15 : 6;
        Eigen::MatrixXf errors(horizon, feature_count);
        const Eigen::Vector3f position = state.template head<3>();
        const Eigen::Vector3f velocity = state.template segment<3>(3);

        for (int step = 0; step < horizon; ++step)
        {
            const auto reference = State(time + step * control_dt_);
            errors.row(step).segment<3>(0) =
                (reference.template head<3>() - position).transpose();
            errors.row(step).segment<3>(3) =
                (reference.template segment<3>(3) - velocity).transpose();

            if (include_rotation)
            {
                Eigen::Quaternionf quaternion(
                    reference(6), reference(7), reference(8), reference(9));
                const Eigen::Matrix3f rotation =
                    quaternion.normalized().toRotationMatrix();
                for (int row = 0; row < 3; ++row)
                {
                    for (int column = 0; column < 3; ++column)
                    {
                        errors(step, 6 + row * 3 + column) =
                            rotation(row, column);
                    }
                }
            }
        }
        return errors;
    }

private:
    void Validate() const
    {
        if (amplitudes_.size() != 3U || frequencies_.size() != 3U ||
            phases_.size() != 3U)
        {
            throw std::invalid_argument(
                "Trajectory amplitudes, frequencies, and phases must each "
                "contain three values.");
        }
        if (!std::isfinite(height_offset_) || !std::isfinite(control_dt_) ||
            control_dt_ <= 0.0f)
        {
            throw std::invalid_argument(
                "Trajectory height and control timestep must be finite, and "
                "the timestep must be positive.");
        }
        for (int axis = 0; axis < 3; ++axis)
        {
            if (!std::isfinite(amplitudes_[axis]) ||
                !std::isfinite(frequencies_[axis]) ||
                !std::isfinite(phases_[axis]) || frequencies_[axis] <= 0.0f)
            {
                throw std::invalid_argument(
                    "Trajectory parameters must be finite and frequencies "
                    "must be positive.");
            }
        }
    }

    std::vector<float> amplitudes_;
    std::vector<float> frequencies_;
    std::vector<float> phases_;
    float height_offset_;
    float control_dt_;
    bool fixed_zero_yaw_ = false;
};

#endif // COMMON_TRAJECTORY_TRACKING_REFERENCE_HH
