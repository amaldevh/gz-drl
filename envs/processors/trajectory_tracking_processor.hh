// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Amal Dev Haridevan

#ifndef TRAJECTORY_TRACKING_PROCESSOR_HH_
#define TRAJECTORY_TRACKING_PROCESSOR_HH_

#include "common/trajectory_tracking_reference.hh"
#include "common/waypoint_generator.hh"
#include "processors/processor.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Dense>

/** Observation, action, and reward processing for trajectory tracking.
 *
 * The observation and quadratic tracking reward follow NMPCRRLProcessor. The
 * four-dimensional low-level action from NMPC-RRL is replaced by the task's
 * three-dimensional world-frame force adjustment.
 */
template <typename EnvSpec, int ActionDimension = 3>
class TrajectoryTrackingProcessor : public GazeboProcessor<EnvSpec>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using State = typename GazeboProcessor<EnvSpec>::State;
    using Action = typename GazeboProcessor<EnvSpec>::Action;
    using Statef = typename GazeboProcessor<EnvSpec>::Statef;

    static constexpr int kActionDimension = ActionDimension;
    using ActionVector =
        Eigen::Matrix<float, kActionDimension, 1>;

    TrajectoryTrackingProcessor(
        std::string state_key,
        bool use_rotation_matrix,
        int state_history_len,
        int future_waypoint_num,
        const std::vector<float> &action_scaling,
        const std::vector<float> &action_bias,
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
        : state_key_(std::move(state_key)),
          use_rotation_matrix_(use_rotation_matrix),
          state_history_len_(state_history_len),
          future_waypoint_num_(future_waypoint_num),
          horizontal_amplitude_min_(horizontal_amplitude_min),
          horizontal_amplitude_max_(horizontal_amplitude_max),
          vertical_amplitude_min_(vertical_amplitude_min),
          vertical_amplitude_max_(vertical_amplitude_max),
          height_offset_min_(height_offset_min),
          height_offset_max_(height_offset_max),
          trajectory_speed_min_(trajectory_speed_min),
          trajectory_speed_max_(trajectory_speed_max),
          maximum_normal_acceleration_(maximum_normal_acceleration),
          vertical_position_weight_(vertical_position_weight),
          vertical_velocity_weight_(vertical_velocity_weight),
          fixed_zero_yaw_probability_(fixed_zero_yaw_probability),
          trajectory_sampling_attempts_(trajectory_sampling_attempts),
          control_dt_(physics_dt *
                      static_cast<float>(physics_steps_per_control)),
          trajectory_rng_(trajectory_seed),
          reference_(
              std::vector<float>{1.5f, 1.5f, 0.35f},
              std::vector<float>{0.35f, 0.35f, 0.175f},
              std::vector<float>{0.0f, 0.5f * kPi, 0.0f},
              0.5f * (height_offset_min + height_offset_max),
              control_dt_)
    {
        if (state_history_len_ <= 0 || future_waypoint_num_ <= 0)
        {
            throw std::invalid_argument(
                "State history and future reference lengths must be positive.");
        }
        if (physics_steps_per_control <= 0 || !std::isfinite(physics_dt) ||
            physics_dt <= 0.0f)
        {
            throw std::invalid_argument(
                "Physics steps and timestep must be positive.");
        }
        if (action_scaling.size() != kActionDimension ||
            action_bias.size() != kActionDimension)
        {
            throw std::invalid_argument(
                "Action scaling and bias must match the action dimension.");
        }

        for (int axis = 0; axis < kActionDimension; ++axis)
        {
            const float scale = action_scaling[axis];
            const float bias = action_bias[axis];
            if (!std::isfinite(scale) || scale <= 0.0f ||
                !std::isfinite(bias))
            {
                throw std::invalid_argument(
                    "Action scaling must be finite and positive, and action "
                    "bias must be finite.");
            }
            action_scaling_(axis) = scale;
            action_bias_(axis) = bias;
        }

        ValidateTrajectorySamplingConfig();

        latest_policy_action_.setZero();
        state_history_.clear();
    }

    /** Samples a new feasible Lissajous curve for each episode. */
    void UpdateTrajectory()
    {
        std::bernoulli_distribution fixed_yaw_distribution(
            fixed_zero_yaw_probability_);
        const bool fixed_zero_yaw =
            fixed_yaw_distribution(trajectory_rng_);
        std::string last_error;
        for (int attempt = 0; attempt < trajectory_sampling_attempts_;
             ++attempt)
        {
            try
            {
                const auto &multipliers = kFrequencyMultiplierFamilies[
                    UniformIndex(kFrequencyMultiplierFamilies.size())];
                ConfigureTrajectory(
                    Uniform(horizontal_amplitude_min_,
                            horizontal_amplitude_max_),
                    Uniform(horizontal_amplitude_min_,
                            horizontal_amplitude_max_),
                    Uniform(vertical_amplitude_min_,
                            vertical_amplitude_max_),
                    multipliers,
                    Uniform(0.0f, 2.0f * kPi),
                    Uniform(kMinimumXYPhaseDifference,
                            kMaximumXYPhaseDifference),
                    Uniform(0.0f, 2.0f * kPi),
                    Uniform(height_offset_min_, height_offset_max_),
                    Uniform(trajectory_speed_min_, trajectory_speed_max_),
                    fixed_zero_yaw);
                return;
            }
            catch (const std::exception &error)
            {
                last_error = error.what();
            }
        }

        // A circular horizontal projection guarantees a non-zero tangent and
        // provides a deterministic safe fallback if randomized candidates are
        // repeatedly rejected.
        try
        {
            ConfigureTrajectory(
                0.5f * (horizontal_amplitude_min_ +
                        horizontal_amplitude_max_),
                0.5f * (horizontal_amplitude_min_ +
                        horizontal_amplitude_max_),
                0.5f * (vertical_amplitude_min_ + vertical_amplitude_max_),
                std::array<int, 3>{2, 2, 1},
                0.0f,
                0.5f * kPi,
                0.0f,
                0.5f * (height_offset_min_ + height_offset_max_),
                trajectory_speed_min_,
                fixed_zero_yaw);
        }
        catch (const std::exception &fallback_error)
        {
            throw std::runtime_error(
                "Unable to sample a feasible trajectory after " +
                std::to_string(trajectory_sampling_attempts_) +
                " attempts. Last rejection: " + last_error +
                ". Fallback rejection: " + fallback_error.what());
        }
    }

    void Reset(const Statef &initial_state)
    {
        current_step_ = 1;
        latest_policy_action_.setZero();
        state_history_.clear();
        const Eigen::VectorXf initial_history =
            MakeHistoryEntry(initial_state, latest_policy_action_);
        state_history_.assign(
            static_cast<std::size_t>(state_history_len_), initial_history);
    }

    void ProcessObservation(
        const std::unordered_map<std::string, Statef> &current_state,
        const std::unordered_map<std::string, Statef> &current_state_dot,
        const std::unordered_map<std::string, Statef> &previous_state,
        const std::unordered_map<std::string, Statef> &previous_state_dot,
        State &processed_obs) override
    {
        (void)current_state_dot;
        (void)previous_state;
        (void)previous_state_dot;

        const Statef &uav_state = current_state.at(state_key_);
        PushState(uav_state, latest_policy_action_);

        int index = 0;
        auto append = [&](float value)
        {
            if (index >= ExpectedObservationDimension())
            {
                throw std::logic_error(
                    "Observation writer exceeded the configured dimension.");
            }
            processed_obs["obs"_][index++] = Sanitize(value);
        };

        for (const Eigen::VectorXf &history : state_history_)
        {
            for (int feature = 0; feature < history.size(); ++feature)
            {
                append(history(feature));
            }
        }

        const Eigen::MatrixXf reference_errors = reference_.Errors(
            uav_state, CurrentTime(), future_waypoint_num_,
            use_rotation_matrix_);
        for (int step = 0; step < reference_errors.rows(); ++step)
        {
            for (int feature = 0; feature < reference_errors.cols(); ++feature)
            {
                append(reference_errors(step, feature));
            }
        }

        if (index != ExpectedObservationDimension())
        {
            throw std::logic_error(
                "Observation writer did not fill the configured dimension.");
        }
    }

    void ProcessAction(
        const Action &policy_action,
        std::unordered_map<std::string, Eigen::VectorXf> &processed_action)
        override
    {
        const float *action_data =
            static_cast<const float *>(policy_action["action"_].Data());
        auto &output = processed_action[state_key_];
        output.resize(kActionDimension);
        for (int axis = 0; axis < kActionDimension; ++axis)
        {
            latest_policy_action_(axis) = std::clamp(
                Sanitize(action_data[axis]), -1.0f, 1.0f);
            output(axis) =
                latest_policy_action_(axis) * action_scaling_(axis) +
                action_bias_(axis);
        }
        ++current_step_;
    }

    void ComputeReward(
        const std::unordered_map<std::string, Statef> &current_state,
        const std::unordered_map<std::string, Statef> &current_state_dot,
        const std::unordered_map<std::string, Statef> &previous_state,
        const std::unordered_map<std::string, Statef> &previous_state_dot,
        const Action &action,
        std::unordered_map<std::string, float> &rewards) override
    {
        (void)current_state_dot;
        (void)previous_state;
        (void)previous_state_dot;
        (void)action;

        const Statef &state = current_state.at(state_key_);
        const Statef desired = reference_.State(
            static_cast<float>(current_step_ - 1) * control_dt_);

        const Eigen::Vector3f position_error =
            desired.template head<3>() - state.template head<3>();
        const Eigen::Vector3f velocity_error =
            desired.template segment<3>(3) -
            state.template segment<3>(3);
        float cost = kHorizontalPositionWeight *
                     position_error.template head<2>().squaredNorm();
        cost += vertical_position_weight_ *
                position_error.z() * position_error.z();
        cost += kHorizontalVelocityWeight *
                velocity_error.template head<2>().squaredNorm();
        cost += vertical_velocity_weight_ *
                velocity_error.z() * velocity_error.z();

        cost += kAngularVelocityWeight *
                (desired.template segment<3>(10) -
                 state.template segment<3>(10))
                    .squaredNorm();
        cost += kActionWeight * latest_policy_action_.squaredNorm();

        // Keep the NMPC-RRL quadratic objective, but map it to a bounded,
        // strictly positive tracking reward. Continuing a valid episode can
        // therefore never be worse than deliberately terminating it early.
        float reward = 1.0f / (1.0f + cost);
        if (IsFailureState(state))
        {
            reward -= kFailurePenalty;
        }
        rewards[state_key_] = Sanitize(reward);
    }

    Statef ReferenceState() const
    {
        return reference_.State(CurrentTime());
    }

    Statef ReferenceStateAt(float time) const
    {
        return reference_.State(time);
    }

    float CurrentTime() const noexcept
    {
        return static_cast<float>(current_step_) * control_dt_;
    }

    static bool IsOutOfBounds(const Eigen::Vector3f &position) noexcept
    {
        return std::abs(position.x()) > 9.0f ||
               std::abs(position.y()) > 9.0f || position.z() < 0.0f ||
               position.z() > 10.0f;
    }

    static constexpr float MaximumTilt() noexcept
    {
        return kMaximumTilt;
    }

    static bool IsFailureState(const Statef &state) noexcept
    {
        if (IsOutOfBounds(state.template head<3>()))
        {
            return true;
        }

        Eigen::Quaternionf quaternion(
            state(6), state(7), state(8), state(9));
        const float quaternion_norm = quaternion.norm();
        if (!std::isfinite(quaternion_norm) || quaternion_norm <= kEpsilon)
        {
            return true;
        }

        const Eigen::Vector3f body_z =
            quaternion.normalized().toRotationMatrix().col(2);
        const float tilt = std::acos(std::clamp(
            body_z.dot(Eigen::Vector3f::UnitZ()), -1.0f, 1.0f));
        return tilt >= kMaximumTilt ||
               state.template segment<3>(10).norm() >
                   kMaximumAngularVelocity;
    }

private:
    static constexpr float kEpsilon = 1.0e-6f;
    static constexpr float kPi = 3.14159265358979323846f;
    static constexpr float kMinimumXYPhaseDifference =
        70.0f * kPi / 180.0f;
    static constexpr float kMaximumXYPhaseDifference =
        110.0f * kPi / 180.0f;
    static constexpr std::array<std::array<int, 3>, 5>
        kFrequencyMultiplierFamilies{{
            {{1, 1, 1}},
            {{2, 2, 1}},
            {{1, 1, 2}},
            {{2, 1, 1}},
            {{1, 2, 1}},
        }};
    static constexpr float kMaximumTilt = 1.75f * kPi / 2.0f;
    static constexpr float kMaximumAngularVelocity = 10.0f;
    static constexpr float kHorizontalPositionWeight = 0.1f;
    static constexpr float kHorizontalVelocityWeight = 0.002f;
    static constexpr float kRotationWeight = 0.02f;
    static constexpr float kAngularVelocityWeight = 0.01f;
    static constexpr float kActionWeight = 0.01f;
    static constexpr float kFailurePenalty = 50.0f;

    int HistoryFeatureCount() const noexcept
    {
        return (use_rotation_matrix_ ? 16 : 7) + kActionDimension;
    }

    int ReferenceFeatureCount() const noexcept
    {
        return use_rotation_matrix_ ? 15 : 6;
    }

    int ExpectedObservationDimension() const noexcept
    {
        return state_history_len_ * HistoryFeatureCount() +
               future_waypoint_num_ * ReferenceFeatureCount();
    }

    static float Sanitize(float value) noexcept
    {
        if (std::isnan(value))
        {
            return 0.0f;
        }
        if (std::isinf(value))
        {
            value = value > 0.0f ? 1.0e5f : -1.0e5f;
        }
        return std::clamp(value, -100.0f, 100.0f);
    }

    void ValidateTrajectorySamplingConfig() const
    {
        const auto valid_range = [](float lower, float upper)
        {
            return std::isfinite(lower) && std::isfinite(upper) &&
                   lower > 0.0f && lower <= upper;
        };
        if (!valid_range(horizontal_amplitude_min_,
                         horizontal_amplitude_max_) ||
            !valid_range(vertical_amplitude_min_, vertical_amplitude_max_) ||
            !valid_range(height_offset_min_, height_offset_max_) ||
            !valid_range(trajectory_speed_min_, trajectory_speed_max_))
        {
            throw std::invalid_argument(
                "Trajectory amplitude, height, and speed ranges must be "
                "finite, positive, and ordered.");
        }
        if (height_offset_min_ - vertical_amplitude_max_ < 0.0f)
        {
            throw std::invalid_argument(
                "The minimum trajectory altitude must not be below zero.");
        }
        if (!std::isfinite(maximum_normal_acceleration_) ||
            maximum_normal_acceleration_ <= 0.0f ||
            !std::isfinite(vertical_position_weight_) ||
            vertical_position_weight_ <= 0.0f ||
            !std::isfinite(vertical_velocity_weight_) ||
            vertical_velocity_weight_ <= 0.0f ||
            !std::isfinite(fixed_zero_yaw_probability_) ||
            fixed_zero_yaw_probability_ < 0.0f ||
            fixed_zero_yaw_probability_ > 1.0f ||
            trajectory_sampling_attempts_ <= 0)
        {
            throw std::invalid_argument(
                "Maximum normal acceleration, vertical tracking weights, "
                "and sampling attempts must be positive, and fixed-yaw "
                "probability must be in [0, 1].");
        }
    }

    float Uniform(float lower, float upper)
    {
        std::uniform_real_distribution<float> distribution(lower, upper);
        return distribution(trajectory_rng_);
    }

    std::size_t UniformIndex(std::size_t count)
    {
        std::uniform_int_distribution<std::size_t> distribution(0, count - 1);
        return distribution(trajectory_rng_);
    }

    static float MaximumParameterSpeed(
        const std::vector<float> &amplitudes,
        const std::array<int, 3> &multipliers,
        const std::vector<float> &phases)
    {
        constexpr int kSamples = 4096;
        float maximum_speed = 0.0f;
        for (int sample = 0; sample < kSamples; ++sample)
        {
            const float parameter =
                2.0f * kPi * static_cast<float>(sample) /
                static_cast<float>(kSamples);
            float squared_speed = 0.0f;
            for (int axis = 0; axis < 3; ++axis)
            {
                const float multiplier =
                    static_cast<float>(multipliers[axis]);
                const float derivative =
                    amplitudes[axis] * multiplier *
                    std::cos(multiplier * parameter + phases[axis]);
                squared_speed += derivative * derivative;
            }
            maximum_speed =
                std::max(maximum_speed, std::sqrt(squared_speed));
        }
        if (!std::isfinite(maximum_speed) || maximum_speed <= kEpsilon)
        {
            throw std::runtime_error(
                "Sampled Lissajous curve has no usable tangent.");
        }
        return maximum_speed;
    }

    void ConfigureTrajectory(
        float amplitude_x,
        float amplitude_y,
        float amplitude_z,
        const std::array<int, 3> &multipliers,
        float phase_x,
        float xy_phase_difference,
        float phase_z,
        float height_offset,
        float maximum_speed,
        bool fixed_zero_yaw)
    {
        const std::vector<float> amplitudes{
            amplitude_x, amplitude_y, amplitude_z};
        const std::vector<float> phases{
            phase_x, phase_x + xy_phase_difference, phase_z};

        WaypointGenerator::FeasibilityLimits limits;
        limits.min_speed = maximum_speed;
        limits.max_speed = maximum_speed;
        limits.max_normal_acceleration = maximum_normal_acceleration_;
        // Use large finite values rather than infinity: this project is built
        // with floating-point optimizations under which infinity-based
        // disabled limits are not reliably recognized by std::isfinite.
        limits.max_specific_thrust = 1000.0;
        limits.max_tilt_rad = 89.0 * static_cast<double>(kPi) / 180.0;
        limits.max_heading_rate_rad_s = 1000.0;
        limits.max_yaw_rate_rad_s = 1000.0;
        limits.safety_factor = 1.0;
        limits.feasibility_samples = 4096;

        feasibility_checker_.ValidateLissajousCurve(
            amplitude_x,
            amplitude_y,
            amplitude_z,
            static_cast<float>(multipliers[0]),
            static_cast<float>(multipliers[1]),
            static_cast<float>(multipliers[2]),
            phases[0],
            phases[1],
            phases[2],
            limits);

        const float parameter_rate =
            maximum_speed /
            MaximumParameterSpeed(amplitudes, multipliers, phases);
        const std::vector<float> frequencies{
            parameter_rate * static_cast<float>(multipliers[0]),
            parameter_rate * static_cast<float>(multipliers[1]),
            parameter_rate * static_cast<float>(multipliers[2])};
        reference_.UpdateCurve(
            amplitudes, frequencies, phases, height_offset, fixed_zero_yaw);
    }

    Eigen::VectorXf MakeHistoryEntry(
        const Statef &state,
        const ActionVector &action) const
    {
        Eigen::VectorXf entry(HistoryFeatureCount());
        int index = 0;
        entry(index++) = Sanitize(state(2));
        for (int axis = 0; axis < 3; ++axis)
        {
            entry(index++) = Sanitize(state(3 + axis));
        }

        if (use_rotation_matrix_)
        {
            Eigen::Quaternionf quaternion(
                state(6), state(7), state(8), state(9));
            Eigen::Matrix3f rotation = Eigen::Matrix3f::Identity();
            if (std::isfinite(quaternion.norm()) &&
                quaternion.norm() > kEpsilon)
            {
                rotation = quaternion.normalized().toRotationMatrix();
            }
            for (int row = 0; row < 3; ++row)
            {
                for (int column = 0; column < 3; ++column)
                {
                    entry(index++) = Sanitize(rotation(row, column));
                }
            }
        }

        for (int axis = 0; axis < 3; ++axis)
        {
            entry(index++) = Sanitize(state(10 + axis));
        }
        for (int axis = 0; axis < kActionDimension; ++axis)
        {
            entry(index++) = Sanitize(action(axis));
        }
        return entry;
    }

    void PushState(const Statef &state, const ActionVector &action)
    {
        state_history_.push_back(MakeHistoryEntry(state, action));
        while (static_cast<int>(state_history_.size()) > state_history_len_)
        {
            state_history_.pop_front();
        }
    }

    std::string state_key_;
    bool use_rotation_matrix_;
    int state_history_len_;
    int future_waypoint_num_;
    ActionVector action_scaling_ = ActionVector::Ones();
    ActionVector action_bias_ = ActionVector::Zero();
    ActionVector latest_policy_action_ = ActionVector::Zero();
    std::deque<Eigen::VectorXf> state_history_;
    float horizontal_amplitude_min_;
    float horizontal_amplitude_max_;
    float vertical_amplitude_min_;
    float vertical_amplitude_max_;
    float height_offset_min_;
    float height_offset_max_;
    float trajectory_speed_min_;
    float trajectory_speed_max_;
    float maximum_normal_acceleration_;
    float vertical_position_weight_;
    float vertical_velocity_weight_;
    float fixed_zero_yaw_probability_;
    int trajectory_sampling_attempts_;
    float control_dt_;
    std::mt19937 trajectory_rng_;
    WaypointGenerator feasibility_checker_;
    int current_step_ = 0;
    TrajectoryTrackingReference reference_;
};

#endif // TRAJECTORY_TRACKING_PROCESSOR_HH_
