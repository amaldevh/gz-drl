// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Amal Dev Haridevan

#ifndef GAZEBO_TRAJECTORY_TRACKING_LL_HH
#define GAZEBO_TRAJECTORY_TRACKING_LL_HH

#include "env_specs/trajectory_tracking_ll_spec.hh"
#include "gazebo_envpool/gazebo_envpool.hh"
#include "processors/trajectory_tracking_ll_processor.hh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace trajectory_tracking_ll_env
{
using ProcessorType =
    TrajectoryTrackingLLProcessor<EnvSpec<TrajectoryTrackingLLSpec>>;

class TrajectoryTrackingLLEnv
    : public GazeboEnvpool<TrajectoryTrackingLLSpec, ProcessorType>
{
public:
    using Parent =
        GazeboEnvpool<TrajectoryTrackingLLSpec, ProcessorType>;
    using Spec = typename Parent::Spec;
    using Action = typename Parent::Action;
    using State = typename Parent::State;

    TrajectoryTrackingLLEnv(const Spec &spec, int envid)
        : Parent(spec, envid)
    {
        envid = UniqueEnvid();
        auto construction_lock = AcquireConstructionLock();

        max_steps_per_episode_ = spec.config["max_steps_per_episode"_];
        physics_steps_per_control_ =
            spec.config["physics_steps_per_control"_];
        domain_randomization_ = spec.config["domain_randomization"_];
        const std::string sdf_file = spec.config["sdf_file"_];
        const bool test_env = spec.config["test_env"_];
        const int test_envid = spec.config["test_envid"_];

        model_names_ = {spec.config["uav_model_name"_]};
        model_link_names_[0] = {spec.config["uav_base_link_name"_]};
        state_key_ = model_names_[0] + model_link_names_[0][0];

        const std::string partition = test_env
                                          ? "test" + std::to_string(test_envid)
                                          : std::to_string(envid);
        drl_server_ = std::make_shared<DRLServer>(
            partition, sdf_file, model_names_, false);

        const float actual_physics_dt = drl_server_->step_size();
        const float configured_physics_dt = spec.config["physics_dt"_];
        if (std::abs(actual_physics_dt - configured_physics_dt) > 1.0e-7f)
        {
            throw std::runtime_error(
                "Configured physics_dt does not match the SDF timestep.");
        }

        const std::uint32_t trajectory_seed =
            static_cast<std::uint32_t>(envid + 42);
        processor_ = std::make_shared<ProcessorType>(
            state_key_,
            spec.config["use_rotation_matrix"_],
            spec.config["state_history_len"_],
            spec.config["future_waypoint_num"_],
            spec.config["min_rotor_vel"_],
            spec.config["max_rotor_vel"_],
            physics_steps_per_control_,
            actual_physics_dt,
            spec.config["trajectory_xy_amplitude_min"_],
            spec.config["trajectory_xy_amplitude_max"_],
            spec.config["trajectory_z_amplitude_min"_],
            spec.config["trajectory_z_amplitude_max"_],
            spec.config["trajectory_height_offset_min"_],
            spec.config["trajectory_height_offset_max"_],
            spec.config["trajectory_speed_min"_],
            spec.config["trajectory_speed_max"_],
            spec.config["trajectory_max_normal_acceleration"_],
            spec.config["trajectory_vertical_position_weight"_],
            spec.config["trajectory_vertical_velocity_weight"_],
            spec.config["trajectory_fixed_zero_yaw_probability"_],
            spec.config["trajectory_sampling_attempts"_],
            trajectory_seed);

        const float seed = static_cast<float>(envid + 42);
        double_rng_ = std::make_unique<RNG<double>>(-1.0, 1.0, seed);
        int_rng_ = std::make_unique<RNG<int>>(0, 1, seed);
    }

    bool IsDone() override
    {
        return done_;
    }

    void Reset() override
    {
        done_ = false;
        current_step_ = 0;
        processor_->UpdateTrajectory();

        if (domain_randomization_ && current_episode_ % 20 == 0)
        {
            UpdateRandomizedParams();
            ApplyDomainRandomization();
        }

        const auto initial_reference = processor_->ReferenceStateAt(0.0f);
        const std::vector<double> sample = double_rng_->sample(3);
        const Eigen::Vector3d spawn_offset(
            0.5 * sample[0], 0.5 * sample[1], 0.25 * sample[2]);
        Eigen::Vector3d spawn_position =
            initial_reference.head<3>().cast<double>() + spawn_offset;
        Eigen::Quaterniond spawn_quaternion(
            initial_reference(6), initial_reference(7),
            initial_reference(8), initial_reference(9));
        Eigen::Vector3d zyx_angles =
            spawn_quaternion.normalized()
                .toRotationMatrix()
                .eulerAngles(2, 1, 0);
        Eigen::Vector3d spawn_orientation = zyx_angles.reverse();
        drl_server_->respawn_model(
            model_names_[0], spawn_position, spawn_orientation);
        drl_server_->run_N(10);
        UpdateControlStates();

        processor_->Reset(current_state_[state_key_]);
        reward_[state_key_] = 0.0f;
        ++current_episode_;
        WriteObs();
    }

    void Step(const Action &action) override
    {
        ++current_step_;
        processor_->ProcessAction(action, processed_action_);
        Eigen::VectorXd rotor_velocity =
            processed_action_[state_key_].cast<double>();

        for (int step = 0; step < physics_steps_per_control_; ++step)
        {
            drl_server_->set_rotor_velocity_cmd(
                model_names_[0], model_link_names_[0][0], rotor_velocity);
            drl_server_->run_once();
        }
        UpdateControlStates();
        CheckDone();

        reward_[state_key_] = 0.0f;
        processor_->ComputeReward(
            current_state_, current_state_dot_, last_state_, last_state_dot_,
            action, reward_);
        WriteObs();
    }

private:
    void CheckDone()
    {
        if (done_)
        {
            return;
        }
        if (current_step_ >= max_steps_per_episode_)
        {
            done_ = true;
            return;
        }

        done_ = ProcessorType::IsFailureState(current_state_[state_key_]);
    }

    void WriteObs()
    {
        State output = Allocate();
        processor_->ProcessObservation(
            current_state_, current_state_dot_, last_state_, last_state_dot_,
            output);
        const auto &current_state = current_state_.at(state_key_);
        const auto trajectory_reference = processor_->ReferenceState();
        for (int index = 0; index < current_state.size(); ++index)
        {
            output["info:state"_][index] = current_state(index);
            output["info:trajectory"_][index] =
                trajectory_reference(index);
        }
        output["reward"_] = reward_[state_key_];
    }

    void UpdateRandomizedParams()
    {
        randomized_params_.clear();
        const std::vector<double> values = double_rng_->sample(4);
        Eigen::Matrix3d inertia = Eigen::Matrix3d::Zero();
        inertia(0, 0) = 0.0147209 * (values[1] + 10.0) / 10.0;
        inertia(1, 1) = 0.0169101 * (values[2] + 10.0) / 10.0;
        inertia(2, 2) = 0.029448 * (values[3] + 10.0) / 10.0;
        const double mass = 1.54 * (values[0] + 10.0) / 10.0;
        randomized_params_[state_key_] = RandomizedParams{mass, inertia};
    }

    void ApplyDomainRandomization()
    {
        const auto &parameters = randomized_params_.at(state_key_);
        Eigen::Matrix3d inertia = parameters.inertia;
        drl_server_->set_mass(
            model_names_[0], model_link_names_[0][0], parameters.mass);
        drl_server_->set_inertia(
            model_names_[0], model_link_names_[0][0], inertia);
    }

    int physics_steps_per_control_ = 10;
    std::string state_key_;
};
} // namespace trajectory_tracking_ll_env

#endif // GAZEBO_TRAJECTORY_TRACKING_LL_HH
