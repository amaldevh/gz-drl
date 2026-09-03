// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Amal Dev Haridevan

#ifndef GAZEBO_TRAJECTORY_TRACKING_HH
#define GAZEBO_TRAJECTORY_TRACKING_HH

#include "controllers/smc_controller.hh"
#include "controllers/tuned_gains.hh"
#include "env_specs/trajectory_tracking_spec.hh"
#include "gazebo_envpool/gazebo_envpool.hh"
#include "processors/trajectory_tracking_processor.hh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trajectory_tracking_env
{
template <
    typename SpecClass,
    typename ProcessorClass =
        TrajectoryTrackingProcessor<EnvSpec<SpecClass>>>
class TrajectoryTrackingEnvBase
    : public GazeboEnvpool<SpecClass, ProcessorClass>
{
public:
    using ProcessorType = ProcessorClass;
    using Parent = GazeboEnvpool<SpecClass, ProcessorType>;
    using Spec = typename Parent::Spec;
    using Action = typename Parent::Action;
    using State = typename Parent::State;

    TrajectoryTrackingEnvBase(const Spec &spec, int envid)
        : Parent(spec, envid)
    {
        envid = this->UniqueEnvid();
        auto construction_lock = this->AcquireConstructionLock();

        this->max_steps_per_episode_ =
            spec.config["max_steps_per_episode"_];
        const std::string sdf_file = spec.config["sdf_file"_];
        const bool test_env = spec.config["test_env"_];
        const int test_envid = spec.config["test_envid"_];
        this->domain_randomization_ =
            spec.config["domain_randomization"_];
        physics_steps_per_control_ =
            spec.config["physics_steps_per_control"_];
        const float configured_physics_dt = spec.config["physics_dt"_];

        this->model_names_ = {spec.config["uav_model_name"_]};
        this->model_link_names_[0] = {
            spec.config["uav_base_link_name"_]};
        state_key_ =
            this->model_names_[0] + this->model_link_names_[0][0];

        const std::string partition = test_env
                                          ? "test" + std::to_string(test_envid)
                                          : std::to_string(envid);
        this->drl_server_ = std::make_shared<DRLServer>(
            partition, sdf_file, this->model_names_, false);

        const float actual_physics_dt = this->drl_server_->step_size();
        if (std::abs(actual_physics_dt - configured_physics_dt) > 1.0e-7f)
        {
            throw std::runtime_error(
                "Configured physics_dt does not match the SDF timestep.");
        }

        const std::uint32_t trajectory_seed =
            static_cast<std::uint32_t>(envid + 42);
        this->processor_ = std::make_shared<ProcessorType>(
            state_key_,
            spec.config["use_rotation_matrix"_],
            spec.config["state_history_len"_],
            spec.config["future_waypoint_num"_],
             spec.config["action_scaling"_],
            spec.config["action_bias"_],
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

        seed_ = static_cast<float>(envid + 42);
        this->double_rng_ = std::make_unique<RNG<double>>(-1.0, 1.0, seed_);
        this->int_rng_ = std::make_unique<RNG<int>>(0, 1, seed_);
    }

    bool IsDone() override
    {
        return this->done_;
    }

    void Reset() override
    {
        this->done_ = false;
        this->current_step_ = 0;
        this->processor_->UpdateTrajectory();

        const bool randomized_this_episode =
            this->domain_randomization_ &&
            this->current_episode_ % 20 == 0;
        if (randomized_this_episode)
        {
            UpdateRandomizedParams();
            ApplyDomainRandomization();
        }
        ConfigureController(randomized_this_episode);

        const auto initial_reference =
            this->processor_->ReferenceStateAt(0.0f);
        const std::vector<double> sample = this->double_rng_->sample(3);
        const Eigen::Vector3d spawn_offset(
            0.5 * sample[0], 0.5 * sample[1], 0.25 * sample[2]);
        Eigen::Vector3d spawn_position =
            initial_reference.template head<3>().template cast<double>() +
            spawn_offset;
        Eigen::Quaterniond spawn_quaternion(
            initial_reference(6), initial_reference(7),
            initial_reference(8), initial_reference(9));
        Eigen::Vector3d zyx_angles =
            spawn_quaternion.normalized()
                .toRotationMatrix()
                .eulerAngles(2, 1, 0);
        Eigen::Vector3d spawn_orientation = zyx_angles.reverse();
        this->drl_server_->respawn_model(
            this->model_names_[0], spawn_position, spawn_orientation);
        this->drl_server_->run_N(10);
        this->UpdateControlStates();

        this->processor_->Reset(this->current_state_[state_key_]);
        this->reward_[state_key_] = 0.0f;
        this->desired_state_ = decltype(this->desired_state_)::Zero();
        ++this->current_episode_;
        WriteObs();
    }

    void Step(const Action &action) override
    {
        ++this->current_step_;
        this->processor_->ProcessAction(action, this->processed_action_);

        const auto &current_state = this->current_state_[state_key_];
        const auto &current_state_dot = this->current_state_dot_[state_key_];
        Eigen::Vector3f desired_force = this->processed_action_[state_key_];
        Eigen::VectorXf reference = this->processor_->ReferenceState();
        desired_force = desired_force - current_mass_*gravity_vec_;
        this->desired_state_ = decltype(this->desired_state_)::Zero();
        this->desired_state_(6) = reference(6);
        this->desired_state_(7) = reference(7);
        this->desired_state_(8) = reference(8);
        this->desired_state_(9) = reference(9);
        Eigen::Vector3d moments = controller_->calculate_moments(current_state.template cast<double>(),
            current_state_dot.template cast<double>(), this->desired_state_, desired_force.template cast<double>() );
        Eigen::Matrix3d rotmat = Eigen::Quaterniond(current_state(6), current_state(7), current_state(8),
    current_state(9)).toRotationMatrix();
        Eigen::Vector3d force_b = (rotmat.transpose()*(desired_force.template cast<double>()));
        Eigen::Vector3d force = rotmat*Eigen::Vector3d(0.0, 0.0, std::max(0.0, force_b(2)));
        moments = rotmat*moments;
        for (int i = 0; i<physics_steps_per_control_; ++i){
            this->drl_server_->set_wrench(this->model_names_[0], 
                this->model_link_names_[0][0], 
                force,
                moments);
                this->drl_server_->run_once();
        }
        this->UpdateControlStates();
        CheckDone();

        this->reward_[state_key_] = 0.0f;
        this->processor_->ComputeReward(
            this->current_state_, this->current_state_dot_,
            this->last_state_, this->last_state_dot_, action,
            this->reward_);
        WriteObs();
    }

protected:
    virtual void ConfigureController(bool randomized_this_episode)
    {
        controller_ =   randomized_this_episode
                ? GetRandomUAVController()
                : GetDefaultUAVController();
    }

    void CheckDone()
    {
        if (this->done_)
        {
            return;
        }
        if (this->current_step_ >= this->max_steps_per_episode_)
        {
            this->done_ = true;
            return;
        }

        this->done_ = ProcessorType::IsFailureState(
            this->current_state_[state_key_]);
    }

    void WriteObs()
    {
        State output = this->Allocate();
        this->processor_->ProcessObservation(
            this->current_state_, this->current_state_dot_,
            this->last_state_, this->last_state_dot_, output);
        const auto &current_state = this->current_state_.at(state_key_);
        const auto trajectory_reference = this->processor_->ReferenceState();
        for (int index = 0; index < current_state.size(); ++index)
        {
            output["info:state"_][index] = current_state(index);
            output["info:trajectory"_][index] =
                trajectory_reference(index);
        }
        output["reward"_] = this->reward_[state_key_];
    }

    void UpdateRandomizedParams()
    {
        this->randomized_params_.clear();
        const std::vector<double> random_values =
            this->double_rng_->sample(4);
        Eigen::Matrix3d inertia = Eigen::Matrix3d::Zero();
        inertia(0, 0) = 0.0147209;
        inertia(1, 1) = 0.0169101;
        inertia(2, 2) = 0.029448;
        double mass = 1.54;
        mass *= (random_values[0] + 10.0) / 10.0;
        inertia(0, 0) *= (random_values[1] + 10.0) / 10.0;
        inertia(1, 1) *= (random_values[2] + 10.0) / 10.0;
        inertia(2, 2) *= (random_values[3] + 10.0) / 10.0;
        this->randomized_params_[state_key_] =
            RandomizedParams{mass, inertia};
    }

    void ApplyDomainRandomization()
    {
        const auto &parameters = this->randomized_params_.at(state_key_);
        Eigen::Matrix3d inertia = parameters.inertia;
        this->drl_server_->set_mass(
            this->model_names_[0], this->model_link_names_[0][0],
            parameters.mass);
        this->drl_server_->set_inertia(
            this->model_names_[0], this->model_link_names_[0][0],
            inertia);
        current_mass_ = parameters.mass;
    }

    std::shared_ptr<UAVController> GetDefaultUAVController() const
    {
        const auto &gains =
            GAIN_MAP.at("qdrone2").at("sliding_mode_controller");
        const auto &parameters =
            PARAMETER_MAP.at("qdrone2").at("sliding_mode_controller");
        return std::make_shared<SlidingModeController>(
            gains.at("lambda_pos"), gains.at("kappa_pos"), gains.at("lambda_att"),
            gains.at("kappa_att"), gains.at("boundary_pos"),
            gains.at("boundary_att"), parameters.max_accel,
            parameters.gravity_vec, parameters.mass, parameters.inertia);
    }

    std::shared_ptr<UAVController> GetRandomUAVController()
    {
        const std::vector<double> values = this->double_rng_->sample(12);
        const auto factor = [](double value)
        {
            return (value + 11.0) / 11.0;
        };
        const auto &gains =
            GAIN_MAP.at("qdrone2").at("sliding_mode_controller");
        const auto &parameters =
            PARAMETER_MAP.at("qdrone2").at("sliding_mode_controller");

        Eigen::Vector3d position_gain;
        Eigen::Vector3d velocity_gain;
        Eigen::Vector3d attitude_gain;
        Eigen::Vector3d angular_velocity_gain;
        for (int axis = 0; axis < 3; ++axis)
        {
            position_gain(axis) =
                gains.at("lambda_pos")(axis) * factor(values[axis]);
            velocity_gain(axis) =
                gains.at("kappa_pos")(axis) * factor(values[3 + axis]);
            attitude_gain(axis) =
                gains.at("lambda_att")(axis) * factor(values[6 + axis]);
            angular_velocity_gain(axis) =
                gains.at("kappa_att")(axis) * factor(values[9 + axis]);
        }
        return std::make_shared<SlidingModeController>(
            position_gain, velocity_gain, attitude_gain,
            angular_velocity_gain, gains.at("boundary_pos"),
            gains.at("boundary_att"), parameters.max_accel,
            parameters.gravity_vec, parameters.mass, parameters.inertia);
    }

    int physics_steps_per_control_ = 10;
    float seed_ = 0.0f;
    std::string state_key_;
    std::shared_ptr<UAVController> controller_;
    double current_mass_ = 1.5;
    Eigen::Vector3f gravity_vec_{0.0f, 0.0f, -9.81f};
};

class TrajectoryTrackingEnv
    : public TrajectoryTrackingEnvBase<TrajectoryTrackingSpec>
{
public:
    using TrajectoryTrackingEnvBase<
        TrajectoryTrackingSpec>::TrajectoryTrackingEnvBase;
};
} // namespace trajectory_tracking_env

#endif // GAZEBO_TRAJECTORY_TRACKING_HH
