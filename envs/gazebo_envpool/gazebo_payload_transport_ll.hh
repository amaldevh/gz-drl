// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Amal Dev Haridevan

#ifndef GAZEBO_PAYLOAD_TRANSPORT_LL_HH_
#define GAZEBO_PAYLOAD_TRANSPORT_LL_HH_

#include "processors/payload_transport_ll_processor.hh"
#include "gazebo_envpool/gazebo_envpool.hh"
#include "env_specs/payload_transport_ll_spec.hh"
#include "payload_transport_common.hh"

namespace payload_transport_ll_env
{
    using ProcessorType = PayloadTransportLLProcessor<EnvSpec<PayloadTransportLLSpec>>;

    class PayloadTransportLLEnv : public GazeboEnvpool<PayloadTransportLLSpec, ProcessorType>
    {
    public:
        PayloadTransportLLEnv(const Spec &spec, int envid)
            : GazeboEnvpool<PayloadTransportLLSpec, ProcessorType>(spec, envid)
        {
            envid = UniqueEnvid();
            auto construction_lock = AcquireConstructionLock();

            act_dim_ = 4;
            max_steps_per_episode_ = spec.config["max_steps_per_episode"_];
            const std::string sdf_file = spec.config["sdf_file"_];
            const bool test_env = spec.config["test_env"_];
            const int test_envid = spec.config["test_envid"_];
            domain_randomization_ = spec.config["domain_randomization"_];

            model_names_.resize(1);
            model_names_[0] = spec.config["uav_model_name"_];
            model_link_names_[0] = {spec.config["uav_base_link_name"_],
                                    spec.config["payload_link_name"_]};

            uav_state_key_ = spec.config["uav_model_name"_] + spec.config["uav_base_link_name"_];
            payload_state_key_ = spec.config["uav_model_name"_] + spec.config["payload_link_name"_];
            // state_key_ = model_names_[0] + model_link_names_[0][0];

            action_history_size_ = spec.config["action_history_size"_];
            physics_steps_per_control_ = spec.config["physics_steps_per_control"_];
            max_rotor_vel_ = spec.config["max_rotor_vel"_];

            double_rng_ = std::make_unique<RNG<double>>(-1.0, 1.0, static_cast<float>(envid + 42));

            const std::string partition =
                test_env ? "test" + std::to_string(test_envid) : std::to_string(envid);
            const std::vector<std::string> model_names = {model_names_[0]};

            processor_ = std::make_shared<decltype(processor_)::element_type>(
                action_history_size_, act_dim_, max_rotor_vel_, uav_state_key_, payload_state_key_);

            drl_server_ = std::make_shared<DRLServer>(partition, sdf_file, model_names, false);
            default_mass_ = drl_server_->get_mass(
                model_names_[0], model_link_names_[0][0]);
            default_inertia_ = drl_server_->get_inertia(
                model_names_[0], model_link_names_[0][0]);
            default_rotor_params_ =
                drl_server_->get_rotor_parameters(model_names_[0]);
            randomized_rotor_params_ = default_rotor_params_;

            position_spawn_bound_mean_ =
                (position_spawn_bound_high_ + position_spawn_bound_low_) / 2.0;
            position_spawn_bound_diff_ =
                (position_spawn_bound_high_ - position_spawn_bound_low_) / 2.0;
        }

        bool IsDone() override
        {
            if (done_)
            {
                return true;
            }
            if (current_step_ >= max_steps_per_episode_)
            {
                done_ = true;
                return true;
            }

            done_ = payload_transport_common::IsFailureState(
                current_state_[uav_state_key_]);
            return done_;
        }

        void Reset() override
        {
            done_ = false;
            current_step_ = 0;

            auto pos = double_rng_->sample(3);
            Eigen::Vector3d random_pos{pos[0], pos[1], pos[2]};
            random_pos = position_spawn_bound_mean_ + random_pos.cwiseProduct(position_spawn_bound_diff_);
            Eigen::Vector3d orientation{0.0, 0.0, 0.0};

            if (domain_randomization_ && current_episode_ % 10 == 0)
            {
                UpdateRandomizedParams();
                ApplyDomainRandomization();
            }
            drl_server_->respawn_model(model_names_[0], random_pos, orientation);
            drl_server_->run_N(10);
            UpdateControlStates();

            reward_[uav_state_key_] = 0.0f;
            // extra_reward_terms_ = 0.0f;
            desired_state_ = decltype(desired_state_)::Zero();

            processor_->Reset(desired_payload_pos_.cast<float>());
            WriteObs();
            current_episode_++;
        }

        void Step(const Action &action) override
        {
            current_step_ += 1;

            processor_->ProcessAction(action, processed_action_);
            Eigen::VectorXf &processed_action = processed_action_[uav_state_key_];

            for (int i = 0; i < physics_steps_per_control_; ++i)
            {
                drl_server_->set_rotor_velocity_cmd(
                    model_names_[0], model_link_names_[0][0], processed_action.cast<double>());
                drl_server_->run_once();
            }

            UpdateControlStates();

            IsDone();

            reward_[uav_state_key_] = 0.0f;
            processor_->ComputeReward(current_state_, current_state_dot_, last_state_, last_state_dot_,
                                      action, reward_);
            WriteObs();
        }

        /** @brief Update the randomized parameters */
        void UpdateRandomizedParams()
        {
            randomized_params_.clear();
            randomized_rotor_params_ = default_rotor_params_;
            std::vector<double> random_values = double_rng_->sample(19);
            Eigen::Matrix3d I = default_inertia_;
            double m = default_mass_;
            // apply randomization
            m *= (random_values[0] + 10.0) / 10.0;
            I(0, 0) *= (random_values[1] + 10.0) / 10.0;
            I(1, 1) *= (random_values[2] + 10.0) / 10.0;
            I(2, 2) *= (random_values[3] + 10.0) / 10.0;
            randomized_params_[uav_state_key_] = RandomizedParams{m, I};

            // update rotor parameters

            randomized_rotor_params_.ground_effect_constant +=
                1e-8 * (random_values[13] + 10.0) / 10.0;
            randomized_rotor_params_.time_constant_up *=
                (random_values[14] + 10.0) / 10.0; //(0.09090909090909098, 1.909090909090909)
            randomized_rotor_params_.time_constant_down *=
                (random_values[15] + 10.0) / 10.0;
            randomized_rotor_params_.rotor_drag_coefficient *=
                (random_values[16] + 10.0) / 10.0;
            randomized_rotor_params_.rotor_inertia *=
                (random_values[17] + 10.0) / 10.0;
            randomized_rotor_params_.rolling_moment_coefficient +=
                1e-8 * (random_values[18] + 10.0) / 10.0;
        }
        /** @brief Applies domain randomization to UAVs mass and Inertia */
        void ApplyDomainRandomization()
        {
            Eigen::Matrix3d I = randomized_params_[uav_state_key_].inertia;
            double m = randomized_params_[uav_state_key_].mass;
            // set to DRL server
            drl_server_->set_mass(model_names_[0], model_link_names_[0][0], m);
            drl_server_->set_inertia(model_names_[0], model_link_names_[0][0], std::move(I));
            drl_server_->set_rotor_parameters(model_names_[0], randomized_rotor_params_);
        }

    private:
        void WriteObs()
        {
            State out = Allocate();
            processor_->ProcessObservation(current_state_, current_state_dot_, last_state_, last_state_dot_,
                                           out);
            out["reward"_] = reward_[uav_state_key_];
            //  + extra_reward_terms_;
        }

        int action_history_size_ = 1;
        int physics_steps_per_control_ = 10;
        double max_rotor_vel_ = 2246.0;
        std::string uav_state_key_;
        std::string payload_state_key_;

        const Eigen::Vector3d position_spawn_bound_high_{5.0, 5.0, 3.0};
        const Eigen::Vector3d position_spawn_bound_low_{-5.0, -5.0, 1.0};
        Eigen::Vector3d position_spawn_bound_diff_;
        Eigen::Vector3d position_spawn_bound_mean_;
        const Eigen::Vector3d desired_payload_pos_{0.0, 0.0, 0.5};
        bool domain_randomization_{false};
        int current_episode_{0};
        RotorParameters randomized_rotor_params_;
        RotorParameters default_rotor_params_;
        double default_mass_ = 1.52;
        Eigen::Matrix3d default_inertia_ = Eigen::Matrix3d::Identity();
        int act_dim_;
    };
} // namespace payload_tracking_env
#endif // GAZEBO_PAYLOAD_TRANSPORT_LL_HH_
