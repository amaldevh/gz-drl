// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Amal Dev Haridevan

#ifndef GAZEBO_MULTIAGENT_FORMATION_LL_HH_
#define GAZEBO_MULTIAGENT_FORMATION_LL_HH_

#include "processors/multiagent_formation_ll_processor.hh"
#include "gazebo_envpool/gazebo_envpool.hh"
#include "env_specs/multiagent_formation_ll_spec.hh"
namespace multiagent_formation_ll_env
{
    using ProcessorType = MultiAgentFormationLLProcessor<EnvSpec<MultiAgentFormationLLSpec>>;
    class MultiAgentFormationLLEnv : public GazeboEnvpool<MultiAgentFormationLLSpec, ProcessorType>
    {
    public:
        MultiAgentFormationLLEnv(const Spec &spec, int envid)
            : GazeboEnvpool<MultiAgentFormationLLSpec, ProcessorType>(spec, envid)
        {
            envid = UniqueEnvid();
            auto construction_lock = AcquireConstructionLock();

            act_dim_ = 4;
            max_steps_per_episode_ = spec.config["max_steps_per_episode"_];
            const int num_agents = spec.config["num_agents"_];
            num_agents_ = num_agents;
            const std::string sdf_file = "world_formation" + std::to_string(num_agents) + ".sdf";
            const bool test_env = spec.config["test_env"_];
            const int test_envid = spec.config["test_envid"_];
            domain_randomization_ = spec.config["domain_randomization"_];

            model_names_.resize(num_agents);
            uav_state_keys_.resize(num_agents);
            std::string base_uav_model_name = spec.config["base_uav_model_name"_];
            std::string uav_base_link_name = spec.config["uav_base_link_name"_];
            for (int i = 0; i < num_agents; ++i)
            {
                model_names_[i] = base_uav_model_name + std::to_string(i + 1);
                model_link_names_[i] = {uav_base_link_name};
                uav_state_keys_[i] = model_names_[i] + uav_base_link_name;
            }

            physics_steps_per_control_ = spec.config["physics_steps_per_control"_];
            max_rotor_vel_ = spec.config["max_rotor_vel"_];

            double_rng_ = std::make_unique<RNG<double>>(-1.0, 1.0, static_cast<float>(envid + 42));

            const std::string partition =
                test_env ? "test" + std::to_string(test_envid) : std::to_string(envid);
            const std::vector<std::string> model_names = model_names_;

            processor_ = std::make_shared<decltype(processor_)::element_type>(
                num_agents_, act_dim_, max_rotor_vel_, uav_state_keys_);

            drl_server_ = std::make_shared<DRLServer>(partition, sdf_file, model_names, false);


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

            if (processor_->done_)
            {
                return true;
            }
            return false;
        }

        void Reset() override
        {

            current_step_ = 0;
            done_ = false;
            if (domain_randomization_ && current_episode_ % 10 == 0)
            {
                UpdateRandomizedParams();
                ApplyDomainRandomization();
            }

            std::vector<Eigen::Vector3d> initial_positions(num_agents_);
            std::vector<double> pos_t = double_rng_->sample(3);
            target_pos_ << pos_t[0] * 10.0, pos_t[1] * 10.0, std::max(0.0, pos_t[2]) * 10.0 + 1.0;
            for (int i = 0; i < num_agents_; ++i)
            {
                if (i == 0)
                {
                    initial_positions[i] = Eigen::Vector3d(
                        target_pos_(0) + double_rng_->sample(1)[0] * 2.0f,
                        target_pos_(1) + double_rng_->sample(1)[0] * 2.0f,
                        target_pos_(2) + std::max(0.0, double_rng_->sample(1)[0]) * 1.5f + 0.5f);

                }
                else{
                    initial_positions[i]  = initial_positions[i-1] + Eigen::Vector3d(0.7, 0.7, 0.0); 
                }

                Eigen::Vector3d spawn_ori(0.0, 0.0, 0.0);
                drl_server_->respawn_model(model_names_[i], initial_positions[i], spawn_ori);
            }

            drl_server_->run_N(10);
            drl_server_->update_control_states();
            UpdateControlStates();
            for (int i = 0; i < num_agents_; ++i)
            {
                reward_[uav_state_keys_[i]] = 0.0f;
            }
            // extra_reward_terms_ = 0.0f;

            processor_->Reset(target_pos_.cast<float>());
            WriteObs();
            current_episode_++;
        }

        void Step(const Action &action) override
        {
            current_step_ += 1;

            processor_->ProcessAction(action, processed_action_);

            for (int i = 0; i < physics_steps_per_control_; ++i)
            {
                for (int j = 0; j < num_agents_; ++j)
                {

                    drl_server_->set_rotor_velocity_cmd(
                        model_names_[j], model_link_names_[j][0], processed_action_[uav_state_keys_[j]].cast<double>());
                }
                drl_server_->run_once();
            }

            drl_server_->update_control_states();
            UpdateControlStates();

            IsDone();
            for (int i = 0; i < num_agents_; ++i)
            {
                reward_[uav_state_keys_[i]] = 0.0f;
            }
            processor_->ComputeReward(current_state_, current_state_dot_, last_state_, last_state_dot_,
                                      action, reward_);
            WriteObs();
        }

        /** @brief Update the randomized parameters */
        void UpdateRandomizedParams()
        {
            randomized_params_.clear();
            std::vector<double> random_values = double_rng_->sample(19);
            Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
            I(0, 0) = 0.0147209;
            I(1, 1) = 0.0169101;
            I(2, 2) = 0.029448;
            double m = 1.54;
            // apply randomization
            m *= (random_values[0] + 10.0) / 10.0;
            I(0, 0) *= (random_values[1] + 10.0) / 10.0;
            I(1, 1) *= (random_values[2] + 10.0) / 10.0;
            I(2, 2) *= (random_values[3] + 10.0) / 10.0;
            for (int i = 0; i < num_agents_; ++i)
            {
                randomized_params_[uav_state_keys_[i]] = RandomizedParams{m, I};
            }

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
            for (int i = 0; i < num_agents_; ++i)
            {
                Eigen::Matrix3d I = randomized_params_[uav_state_keys_[i]].inertia;
                double m = randomized_params_[uav_state_keys_[i]].mass;
                // set to DRL server
                drl_server_->set_mass(model_names_[i], model_link_names_[i][0], m);
                drl_server_->set_inertia(model_names_[i], model_link_names_[i][0], std::move(I));
                drl_server_->set_rotor_parameters(model_names_[i], randomized_rotor_params_);
            }
        }

    private:
        void WriteObs()
        {
            State out = Allocate();
            processor_->ProcessObservation(current_state_, current_state_dot_, last_state_, last_state_dot_,
                                           out);
            // for (int i = 0; i < num_agents_; ++i) {
            out["reward"_] = reward_[uav_state_keys_[0]];
            // }
        }

        int action_history_size_ = 1;
        int physics_steps_per_control_ = 10;
        double max_rotor_vel_ = 2300.0;
        std::vector<std::string> uav_state_keys_;

        const Eigen::Vector3d max_xyz_{7.0, 7.0, 7.0};
        const double max_yaw_{1.57};
        const Eigen::Vector3d position_spawn_bound_high_{5.0, 5.0, 3.0};
        const Eigen::Vector3d position_spawn_bound_low_{-5.0, -5.0, 1.0};
        Eigen::Vector3d position_spawn_bound_diff_;
        Eigen::Vector3d position_spawn_bound_mean_;
        const Eigen::Vector3d desired_payload_pos_{0.0, 0.0, 2.0};
        bool domain_randomization_{false};
        int current_episode_{0};
        RotorParameters randomized_rotor_params_;
        const RotorParameters default_rotor_params_;
        int num_agents_;
        Eigen::Vector3f target_pos_;
        int act_dim_;
    };
} // namespace inverted_pendulum_env
#endif
