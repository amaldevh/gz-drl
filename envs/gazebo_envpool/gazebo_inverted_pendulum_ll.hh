// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Amal Dev Haridevan

#ifndef GAZEBO_INVERTED_PENDULUM_LL_HH_
#define GAZEBO_INVERTED_PENDULUM_LL_HH_

#include "processors/inverted_pendulum_ll_processor.hh"
#include "gazebo_envpool/gazebo_envpool.hh"
#include "env_specs/inverted_pendulum_ll_spec.hh"
namespace inverted_pendulum_ll_env
{
    using ProcessorType = InvertedPendulumLLProcessor<EnvSpec<InvertedPendulumLLSpec>>;

    class InvertedPendulumLLEnv : public GazeboEnvpool<InvertedPendulumLLSpec, ProcessorType>
    {
    public:
        InvertedPendulumLLEnv(const Spec &spec, int envid)
            : GazeboEnvpool<InvertedPendulumLLSpec, ProcessorType>(spec, envid)
        {
            envid = UniqueEnvid();
            auto construction_lock = AcquireConstructionLock();

            act_dim_ = 4;
            max_steps_per_episode_ = spec.config["max_steps_per_episode"_];
            const std::string sdf_file = spec.config["sdf_file"_];
            const bool test_env = spec.config["test_env"_];
            const int test_envid = spec.config["test_envid"_];
            domain_randomization_ = spec.config["domain_randomization"_];
            rotor_link_names_ = spec.config["rotor_link_names"_];
            turning_dirs_ = spec.config["turning_directions"_];
            ktau_ = spec.config["ktau"_];
            max_rotor_thrust_ = spec.config["max_rotor_thrust"_];

            model_names_.resize(1);
            model_names_[0] = spec.config["uav_model_name"_];
            model_link_names_[0] = {spec.config["uav_base_link_name"_],
                                    spec.config["payload_link_name"_]};

            uav_state_key_ = spec.config["uav_model_name"_] + spec.config["uav_base_link_name"_];
            payload_state_key_ = spec.config["uav_model_name"_] + spec.config["payload_link_name"_];

            action_history_size_ = spec.config["action_history_size"_];
            state_history_size_ = spec.config["state_history_size"_];
            physics_steps_per_control_ = spec.config["physics_steps_per_control"_];

            double_rng_ = std::make_unique<RNG<double>>(-1.0, 1.0, static_cast<float>(envid + 42));

            const std::string partition =
                test_env ? "test" + std::to_string(test_envid) : std::to_string(envid);
            const std::vector<std::string> model_names = {model_names_[0]};

            processor_ = std::make_shared<decltype(processor_)::element_type>(
                action_history_size_, state_history_size_, act_dim_, max_rotor_thrust_, uav_state_key_, payload_state_key_,
            spec.config["privileged_obs"_], max_xyz_);

            drl_server_ = std::make_shared<DRLServer>(partition, sdf_file, model_names, false);
            if (test_env)
            {
                drl_server_->set_trajectory_trace(model_names_[0], model_link_names_[0][0]);
            }


            position_spawn_bound_mean_ =
                (position_spawn_bound_high_ + position_spawn_bound_low_) / 2.0;
            position_spawn_bound_diff_ =
                (position_spawn_bound_high_ - position_spawn_bound_low_) / 2.0;
            // mass inertia obs
            Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
            I(0, 0) = 0.0147209;
            I(1, 1) = 0.0169101;
            I(2, 2) = 0.029448;
            double m = 1.54;
            double tc_up = 0.0125;
            double tc_down = 0.025;
            mass_inertia_tc_obs_.resize(6);
            mass_inertia_tc_obs_(0) = m;
            mass_inertia_tc_obs_(1) = I(0,0);
            mass_inertia_tc_obs_(2) = I(1,1);
            mass_inertia_tc_obs_(3) = I(2,2);
            mass_inertia_tc_obs_(4) = tc_up;
            mass_inertia_tc_obs_(5) = tc_down;
            drl_server_->run_once();
            UpdateControlStates();
            processor_->Reset(desired_payload_pos_.cast<float>(), 
                current_state_[uav_state_key_],
                current_state_[payload_state_key_],
                current_state_dot_[uav_state_key_],
                current_state_dot_[payload_state_key_], mass_inertia_tc_obs_);
        }

        bool IsDone() override
        {
            // if (done_)
            // {
            //     return true;
            // }
            if (current_step_ >= max_steps_per_episode_)
            {
                done_ = true;
                return true;
            }

            const auto &uav_state = current_state_[uav_state_key_];
            const auto &payload_state = current_state_[payload_state_key_];

            const Eigen::Vector3d current_pos = uav_state.head<3>().cast<double>();
            const Eigen::Vector3d payload_pos = payload_state.head<3>().cast<double>();

            if (((current_pos.cwiseAbs() - max_xyz_).array() > 0.0).any())
            {
                done_ = true;
                return true;
            }

            Eigen::Quaterniond quat(uav_state(6), uav_state(7), uav_state(8), uav_state(9));
            const Eigen::Matrix3d rotmat = quat.normalized().toRotationMatrix();
            const double cosang =
                std::abs(std::acos(std::clamp(rotmat.col(2).dot(Eigen::Vector3d::UnitZ()), -1.0, 1.0)));
            if (cosang >  1.57)
            {
                done_ = true;
                return true;
            }
            if (current_pos(2) < 0.3){
                done_ = true;
                return true;
            }
            const double unit_z_dot = (payload_pos - current_pos).normalized()(2);
            if (unit_z_dot < 0.2)
            {
                done_ = true;
                return true;
            }

            return false;
        }

        void Reset() override
        {
            done_ = false;
            current_step_ = 0;

            auto pos = double_rng_->sample(3);
            Eigen::Vector3d random_pos{pos[0], pos[1], pos[2]};
            random_pos = position_spawn_bound_mean_ + random_pos.cwiseProduct(position_spawn_bound_diff_);
            double yaw = 0.0;
            Eigen::Vector3d orientation{0.0, 0.0, yaw};

            if (domain_randomization_)
            {
                UpdateRandomizedParams();
                ApplyDomainRandomization();
            }

            drl_server_->respawn_model(model_names_[0], random_pos, orientation);
            drl_server_->run_N(10);
            drl_server_->update_control_states();
            UpdateControlStates();
            
            reward_[uav_state_key_] = 0.0f;
            // extra_reward_terms_ = 0.0f;

            processor_->Reset(desired_payload_pos_.cast<float>(), 
            current_state_[uav_state_key_],
             current_state_[payload_state_key_],
             current_state_dot_[uav_state_key_],
             current_state_dot_[payload_state_key_], mass_inertia_tc_obs_);
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
                // drl_server_->set_rotor_velocity_cmd(
                //     model_names_[0], model_link_names_[0][0], processed_action.cast<double>());
                drl_server_->set_srt_cmd(model_names_[0], model_link_names_[0][0],
                rotor_link_names_, turning_dirs_, processed_action.cast<double>(),
            ktau_);
                drl_server_->run_once();
            }

            drl_server_->update_control_states();
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
            randomized_params_[uav_state_key_] = RandomizedParams{m, I};

            // update rotor parameters
            randomized_rotor_params_ = default_rotor_params_;
            randomized_rotor_params_.time_constant_up *=
                (random_values[14] + 10.0) / 10.0; //(0.09090909090909098, 1.909090909090909)
            randomized_rotor_params_.time_constant_down *=
                (random_values[15] + 10.0) / 10.0;
            // update mass_inertia_tc_obs_
            mass_inertia_tc_obs_(0) = m;
            mass_inertia_tc_obs_(1) = I(0,0);  
            mass_inertia_tc_obs_(2) = I(1,1);
            mass_inertia_tc_obs_(3) = I(2,2);
            mass_inertia_tc_obs_(4) = randomized_rotor_params_.time_constant_up;
            mass_inertia_tc_obs_(5) = randomized_rotor_params_.time_constant_down;

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
            drl_server_->set_srt_rate_limiter_time_constants(model_names_[0], model_link_names_[0][0],
            randomized_rotor_params_.time_constant_up, randomized_rotor_params_.time_constant_down, Eigen::Vector4d::Zero());
        }

    private:
        void WriteObs()
        {
            State out = Allocate();
            processor_->ProcessObservation(current_state_, current_state_dot_, last_state_, last_state_dot_,
                                           out);
            out["reward"_] = reward_[uav_state_key_] ;
            // + extra_reward_terms_;
        }

        int action_history_size_ = 1;
        int state_history_size_ = 1;
        int physics_steps_per_control_ = 10;
        double max_rotor_thrust_ = 12.0;
        std::string uav_state_key_;
        std::string payload_state_key_;

        const Eigen::Vector3d max_xyz_{10.0, 10.0, 10.0};
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
        int act_dim_;
        Eigen::VectorXd mass_inertia_tc_obs_;
        std::vector<std::string> rotor_link_names_;
        std::vector<int>        turning_dirs_;
        double ktau_;
    };
} // namespace inverted_pendulum_env
#endif
