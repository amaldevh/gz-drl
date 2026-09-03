// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Amal Dev Haridevan

#ifndef GAZEBO_HOVER_HH_
#define GAZEBO_HOVER_HH_

#include "processors/hover_processor.hh"
#include "gazebo_envpool/gazebo_envpool.hh"
#include "controllers/inner_loop_controller.hh"
#include "env_specs/hover_spec.hh"

namespace hover_env
{
    /** @class GazeboHoverEnvpool
     * @brief Gazebo Envpool environment for Hover task
     */
    using ProcessorType = HoverProcessor<EnvSpec<HoverSpec>>;
    class HoverEnv : public GazeboEnvpool<HoverSpec, ProcessorType>
    {
    public:
        /** @brief Constructor */
        HoverEnv(const Spec &spec, int envid) : GazeboEnvpool<HoverSpec, ProcessorType>(spec, envid)
        {
            // Keep RNG streams tied to EnvPool's stable logical ID.  The
            // partition ID is offset separately and must not affect resets.
            const int logical_envid = envid;
            const int partition_envid = UniqueEnvid();

            auto construction_lock = AcquireConstructionLock();
            act_dim_ = 4; // outer-loop control always 4
            max_steps_per_episode_ = spec.config["max_steps_per_episode"_];
            const std::string sdf_file = spec.config["sdf_file"_];
            const bool test_env = spec.config["test_env"_];
            const int test_envid = spec.config["test_envid"_];
            model_names_.resize(1);
            model_names_[0] = spec.config["uav_model_name"_];
            model_link_names_[0] = {spec.config["uav_base_link_name"_]};
            domain_randomization_ = spec.config["domain_randomization"_];
            physics_steps_per_control_ = spec.config["physics_steps_per_control"_];

            state_key_ = model_names_[0] + model_link_names_[0][0];
            const int config_seed = spec.config["seed"_];
            double_rng_ = std::make_unique<RNG<double>>(-1.0, 1.0,
                                                        static_cast<float>(config_seed + logical_envid));
            int_rng_ = std::make_unique<RNG<int>>(0, 1000,
                                                  static_cast<float>(config_seed + logical_envid + 104729));

            const std::string partition = test_env ? "test" + std::to_string(test_envid) : std::to_string(partition_envid);
            const std::vector<std::string> model_names = {model_names_[0]};
            processor_ = std::make_shared<decltype(processor_)::element_type>(
                spec.config["action_history_size"_], act_dim_, state_key_);

            drl_server_ = std::make_shared<DRLServer>(partition,
                                                      sdf_file,
                                                      model_names,
                                                      false);
        }

        /** @brief Done method */
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

            // check for crash
            // attitude crash check

            auto &current_state = current_state_[state_key_];
            extra_reward_terms_ = 0.0f;
            Eigen::Quaternionf q(current_state(6), current_state(7), current_state(8), current_state(9));
            q.normalize();
            const auto R = q.toRotationMatrix();

            // roll/pitch angle from body z vs world z
            double dot = std::clamp(static_cast<double>(R.col(2).dot(Eigen::Vector3f::UnitZ())), -1.0, 1.0);
            if (std::abs(std::acos(dot)) > 1.57f)
            {
                return true;
            }

            // bounds check
            if ((current_state.head<3>().array().abs() > 7.0f).any())
            {
                return true;
            }

            const float yaw = std::atan2(R(1, 0), R(0, 0));
            if (std::abs(yaw) > 1.57f)
            {
                return true;
            }

            return false;
        }
        /** @brief Reset method */
        void Reset() override
        {
            // Reset DRL server

            auto random_values = double_rng_->sample(8);
            Eigen::Vector3d position{random_values[0] * 5.0,
                                     random_values[1] * 5.0,
                                     1.5 + random_values[2] * 1.5};
            Eigen::Vector3d euler_angles{0.0, 0.0, random_values[3] * 0.72};
            hover_pos_(0) = random_values[4] * 5.0;
            hover_pos_(1) = random_values[5] * 5.0;
            hover_pos_(2) = 1.5 + random_values[6] * 1.5;
            hover_pos_(3) = random_values[7] * 0.72;
            // drl_server_->respawn_model(model_names_[0], position, euler_angles);
            processor_->Reset(hover_pos_.cast<float>());
            current_step_ = 0;
            current_episode_ += 1;
            done_ = false;
            drl_server_->respawn_model(model_names_[0], position, euler_angles);
            // write initial obs
            drl_server_->run_once();
            UpdateControlStates();
            reward_[state_key_] = 0.0f;
            extra_reward_terms_ = 0.0f;

            // reset desired state
            desired_state_ = decltype(desired_state_)::Zero();

            WriteObs();
        }
        /** @brief Writes the current obs by allocating a vector,
         * and calling in the processor to fill it */
        void WriteObs()
        {
            State out = Allocate();
            processor_->ProcessObservation(current_state_,
                                           current_state_dot_,
                                           last_state_,
                                           last_state_dot_,
                                           out);
            out["reward"_] = reward_[state_key_] + extra_reward_terms_;
        }

        /** @brief Implements the step method */
        void Step(const Action &action) override
        {
            // update current step
            current_step_ += 1;

            // Process action
            processor_->ProcessAction(action, processed_action_);

            Eigen::VectorXf &processed_action = processed_action_[state_key_];

            // Python sends the four processed values directly as rotor velocities.
            for (int i = 0; i < physics_steps_per_control_; ++i)
            {
                drl_server_->set_rotor_velocity_cmd(
                    model_names_[0],
                    model_link_names_[0][0],
                    processed_action.cast<double>());
                drl_server_->run_once();
                UpdateControlStates();
            }

            // Compute reward
            reward_[state_key_] = 0.0f;
            processor_->ComputeReward(current_state_,
                                      current_state_dot_,
                                      last_state_,
                                      last_state_dot_,
                                      action, reward_);

            // Update the obs
            WriteObs();
        }
        /** @brief Update the randomized parameters */
        void UpdateRandomizedParams()
        {
            randomized_params_.clear();
            std::vector<double> random_values = double_rng_->sample(4);
            Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
            I(0, 0) = 0.0147209;
            I(1, 1) = 0.0169101;
            I(2, 2) = 0.029448;
            double m = 1.54;
            // apply randomization
            m *= (random_values[0] + 2.5) / 2.5;
            I(0, 0) *= (random_values[1] + 2.5) / 2.5;
            I(1, 1) *= (random_values[2] + 2.5) / 2.5;
            I(2, 2) *= (random_values[3] + 2.5) / 2.5;
            randomized_params_[state_key_] = RandomizedParams{m, I};
        }
        /** @brief Applies domain randomization to UAVs mass and Inertia */
        void ApplyDomainRandomization()
        {
            Eigen::Matrix3d I = randomized_params_[state_key_].inertia;
            double m = randomized_params_[state_key_].mass;
            // set to DRL server
            drl_server_->set_mass(model_names_[0], model_link_names_[0][0], m);
            drl_server_->set_inertia(model_names_[0], model_link_names_[0][0], std::move(I));
        }
        /** @brief Returns the default tuned controller
         * Useful for ablation studies without domain randomization
         * as well as evaluating the performance of a fixed controller
         * @return A shared pointer to a InnerLoopController. See inner_loop_controller.hh for details
         */
        std::shared_ptr<InnerLoopController> GetDefaultUAVController()
        {
            Eigen::Vector3d kp_angle(12.0, 12.0, 5);
            Eigen::Vector3d kd_angle(0.1, 0.1, 0.05);
            Eigen::Vector3d kp_angular_rate(0.1876, 0.1544, 0.09);
            Eigen::Vector3d kd_angular_rate(0.0032, 0.0026, 0.0001);
            Eigen::Vector3d g(0.0, 0.0, -9.82);
            Eigen::Matrix3d I;
            I << 0.0147209, 0.0, 0.0,
                0.0, 0.0169101, 0.0,
                0.0, 0.0, 0.029448;
            double m = 1.54;

            return std::make_shared<InnerLoopController>(kp_angle, kd_angle,
                                                         kp_angular_rate, kd_angular_rate,
                                                         m, std::move(I), g);
        }
        /** @brief Returns a random UAV controller . Useful for domain randomization
         * prevents from overfitting to a single controller
         * @return A shared pointer to a InnerLoopController. See inner_loop_controller.hh for details
         *
         */
        std::shared_ptr<InnerLoopController> GetRandomUAVController()
        {
            std::vector<double> random_values = double_rng_->sample(15);
            auto factor = [](double rand)
            { return (rand + 1.5) / 2.0; }; // Randomize between 0.25 to 1.25
            Eigen::Vector3d kp_angle(12.0 * factor(random_values[0]),
                                     12.0 * factor(random_values[1]),
                                     5 * factor(random_values[2]));
            Eigen::Vector3d kd_angle(0.1 * factor(random_values[3]),
                                     0.1 * factor(random_values[4]),
                                     0.05 * factor(random_values[5]));
            Eigen::Vector3d kp_angular_rate(0.1876 * factor(random_values[6]),
                                            0.1544 * factor(random_values[7]),
                                            0.09 * factor(random_values[8]));
            Eigen::Vector3d kd_angular_rate(0.0032 * factor(random_values[9]),
                                            0.0026 * factor(random_values[10]),
                                            0.0001 * factor(random_values[11]));

            Eigen::Vector3d g(random_values[12], random_values[13], -9.82);
            Eigen::Matrix3d I = randomized_params_[state_key_].inertia * 0.9;
            double m = randomized_params_[state_key_].mass * 0.9;
            g(2) += random_values[14] * 0.5;
            return std::make_shared<InnerLoopController>(kp_angle, kd_angle,
                                                         kp_angular_rate, kd_angular_rate,
                                                         m, std::move(I), g);
        }

    private:
        int act_dim_;
        float extra_reward_terms_ = 0.0f;
        std::shared_ptr<InnerLoopController> controller_; // The UAV controller
        Eigen::Vector4d hover_pos_;
        const Stated desired_state_dot_ = Stated::Zero();
        std::string state_key_;
        std::function<Eigen::VectorXd(const Eigen::Vector4d &)> thrust_moments_to_rotor_velocity_;
        int physics_steps_per_control_ = 1;
    };
} // namespace hover_env
#endif // GAZEBO_HOVER_HH_
