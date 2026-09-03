// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Amal Dev Haridevan

#ifndef GAZEBO_PAYLOAD_TRANSPORT_HH_
#define GAZEBO_PAYLOAD_TRANSPORT_HH_

#include "processors/payload_transport_processor.hh"
#include "gazebo_envpool/gazebo_envpool.hh"
#include "env_specs/payload_transport_spec.hh"
#include "controllers/smc_controller.hh"
#include "payload_transport_common.hh"

namespace payload_transport_env
{
    /** @class GazeboPayloadTransportEnvpool
     * @brief Gazebo Envpool environment for PayloadTransport task
     */
    using ProcessorType = PayloadTransportProcessor<EnvSpec<PayloadTransportSpec>>;
    class PayloadTransportEnv : public GazeboEnvpool<PayloadTransportSpec, ProcessorType>
    {
    public:
        /** @brief Constructor */
        PayloadTransportEnv(const Spec &spec, int envid) : GazeboEnvpool<PayloadTransportSpec, ProcessorType>(spec, envid)
        {

            // Ensure unique id
            envid = UniqueEnvid();

            auto construction_lock = AcquireConstructionLock();
            act_dim_ = 3; // high-level control [fx fy fz]
            max_steps_per_episode_ = spec.config["max_steps_per_episode"_];
            const std::string sdf_file = spec.config["sdf_file"_];
            const bool test_env = spec.config["test_env"_];
            const int test_envid = spec.config["test_envid"_];
            domain_randomization_ = spec.config["domain_randomization"_];

            model_names_.resize(1);
            model_names_[0] = spec.config["uav_model_name"_];
            model_link_names_[0] = {spec.config["uav_base_link_name"_], spec.config["payload_link_name"_]};

            payload_state_key_ = spec.config["uav_model_name"_] + spec.config["payload_link_name"_];
            uav_state_key_ = spec.config["uav_model_name"_] + spec.config["uav_base_link_name"_];
            state_key_ = model_names_[0] + model_link_names_[0][0];

            physics_steps_per_control_ = spec.config["physics_steps_per_control"_];
            force_scaling_ = spec.config["force_scaling"_];
            action_history_size_ = spec.config["action_history_size"_];

            double_rng_ = std::make_unique<RNG<double>>(-1.0, 1.0, static_cast<float>(envid + 42));

            const std::string partition =
                test_env ? "test" + std::to_string(test_envid) : std::to_string(envid);
            const std::vector<std::string> model_names = {model_names_[0]};

            processor_ = std::make_shared<decltype(processor_)::element_type>(
                action_history_size_, act_dim_, force_scaling_, uav_state_key_, payload_state_key_);

            drl_server_ = std::make_shared<DRLServer>(partition, sdf_file, model_names, false);
            default_mass_ = drl_server_->get_mass(
                model_names_[0], model_link_names_[0][0]);
            default_inertia_ = drl_server_->get_inertia(
                model_names_[0], model_link_names_[0][0]);
            default_rotor_params_ =
                drl_server_->get_rotor_parameters(model_names_[0]);
            randomized_rotor_params_ = default_rotor_params_;
            current_mass_ = default_mass_;

            position_spawn_bound_mean_ =
                (position_spawn_bound_high_ + position_spawn_bound_low_) / 2.0;
            position_spawn_bound_diff_ =
                (position_spawn_bound_high_ - position_spawn_bound_low_) / 2.0;
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

            done_ = payload_transport_common::IsFailureState(
                current_state_[uav_state_key_]);
            return done_;
        }
        /** @brief Reset method */
        void Reset() override
        {
            done_ = false;
            current_step_ = 0;

            auto pos = double_rng_->sample(3);
            Eigen::Vector3d random_pos{pos[0], pos[1], pos[2]};
            random_pos = position_spawn_bound_mean_ + random_pos.cwiseProduct(position_spawn_bound_diff_);
            Eigen::Vector3d orientation{0.0, 0.0, 0.0};

            // Apply a new physical randomization every 10 episodes.
            if (domain_randomization_ && current_episode_ % 10 == 0)
            {
                UpdateRandomizedParams();
                ApplyDomainRandomization();
            }
            // Keep the modular controller fixed so that only the policy/control
            // interface differs from the end-to-end environment.
            controller_ = GetDefaultUAVController();
            drl_server_->set_controller(
                model_names_[0], model_link_names_[0][0], controller_);
            thrust_to_rotor_velocity_func_ =
                drl_server_->get_thrust_moment_to_rotor_velocity_mapping_function(model_names_[0]);

            drl_server_->respawn_model(model_names_[0], random_pos, orientation);
            drl_server_->run_N(10);
            UpdateControlStates();

            reward_[uav_state_key_] = 0.0f;
            extra_reward_terms_ = 0.0f;
            desired_state_ = decltype(desired_state_)::Zero();

            processor_->Reset(desired_payload_pos_.cast<float>());
            WriteObs();
            current_episode_++;
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
            out["reward"_] = reward_[uav_state_key_] + extra_reward_terms_;
        }

        /** @brief Implements the step method */
        void Step(const Action &action) override
        {
            current_step_ += 1;

            processor_->ProcessAction(action, processed_action_);

            Statef &current_state = current_state_[uav_state_key_];
            Statef &current_state_dot = current_state_dot_[uav_state_key_];
            Eigen::VectorXf &processed_action = processed_action_[uav_state_key_];

            Eigen::Matrix<double, 13, 1> desired_state = Eigen::Matrix<double, 13, 1>::Zero();
            desired_state(6) = 1.0;
            desired_state.segment<6>(0) = current_state.segment<6>(0).cast<double>();
            Eigen::Vector3d force = processed_action.cast<double>();
            force -= current_mass_ * gravity_vec_;
            Eigen::Vector3d moments = controller_->calculate_moments(
                current_state.cast<double>(), current_state_dot.cast<double>(), desired_state, force);
            Eigen::Quaterniond q(current_state[6], current_state[7], current_state[8], current_state[9]);
            q.normalize();
            Eigen::Vector4d thrust_moments;
            thrust_moments << (q.inverse() * force)[2], moments[0], moments[1], moments[2];
            Eigen::VectorXd rotor_velocities = thrust_to_rotor_velocity_func_(thrust_moments);

            for (int i = 0; i < physics_steps_per_control_; ++i)
            {
                drl_server_->set_rotor_velocity_cmd(
                    model_names_[0], model_link_names_[0][0], rotor_velocities);
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

            // randomized_rotor_params_.thrust_constant_quadratic_params[0] *=
            //     (random_values[7] + 10.0)/10.0;
            // randomized_rotor_params_.thrust_constant_quadratic_params[1] *=
            //     (random_values[8] + 10.0)/10.0;
            // randomized_rotor_params_.thrust_constant_quadratic_params[2] *=
            //     (random_values[9] + 10.0)/10.0;
            // randomized_rotor_params_.torque_constant_quadratic_params[0] *=
            //     (random_values[10] + 10.0)/10.0;
            // randomized_rotor_params_.torque_constant_quadratic_params[1] *=
            //     (random_values[11] + 10.0)/10.0;
            // randomized_rotor_params_.torque_constant_quadratic_params[2] *=
            //     (random_values[12] + 10.0)/10.0;
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
            current_mass_ = m;
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
    private:
        // void WriteObs()  {
        //     State out = Allocate();
        //     processor_->ProcessObservation(current_state_,
        //         current_state_dot_,
        //         last_state_,
        //         last_state_dot_,
        //          out);
        //     out["reward"_] = reward_[uav_state_key_] + extra_reward_terms_;

        // }

        float extra_reward_terms_ = 0.0f;
        int action_history_size_ = 1;
        int physics_steps_per_control_ = 1;
        std::string uav_state_key_;
        std::string payload_state_key_;                      // The pendulum state key
        Eigen::Vector3d desired_payload_pos_{0.0, 0.0, 0.5}; // Desired payload position
        std::string state_key_;
        RotorParameters randomized_rotor_params_;
        RotorParameters default_rotor_params_;
        double default_mass_ = 1.52;
        double current_mass_ = 1.52;
        Eigen::Matrix3d default_inertia_ = Eigen::Matrix3d::Identity();
        const Eigen::Vector3d gravity_vec_{0.0, 0.0, -9.82};
        std::shared_ptr<UAVController> controller_;
        std::function<Eigen::VectorXd(const Eigen::Vector4d &)> thrust_to_rotor_velocity_func_;
        std::vector<float> force_scaling_;
        const Eigen::Vector3d position_spawn_bound_high_{5.0, 5.0, 3.0};
        const Eigen::Vector3d position_spawn_bound_low_{-5.0, -5.0, 1.0};
        Eigen::Vector3d position_spawn_bound_diff_;
        Eigen::Vector3d position_spawn_bound_mean_;
        bool domain_randomization_{false};
        int current_episode_{0};
        int act_dim_;
    };
} // namespace payload_transport_env
#endif // GAZEBO_PAYLOAD_TRANSPORT_LL_HH_
