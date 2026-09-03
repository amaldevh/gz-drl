// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Amal Dev Haridevan

#ifndef PAYLOAD_TRANSPORT_LL_PROCESSOR_HH_
#define PAYLOAD_TRANSPORT_LL_PROCESSOR_HH_

#include "processors/processor.hh"
#include "print_utils.hh"
#include "common_utils.hh"
#include "payload_transport_common.hh"

template <typename EnvSpec>
class PayloadTransportLLProcessor : public GazeboProcessor<EnvSpec>
{
public:
    using State = typename GazeboProcessor<EnvSpec>::State;
    using Action = typename GazeboProcessor<EnvSpec>::Action;

    PayloadTransportLLProcessor(int action_history_size,
                                int act_dim,
                                double max_rotor_vel,
                                std::string uav_state_key,
                                std::string payload_state_key)
        : action_history_size_(action_history_size),
          act_dim_(act_dim),
          max_rotor_vel_(max_rotor_vel),
          uav_state_key_(std::move(uav_state_key)),
          payload_state_key_(std::move(payload_state_key))
    {
        if (act_dim_ != 4)
        {
            throw std::runtime_error("PayloadTransportLLProcessor only supports act_dim = 4");
        }
        action_history_.assign(action_history_size_ * act_dim_, 0.0f);
        action_scaling_.resize(act_dim_);
        action_scaling_ << max_rotor_vel_, max_rotor_vel_, max_rotor_vel_, max_rotor_vel_;
        desired_payload_pos_ = Eigen::Vector3f(0.0f, 0.0f, 0.5f);
    }

    void Reset(const Eigen::Vector3f &desired_payload_pos)
    {
        action_history_.clear();
        action_history_.assign(action_history_size_ * act_dim_, 0.0f);
        desired_payload_pos_ = desired_payload_pos;
    }

    void ProcessObservation(const std::unordered_map<std::string, Statef> &current_state,
                            const std::unordered_map<std::string, Statef> &current_state_dot,
                            const std::unordered_map<std::string, Statef> &previous_state,
                            const std::unordered_map<std::string, Statef> &previous_state_dot,
                            State &processed_obs) override
    {
        const auto &uav_state = current_state.at(uav_state_key_);
        const auto &payload_state = current_state.at(payload_state_key_);

        int offset = 0;

        for (int i = 0; i < 3; ++i)
        {
            processed_obs["obs"_][offset++] = payload_state(i) - desired_payload_pos_(i);
        }

        for (int i = 0; i < 3; ++i)
        {
            processed_obs["obs"_][offset++] = payload_state(i) - uav_state(i);
        }

        for (int i = 6; i < 10; ++i)
        {
            processed_obs["obs"_][offset++] = uav_state(i);
        }

        for (int i = 3; i < 6; ++i)
        {
            processed_obs["obs"_][offset++] = uav_state(i);
        }

        for (int i = 10; i < 13; ++i)
        {
            processed_obs["obs"_][offset++] = uav_state(i);
        }

        for (int i = 3; i < 6; ++i)
        {
            processed_obs["obs"_][offset++] = payload_state(i);
        }

        for (int i = 10; i < 13; ++i)
        {
            processed_obs["obs"_][offset++] = payload_state(i);
        }

        for (const float &act : action_history_)
        {
            processed_obs["obs"_][offset++] = act;
        }
    }

    void ProcessAction(const Action &policy_action,
                       std::unordered_map<std::string, Eigen::VectorXf> &processed_action) override
    {
        const float *action_data = static_cast<const float *>(policy_action["action"_].Data());
        processed_action[uav_state_key_].resize(act_dim_);
        processed_action[uav_state_key_] << action_data[0], action_data[1], action_data[2], action_data[3];

        PushAction(processed_action[uav_state_key_]);

        processed_action[uav_state_key_] =
            ((Eigen::VectorXf::Ones(act_dim_) + processed_action[uav_state_key_]) / 2.0f)
                .cwiseProduct(action_scaling_);
    }

    void ComputeReward(const std::unordered_map<std::string, Statef> &current_state,
                       const std::unordered_map<std::string, Statef> &current_state_dot,
                       const std::unordered_map<std::string, Statef> &previous_state,
                       const std::unordered_map<std::string, Statef> &previous_state_dot,
                       const Action &action,
                       std::unordered_map<std::string, float> &rewards) override
    {
        const Eigen::Vector3f relative_pos =
            desired_payload_pos_ - current_state.at(payload_state_key_).head<3>();
        const Eigen::Vector3f payload_ang_vel = current_state.at(payload_state_key_).segment<3>(10);
        const Eigen::Vector3f uav_ang_vel = current_state.at(uav_state_key_).segment<3>(10);

        const float payload_pos_rew = 1.0f / (1.0f + 1.6f * relative_pos.squaredNorm());
        const float omega_cost =
            0.1f / (1.0f + payload_ang_vel.squaredNorm() + uav_ang_vel.squaredNorm());

        float reward = payload_pos_rew + omega_cost;
        if (payload_transport_common::IsFailureState(
                current_state.at(uav_state_key_)))
        {
            reward -= payload_transport_common::kFailurePenalty;
        }
        rewards[uav_state_key_] = reward;
    }

private:
    void PushAction(const Eigen::VectorXf &action)
    {
        for (int i = 0; i < action.size(); ++i)
        {
            action_history_.push_back(action(i));
            if (static_cast<int>(action_history_.size()) > action_history_size_ * act_dim_)
            {
                action_history_.pop_front();
            }
        }
    }

    int action_history_size_;
    int act_dim_;
    Eigen::VectorXf action_scaling_;
    double max_rotor_vel_;
    std::string uav_state_key_;
    std::string payload_state_key_;
    std::deque<float> action_history_;
    Eigen::Vector3f desired_payload_pos_;
};

#endif
