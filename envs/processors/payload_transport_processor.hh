// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Amal Dev Haridevan

#ifndef PAYLOAD_TRANSPORT_PROCESSOR_HH_
#define PAYLOAD_TRANSPORT_PROCESSOR_HH_

#include "processors/processor.hh"
#include "print_utils.hh"
#include "common_utils.hh"
#include "payload_transport_common.hh"

/** @brief PayloadTransportProcessor
 * Processor for Invertedpayload environment
 */
template <typename EnvSpec>
class PayloadTransportProcessor : public GazeboProcessor<EnvSpec>
{
public:
    using State = typename GazeboProcessor<EnvSpec>::State;
    using Action = typename GazeboProcessor<EnvSpec>::Action;
    /** @brief Constructor
     * @param action_history_size The size of the action history
     * @param act_dim Action dimension (3 for residual world-frame force)
     * @param uav_state_key The UAV state key
     * @param payload_state_key The payload state key
     */
    PayloadTransportProcessor(int action_history_size,
                              int act_dim,
                              std::vector<float> force_scaling,
                              std::string uav_state_key,
                              std::string payload_state_key) : action_history_size_(action_history_size),
                                                               act_dim_(act_dim),
                                                               uav_state_key_(std::move(uav_state_key)),
                                                               payload_state_key_(std::move(payload_state_key))
    {

        if (act_dim_ != 3)
            throw std::runtime_error("PayloadTransportProcessor only supports act_dim = 3");
        action_history_.assign(action_history_size_ * act_dim_, 0.0f);
        action_scaling_.resize(act_dim_);
        action_scaling_ << force_scaling[0], force_scaling[1], force_scaling[2];
        desired_payload_pos_ = Eigen::Vector3f(0.0f, 0.0f, 0.5f);
    }
    /** @brief Pushes an action to the queue, should be done before processing observation */
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
    /** @brief Resets the processor state
     *
     * @param desired_payload_pos The desired position for the payload (3D position)
     */
    void Reset(const Eigen::Vector3f &desired_payload_pos)
    {
        action_history_.clear();
        action_history_.assign(action_history_size_ * act_dim_, 0.0f);
        desired_payload_pos_ = desired_payload_pos;
        print_info("Desired payload pos: ", desired_payload_pos_.transpose());
    }

    /** @brief ProcessObservation processes the raw observation from the environment
     * Observation consists of:
     * relative position of payload to desired pos (3),
     * relative position of payload to UAV (3),
     * quaternion (4), velocity (3), omega (3) of UAV,
     * payload linear vel (3), payload angular vel (3),
     * last actions (3 * action_history_size) = 22 + last_action
     */
    void ProcessObservation(const std::unordered_map<std::string, Statef> &current_state,
                            const std::unordered_map<std::string, Statef> &current_state_dot,
                            const std::unordered_map<std::string, Statef> &previous_state,
                            const std::unordered_map<std::string, Statef> &previous_state_dot,
                            State &processed_obs) override
    {
        const auto &uav_state = current_state.at(uav_state_key_);
        const auto &payload_state = current_state.at(payload_state_key_);
        int offset = 0;

        // Relative position of payload to desired pos (3)
        for (int i = 0; i < 3; ++i)
        {
            processed_obs["obs"_][offset++] =
                payload_state(i) - desired_payload_pos_(i);
        }

        // Relative position of payload to UAV (3)
        for (int i = 0; i < 3; ++i)
        {
            processed_obs["obs"_][offset++] =
                payload_state(i) - uav_state(i);
        }

        // UAV quaternion (4)
        for (int i = 6; i < 10; ++i)
        {
            processed_obs["obs"_][offset++] = uav_state(i);
        }

        // UAV velocity (3)
        for (int i = 3; i < 6; ++i)
        {
            processed_obs["obs"_][offset++] = uav_state(i);
        }

        // UAV omega (3)
        for (int i = 10; i < 13; ++i)
        {
            processed_obs["obs"_][offset++] = uav_state(i);
        }

        // Payload linear velocity (3)
        for (int i = 3; i < 6; ++i)
        {
            processed_obs["obs"_][offset++] = payload_state(i);
        }

        // Payload angular velocity (3)
        for (int i = 10; i < 13; ++i)
        {
            processed_obs["obs"_][offset++] = payload_state(i);
        }

        // Action history
        for (const float &act : action_history_)
        {
            processed_obs["obs"_][offset++] = act;
        }
    }

    /* @brief ProcessAction processes the raw action before sending to the environment
     * The action is a normalized residual world-frame force [Fx, Fy, Fz].
     */
    void ProcessAction(const Action &policy_action,
                       std::unordered_map<std::string, Eigen::VectorXf> &processed_action) override
    {
        const float *action_data = static_cast<const float *>(policy_action["action"_].Data());
        processed_action[uav_state_key_].resize(act_dim_);
        processed_action[uav_state_key_] << action_data[0], action_data[1], action_data[2];
        // Push to action history
        PushAction(processed_action[uav_state_key_]);
        processed_action[uav_state_key_] = (processed_action[uav_state_key_]).cwiseProduct(action_scaling_);
    }
    /** @brief Reward function */
    void ComputeReward(const std::unordered_map<std::string, Statef> &current_state,
                       const std::unordered_map<std::string, Statef> &current_state_dot,
                       const std::unordered_map<std::string, Statef> &previous_state,
                       const std::unordered_map<std::string, Statef> &previous_state_dot,
                       const Action &action,
                       std::unordered_map<std::string, float> &rewards) override
    {
        const Eigen::Vector3f relative_pos =
            desired_payload_pos_ - current_state.at(payload_state_key_).head<3>();
        const Eigen::Vector3f payload_omega =
            current_state.at(payload_state_key_).segment<3>(10);
        const Eigen::Vector3f uav_ang_vel =
            current_state.at(uav_state_key_).segment<3>(10);

        const float payload_pos_rew = 1.0f / (1.0f + 1.6f * relative_pos.squaredNorm());
        const float omega_cost =
            0.1f / (1.0f + payload_omega.squaredNorm() + uav_ang_vel.squaredNorm());

        float reward = payload_pos_rew + omega_cost;
        if (payload_transport_common::IsFailureState(
                current_state.at(uav_state_key_)))
        {
            reward -= payload_transport_common::kFailurePenalty;
        }
        rewards[uav_state_key_] = reward;
    }

private:
    int action_history_size_;
    int act_dim_;
    Eigen::VectorXf action_scaling_;
    std::string uav_state_key_;
    std::string payload_state_key_;
    std::deque<float> action_history_;
    Eigen::Vector3f desired_payload_pos_;
};
#endif
