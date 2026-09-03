// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Amal Dev Haridevan

#ifndef HOVER_PROCESSOR_HH_
#define HOVER_PROCESSOR_HH_

#include "processors/processor.hh"
#include "print_utils.hh"
#include "common_utils.hh"

/** @brief HoverProcessor
 * Processor for Hover environment
 */
template <typename EnvSpec>
class HoverProcessor : public GazeboProcessor<EnvSpec>
{
public:
    using State = typename GazeboProcessor<EnvSpec>::State;
    using Action = typename GazeboProcessor<EnvSpec>::Action;
    /** @brief Constructor
     * @param action_history_size The size of the action history
     * @param act_dim Action dim (3 if xyz, 4 if xyz-yaw)
     * @param state_key The UAV state key
     */
    HoverProcessor(int action_history_size,
                   int act_dim,
                   std::string state_key) : action_history_size_(action_history_size),
                                            act_dim_(act_dim),
                                            state_key_(state_key)
    {
        action_history_.assign(action_history_size_ * act_dim_, 0.0f);
        action_scaling_.resize(act_dim_);
        if (act_dim_ != 4)
            throw std::runtime_error("HoverProcessor only supports act_dim = 4");
        // Python HoverEnv maps each policy action from [-1, 1] to a rotor
        // velocity in [0, 2300].
        action_scaling_.setConstant(1150.0f);
    }
    /** @brief Pushes an action to the queue, should be done before processing observation */
    void PushAction(const Eigen::VectorXf &action)
    {
        // Keep the newest action first, matching action_buffer1/action_buffer2
        // in the Python environment.
        for (int i = action.size() - 1; i >= 0; --i)
        {
            action_history_.push_front(action(i));
            if (static_cast<int>(action_history_.size()) > action_history_size_ * act_dim_)
            {
                action_history_.pop_back();
            }
        }
    }
    /** @brief Resets the processor state
     *
     * @param hover_pos The desired hover position
     * as xyz-yaw (4)
     */
    void Reset(const Eigen::Vector4f &hover_pos)
    {
        action_history_.clear();
        // Python resets rotor buffers to zero, which its observation transform
        // exposes as 0 / 1300 - 1 == -1.
        action_history_.assign(action_history_size_ * act_dim_, -1.0f);
        desired_hover_pos_ = hover_pos;
        print_info("Hover pos: ", desired_hover_pos_.transpose());
    }

    /** @brief ProcessObservation processes the raw observation from the environment */
    void ProcessObservation(const std::unordered_map<std::string, Statef> &current_state,
                            const std::unordered_map<std::string, Statef> &current_state_dot,
                            const std::unordered_map<std::string, Statef> &previous_state,
                            const std::unordered_map<std::string, Statef> &previous_state_dot,
                            State &processed_obs) override
    {
        // Copy  vel, quat, omega
        // from index 4 to 13
        const auto &uav_state = current_state.at(state_key_);
        for (int i = 4; i < 14; ++i)
        {
            processed_obs["obs"_][i] = uav_state(i - 1);
        }
        // Relative position to desired hover pos
        // at index 0,1,2
        for (int i = 0; i < 3; ++i)
        {
            processed_obs["obs"_][i] =
                desired_hover_pos_(i) - uav_state(i);
        }
        // Yaw error
        Eigen::Quaternionf q(
            uav_state(6),
            uav_state(7),
            uav_state(8),
            uav_state(9));
        q.normalize();
        Eigen::Matrix3f R = q.toRotationMatrix();
        float yaw = (std::atan2(R(1, 0), R(0, 0)));
        float yaw_error = desired_hover_pos_(3) - yaw;
        // Normalize yaw error to [-pi, pi]
        // yaw_error = wrap_angle(yaw_error);

        processed_obs["obs"_][3] = yaw_error;
        // Action history
        int offset = 14;
        for (const float &act : action_history_)
        {
            processed_obs["obs"_][offset++] = act;
        }
    }

    /* @brief ProcessAction processes the raw action before sending to the environment  */
    void ProcessAction(const Action &policy_action,
                       std::unordered_map<std::string, Eigen::VectorXf> &processed_action) override
    {
        const float *action_data = static_cast<const float *>(policy_action["action"_].Data());
        processed_action[state_key_].resize(act_dim_);
        for (int i = 0; i < act_dim_; ++i)
        {
            const float action = std::isnan(action_data[i]) ? 0.0f : action_data[i];
            processed_action[state_key_](i) = (action + 1.0f) * action_scaling_(i);
        }
        // Python exposes rotor_velocity / 1300 - 1 in its action history.
        PushAction(processed_action[state_key_] / 1300.0f - Eigen::VectorXf::Ones(act_dim_));
    }
    /** @brief Reward function */
    void ComputeReward(const std::unordered_map<std::string, Statef> &current_state,
                       const std::unordered_map<std::string, Statef> &current_state_dot,
                       const std::unordered_map<std::string, Statef> &previous_state,
                       const std::unordered_map<std::string, Statef> &previous_state_dot,
                       const Action &action,
                       std::unordered_map<std::string, float> &rewards) override
    {
        const auto &uav_state = current_state.at(state_key_);
        const Eigen::Vector3f relative_pos = desired_hover_pos_.head<3>() - uav_state.head<3>();
        Eigen::Quaternionf q(
            uav_state(6),
            uav_state(7),
            uav_state(8),
            uav_state(9));
        q.normalize();
        Eigen::Matrix3f R = q.toRotationMatrix();
        float yaw = (std::atan2(R(1, 0), R(0, 0)));
        const float yaw_error = desired_hover_pos_(3) - yaw;
        const float state_error = 1.6f * 1.6f *
                                  (relative_pos.squaredNorm() + yaw_error * yaw_error);
        const float position_reward = 1.0f / (1.0f + state_error);
        const float angular_velocity_reward =
            0.01f / (1.0f + uav_state.segment<3>(10).squaredNorm());
        rewards[state_key_] = position_reward + angular_velocity_reward;
    }

private:
    inline float wrap_angle_(float a)
    {
        while (a > M_PI)
            a -= 2.0 * M_PI;
        while (a <= -M_PI)
            a += 2.0 * M_PI;
        return a;
    }
    int action_history_size_;
    int act_dim_;
    Eigen::VectorXf action_scaling_;
    std::string state_key_;
    std::deque<float> action_history_;
    Eigen::Vector4f desired_hover_pos_;
};
#endif
