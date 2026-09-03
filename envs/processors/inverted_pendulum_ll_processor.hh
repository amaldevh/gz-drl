// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Amal Dev Haridevan

#ifndef INVERTED_PENDULUM_LL_PROCESSOR_HH_
#define INVERTED_PENDULUM_LL_PROCESSOR_HH_

#include "processors/processor.hh"
#include "print_utils.hh"
#include "common_utils.hh"

template <typename EnvSpec>
class InvertedPendulumLLProcessor : public GazeboProcessor<EnvSpec>
{
public:
    using State = typename GazeboProcessor<EnvSpec>::State;
    using Action = typename GazeboProcessor<EnvSpec>::Action;

    InvertedPendulumLLProcessor(int action_history_size,
                                int state_history_size,
                                int act_dim,
                                double max_rotor_thrust,
                                std::string uav_state_key,
                                std::string payload_state_key,
                                bool privileged_obs,
                            const Eigen::Vector3d& max_xyz)
        : action_history_size_(action_history_size),
          state_history_size_(state_history_size),
          act_dim_(act_dim),
          max_rotor_thrust_(max_rotor_thrust),
          uav_state_key_(std::move(uav_state_key)),
          payload_state_key_(std::move(payload_state_key)),
          privileged_obs_(privileged_obs),
          max_xyz_(max_xyz)
    {
        if (act_dim_ != 4)
        {
            throw std::runtime_error("InvertedPendulumLLProcessor only supports act_dim = 4");
        }
        action_history_.assign(action_history_size_ , Eigen::Vector4f::Zero());
        state_history_.assign(state_history_size_ , Statef::Zero());
        payload_state_history_.assign(state_history_size_, Statef::Zero());
        state_dot_history_.assign(state_history_size_ , Statef::Zero());
        payload_state_dot_history_.assign(state_history_size_, Statef::Zero());
        action_scaling_.resize(act_dim_);
        // action_scaling_ << max_rotor_vel_, max_rotor_vel_, max_rotor_vel_, max_rotor_vel_;
        desired_payload_pos_ = Eigen::Vector3f(0.0f, 0.0f, 2.0f);
    }

    void Reset(const Eigen::Vector3f &desired_payload_pos, const Statef& initial_uav_state,
        const Statef& initial_payload_state, const Statef& initial_uav_state_dot,
        const Statef& initial_payload_state_dot, const Eigen::VectorXd& mass_inertia_tc_obs)
    {
        action_history_.clear();
        action_history_.assign(action_history_size_ , Eigen::Vector4f::Zero());
        desired_payload_pos_ = desired_payload_pos;
        state_history_.clear();
        state_history_.assign(state_history_size_, initial_uav_state);
        payload_state_history_.clear();
        payload_state_history_.assign(state_history_size_, initial_payload_state);
        state_dot_history_.clear();
        state_dot_history_.assign(state_history_size_, initial_uav_state_dot);
        payload_state_dot_history_.clear();
        payload_state_dot_history_.assign(state_history_size_, initial_payload_state_dot);
        mass_inertia_tc_obs_ = mass_inertia_tc_obs;
    }

    void ProcessObservation(const std::unordered_map<std::string, Statef> &current_state,
                            const std::unordered_map<std::string, Statef> &current_state_dot,
                            const std::unordered_map<std::string, Statef> &previous_state,
                            const std::unordered_map<std::string, Statef> &previous_state_dot,
                            State &processed_obs) override
    {
        const auto &uav_state = current_state.at(uav_state_key_);
        const auto &payload_state = current_state.at(payload_state_key_);
        const auto &uav_state_dot = current_state_dot.at(uav_state_key_);
        const auto &payload_state_dot = current_state_dot.at(payload_state_key_);
        PushStates(uav_state, payload_state, uav_state_dot, payload_state_dot);

        int offset = 0;
        for (size_t i = 0; i < state_history_size_; ++i){
            const auto& statei = state_history_[i];
            const auto& payload_statei = payload_state_history_[i];
            for (int j = 0; j < 3; ++j)
            {   // payload pos err [3]
                float val = desired_payload_pos_(j) - payload_statei(j);
                processed_obs["obs"_][offset++] = Sanitize(val);
            }
            for (int j = 0; j < 3; ++j)
            {   // payload pos w.r.t uav [3]
                float val = payload_statei(j) - statei(j);
                processed_obs["obs"_][offset++] = Sanitize(val);
            }
            for (int j = 3; j < 6; ++j){
                // paylaod vel [3]
                processed_obs["obs"_][offset++] = Sanitize(payload_statei(j));
            }
            for (int j = 10; j < 13; ++j){
                // payloadd ang vel [3]
                processed_obs["obs"_][offset++] = Sanitize(payload_statei(j));
            }
            for (int j = 3; j < 13; ++j)
            {
                // uav vel, quat, and ang vel (10)
                processed_obs["obs"_][offset++] = Sanitize(statei(j));
            }
        }
        

        for (const auto &act : action_history_)
        {   // action history [4]
            for (int i = 0; i<act_dim_; ++i){
                processed_obs["obs"_][offset++] = Sanitize(act(i)); // act is already in [-1, 1] range as recorded
            }
        }
        // privileged infos
        processed_obs["info:policy_obs_dim"_] = offset;
        if (privileged_obs_){
            // privileged infos
            for (size_t i = 0; i < state_history_size_; ++i){
                const auto& state_doti = state_dot_history_[i];
                const auto& payload_state_doti = payload_state_dot_history_[i];
                for (int j = 0; j < 13; ++j){
                    processed_obs["obs"_][offset++] = Sanitize(state_doti(j));
                }

                for (int j = 0; j < 13; ++j){
                    processed_obs["obs"_][offset++] = Sanitize(payload_state_doti(j));
                }
            }
            for (size_t i = 0; i< 6; ++i){
                processed_obs["obs"_][offset++] = Sanitize(mass_inertia_tc_obs_(i));
            }
        }
        processed_obs["info:privileged_obs_dim"_] = offset - processed_obs["info:policy_obs_dim"_];
    }

    void ProcessAction(const Action &policy_action,
                       std::unordered_map<std::string, Eigen::VectorXf> &processed_action) override
    {
        const float *action_data = static_cast<const float *>(policy_action["action"_].Data());
        processed_action[uav_state_key_].resize(act_dim_);

        Eigen::VectorXf raw_action(act_dim_);
        for (int i = 0; i < act_dim_; ++i)
        {
            raw_action(i) = action_data[i];
        }

        PushAction(raw_action);

        // float hover_vel = max_rotor_vel_ / 2.0f; // Approximate hover rotor velocity
        // float action_range = max_rotor_vel_ - hover_vel;
        for (int i = 0; i < act_dim_; ++i)
        {
            // Map [-1, 1] to [hover_vel - 900, max_rotor_vel_] properly centered
            // float scaled_vel = hover_vel + action_data[i] * action_range;
            processed_action[uav_state_key_](i) = Sanitize((raw_action(i) + 1.0) *max_rotor_thrust_/2.0);
        }
    }

    void ComputeReward(const std::unordered_map<std::string, Statef> &current_state,
                       const std::unordered_map<std::string, Statef> &current_state_dot,
                       const std::unordered_map<std::string, Statef> &previous_state,
                       const std::unordered_map<std::string, Statef> &previous_state_dot,
                       const Action &action,
                       std::unordered_map<std::string, float> &rewards) override
    {
        
        
        const Eigen::Vector3f current_payload_pos = current_state.at(payload_state_key_).head<3>();
        const Eigen::Vector3f previous_payload_pos = previous_state.at(payload_state_key_).head<3>();
        const Eigen::Vector3f current_uav_pos = current_state.at(uav_state_key_).head<3>();
        Eigen::Quaterniond quat(current_state.at(uav_state_key_)(6), current_state.at(uav_state_key_)(7),
         current_state.at(uav_state_key_)(8), current_state.at(uav_state_key_)(9));
        const Eigen::Matrix3d rotmat = quat.normalized().toRotationMatrix();
        // Reward is based on reduction in distance of payload to desired position
        float distance = (current_payload_pos - desired_payload_pos_).norm();
        float prev_distance = (previous_payload_pos - desired_payload_pos_).norm();
        float reward = (prev_distance - distance)*2.0;
        // alive bonus
        reward += 0.01f;
        // float reward = 1.0/(1.0 + 1.6*(current_payload_pos - desired_payload_pos_).squaredNorm());
        // float reward = 1.0f / (1.0f + distance*distance); 
        // Penalty for payload omega (encourage stability)
        // const Eigen::Vector3f payload_omega = current_state.at(payload_state_key_).segment<3>(10);
        // const Eigen::Vector3f uav_omega = current_state.at(uav_state_key_).segment<3>(10);
        // reward  = reward +  0.1/(1.0+ payload_omega.squaredNorm() + uav_omega.squaredNorm());
        // reward += 0.0001f /(1.0f + payload_omega.squaredNorm());
        // reward += 0.01f /(1.0f + uav_omega.squaredNorm());
        // Penalty for crash
        if (((current_uav_pos.cast<double>().cwiseAbs() - max_xyz_).array() > 0.0f).any())
            {
                reward -=  5.0;
            }

        const double cosang =
            std::abs(std::acos(std::clamp(rotmat.col(2).dot(Eigen::Vector3d::UnitZ()), -1.0, 1.0)));
        if (cosang >  1.57)
        {
            reward -=  5.0;
        }
        if (current_uav_pos(2) < 0.3){
            reward -= 5.0;
        }
        const double unit_z_dot = (current_payload_pos - current_uav_pos).normalized()(2);
        reward += unit_z_dot;
        if (unit_z_dot < 0.2)
        {
            reward -= 5.0;
        }
        rewards[uav_state_key_] = Sanitize(reward);
    }

private:
    void PushAction(const Eigen::VectorXf &action)
    {
        action_history_.push_back(action);
        if (static_cast<int>(action_history_.size()) > action_history_size_ )
            {
                action_history_.pop_front();
            }
    }
    void PushStates(const Eigen::VectorXf &state, const Eigen::VectorXf &payload_state,
        const Eigen::VectorXf &state_dot, const Eigen::VectorXf &payload_state_dot)
    {
        state_history_.push_back(state);
        if (static_cast<int>(state_history_.size()) > state_history_size_ )
            {
                state_history_.pop_front();
            }
        payload_state_history_.push_back(payload_state);
        if (static_cast<int>(payload_state_history_.size()) > state_history_size_ )
            {
                payload_state_history_.pop_front();
            }
        state_dot_history_.push_back(state_dot);
        if (static_cast<int>(state_dot_history_.size()) > state_history_size_){
            state_dot_history_.pop_front();
        }
        payload_state_dot_history_.push_back(payload_state_dot);
        if (static_cast<int>(payload_state_dot_history_.size()) > state_history_size_){
            payload_state_dot_history_.pop_front();
        }
    }

    template<typename T>
    T Sanitize(T val){
        T ret = val;
        if (std::isnan(val)){
            ret = 0.0;
        }
        else if (std::isinf(val)){
            ret = 1e5;
        }
        return ret;
    }
    int action_history_size_;
    int state_history_size_;
    int act_dim_;
    Eigen::VectorXf action_scaling_;
    double max_rotor_thrust_;
    std::string uav_state_key_;
    std::string payload_state_key_;
    std::deque<Eigen::VectorXf> action_history_;
    std::deque<Eigen::VectorXf> state_history_;
    std::deque<Eigen::VectorXf> payload_state_history_;
    std::deque<Eigen::VectorXf> state_dot_history_;
    std::deque<Eigen::VectorXf> payload_state_dot_history_;
    Eigen::Vector3f desired_payload_pos_;
    Eigen::VectorXd mass_inertia_tc_obs_;
    bool privileged_obs_{false};
    Eigen::Vector3d max_xyz_;
};

#endif
