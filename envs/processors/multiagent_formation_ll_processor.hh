// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Amal Dev Haridevan

#ifndef MULTIAGENT_FORMATION_LL_PROCESSOR_HH_
#define MULTIAGENT_FORMATION_LL_PROCESSOR_HH_

#include "processors/processor.hh"
#include "print_utils.hh"
#include "common_utils.hh"

/** @brief MultiAgentFormationLLProcessor
 * Processor for Hover environment
 */
template <typename EnvSpec>
class MultiAgentFormationLLProcessor : public GazeboProcessor<EnvSpec>
{
public:
    using State = typename GazeboProcessor<EnvSpec>::State;
    using Action = typename GazeboProcessor<EnvSpec>::Action;
    /** @brief Constructor
     * @param action_history_size The size of the action history
     * @param act_dim Action dim (3 if xyz, 4 if xyz-yaw)
     * @param state_key The UAV state key
     */
    MultiAgentFormationLLProcessor(int num_agents,
                                   int act_dim,
                                   float max_rotor_vel,
                                   std::vector<std::string> state_keys) : num_agents_(num_agents),
                                                                          act_dim_(act_dim),
                                                                          max_rotor_vel_(max_rotor_vel),
                                                                          state_keys_(state_keys)
    {
        action_scaling_.resize(act_dim_);
        if (act_dim_ != 4)
            throw std::runtime_error("MultiAgentFormationLLProcessor only supports act_dim = 4");
        action_scaling_ << max_rotor_vel_, max_rotor_vel_, max_rotor_vel_, max_rotor_vel_;

        target_formation_.resize(num_agents_, 3);
        target_formation_ = GenerateTargetFormation(num_agents_);
        print_info("Target formation:\n", target_formation_);
    }
    /** @brief Resets the processor state
     *
     * @param target_pos The desired target position
     * as xyz (3)
     */
    void Reset(const Eigen::Vector3f &target_pos)
    {
        desired_target_pos_ = target_pos;
        print_info("Target pos: ", desired_target_pos_.transpose());
        done_ = false;
    }

    /** @brief ProcessObservation processes the raw observation from the environment */
    void ProcessObservation(const std::unordered_map<std::string, Statef> &current_state,
                            const std::unordered_map<std::string, Statef> &current_state_dot,
                            const std::unordered_map<std::string, Statef> &previous_state,
                            const std::unordered_map<std::string, Statef> &previous_state_dot,
                            State &processed_obs) override
    {
        for (int i_ = 0; i_ < num_agents_; ++i_)
        {
            auto &state_key = state_keys_[i_];
            State &state = processed_obs;
            int obs_idx = 0;

            // 1. Self State (18 dims)
            // Pos relative to target (3)
            Eigen::Vector3f pos = current_state.at(state_key).segment<3>(0);
            Eigen::Vector3f rel_pos = desired_target_pos_ - pos;
            state["obs"_](i_, obs_idx++) = rel_pos(0);
            state["obs"_](i_, obs_idx++) = rel_pos(1);
            state["obs"_](i_, obs_idx++) = rel_pos(2);

            // Linear velocity (3)
            state["obs"_](i_, obs_idx++) = current_state.at(state_key)(3);
            state["obs"_](i_, obs_idx++) = current_state.at(state_key)(4);
            state["obs"_](i_, obs_idx++) = current_state.at(state_key)(5);
            // Rotmat (9)
            Eigen::Quaternionf q(current_state.at(state_key)(6), current_state.at(state_key)(7), current_state.at(state_key)(8), current_state.at(state_key)(9));
            q.normalize();
            auto R = q.toRotationMatrix();
            for (int r = 0; r < 3; ++r)
            {
                for (int c = 0; c < 3; ++c)
                {
                    state["obs"_](i_, obs_idx++) = R(r, c);
                }
            }

            // Angular vel (3)
            state["obs"_](i_, obs_idx++) = current_state.at(state_key)(10);
            state["obs"_](i_, obs_idx++) = current_state.at(state_key)(11);
            state["obs"_](i_, obs_idx++) = current_state.at(state_key)(12);

            // 2. Others State ((NUM_AGENTS - 1) * 16 dims = 64)
            for (int j_ = 0; j_ < num_agents_; ++j_)
            {
                if (i_ == j_)
                    continue;
                auto &other_state_key = state_keys_[j_];
                Eigen::Vector3f other_pos = current_state.at(other_state_key).segment<3>(0);

                // Relative position mapping self to other (3)
                Eigen::Vector3f p_rel = other_pos - pos;
                state["obs"_](i_, obs_idx++) = p_rel(0);
                state["obs"_](i_, obs_idx++) = p_rel(1);
                state["obs"_](i_, obs_idx++) = p_rel(2);

                // Pairwise distance (1)
                state["obs"_](i_, obs_idx++) = p_rel.norm();

                // Other velocity (3)
                state["obs"_](i_, obs_idx++) = current_state.at(other_state_key)(3);
                state["obs"_](i_, obs_idx++) = current_state.at(other_state_key)(4);
                state["obs"_](i_, obs_idx++) = current_state.at(other_state_key)(5);

                // Other rotmat (9)
                Eigen::Quaternionf oq(current_state.at(other_state_key)(6), current_state.at(other_state_key)(7), current_state.at(other_state_key)(8), current_state.at(other_state_key)(9));
                oq.normalize();
                auto oR = oq.toRotationMatrix();
                for (int r = 0; r < 3; ++r)
                {
                    for (int c = 0; c < 3; ++c)
                    {
                        state["obs"_](i_, obs_idx++) = oR(r, c);
                    }
                }
            }
        }
    }

    /* @brief ProcessAction processes the raw action before sending to the environment  */
    void ProcessAction(const Action &policy_action,
                       std::unordered_map<std::string, Eigen::VectorXf> &processed_action) override
    {
        for (int i_ = 0; i_ < num_agents_; ++i_)
        {
            processed_action[state_keys_[i_]].resize(act_dim_);
            processed_action[state_keys_[i_]] << policy_action["action"_][i_][0] + 1.0f,
                policy_action["action"_][i_][1] + 1.0f, policy_action["action"_][i_][2] + 1.0f,
                policy_action["action"_][i_][3] + 1.0f;
            processed_action[state_keys_[i_]] = (processed_action[state_keys_[i_]]) / 2.0f * max_rotor_vel_;
        }
    }
    /** @brief Reward function */
    void ComputeReward(const std::unordered_map<std::string, Statef> &current_state,
                       const std::unordered_map<std::string, Statef> &current_state_dot,
                       const std::unordered_map<std::string, Statef> &previous_state,
                       const std::unordered_map<std::string, Statef> &previous_state_dot,
                       const Action &action,
                       std::unordered_map<std::string, float> &rewards) override
    {

        // Calculate 0-mean current formation positions
        Eigen::Vector3f sum_pos = Eigen::Vector3f::Zero();
        for (int i = 0; i < num_agents_; ++i)
        {
            auto &state_key = state_keys_[i];
            sum_pos += current_state.at(state_key).segment<3>(0);
        }
        Eigen::Vector3f mean_pos = sum_pos / static_cast<float>(num_agents_);
        std::vector<Eigen::Vector3f> centered_pos(num_agents_);
        for (int i = 0; i < num_agents_; ++i)
        {
            centered_pos[i] = current_state.at(state_keys_[i]).segment<3>(0) - mean_pos;
        }

        // Directed Hausdorff (pos to desired)
        float d1 = 0;
        for (int i = 0; i < num_agents_; ++i)
        {
            float min_dist = std::numeric_limits<float>::max();
            for (int j = 0; j < num_agents_; ++j)
            {
                float dist = (centered_pos[i].transpose() - target_formation_.row(j)).norm();
                min_dist = std::min(min_dist, dist);
            }
            d1 = std::max(d1, min_dist);
        }
        // Directed Hausdorff (desired to pos)
        float d2 = 0;
        for (int i = 0; i < num_agents_; ++i)
        {
            float min_dist = std::numeric_limits<float>::max();
            for (int j = 0; j < num_agents_; ++j)
            {
                float dist = (target_formation_.row(i) - centered_pos[j].transpose()).norm();
                min_dist = std::min(min_dist, dist);
            }
            d2 = std::max(d2, min_dist);
        }

        float cost_h = std::max(d1, d2);

        float distance_to_target = (mean_pos - desired_target_pos_).norm();

        float reward_formation = 1.0f / (1.0f + static_cast<float>(std::pow(cost_h * 1.6f, 2)));
        float reward_pos = std::exp(-distance_to_target);

        // Calculate separation and heading
        float min_separation = std::numeric_limits<float>::max();
        std::vector<float> headings(num_agents_);
        bool misbehave_alt = false;

        for (int i = 0; i < num_agents_; ++i)
        {

            // Heading logic (from rotmat col 0)
            Eigen::Quaternionf q(current_state.at(state_keys_[i])(6), current_state.at(state_keys_[i])(7), current_state.at(state_keys_[i])(8), current_state.at(state_keys_[i])(9));
            q.normalize();
            auto R = q.toRotationMatrix();
            headings[i] = R(0, 0); // approx to x-axis alignment

            for (int j = i + 1; j < num_agents_; ++j)
            {
                float dist = (centered_pos[i] - centered_pos[j]).norm();
                min_separation = std::min(min_separation, dist);
            }
            misbehave_alt = (misbehave_alt || (std::acos(R(2, 2)) > 1.57));
        }

        float reward_separation_val = std::clamp(static_cast<float>(std::pow(min_separation / 0.5, 2)), 0.0f, 1.0f);
        // if (min_separation < 0.23 || misbehave_alt)
        // {
        //     done_ = true;
        //     std::cout << "Min sep: " << min_separation << " Misbehave alt: " << misbehave_alt << std::endl;
        // }

        float mean_heading = 0.0f;
        for (float h : headings)
            mean_heading += h;
        mean_heading /= num_agents_;
        float reward_heading = mean_heading; // To align with [-1, 0, 0] or similar target. Original takes mean of heading[..., 0].

        float combined_reward = reward_separation_val * (reward_formation + reward_formation * (reward_pos + reward_heading) + 0.4f * reward_pos);

        for (int i = 0; i < num_agents_; ++i)
        {
            rewards[state_keys_[i]] = combined_reward;
        }
    }

private:
    // Helper algorithm to programmatically establish robust scale-invariant base nodes
    Eigen::MatrixXf GenerateTargetFormation(int n_agents) const
    {
        if (n_agents < 2 || n_agents > 20)
        {
            throw std::invalid_argument("Formation size N must be internally bound within [2, 6]");
        }

        Eigen::MatrixXf target(n_agents, 3);
        target.setZero();

        // N < 5 is purely an exterior envelope. N >= 5 embeds a center anchor natively.
        bool has_center = (n_agents >= 5);
        int perimeter_agents = has_center ? (n_agents - 1) : n_agents;
        int start_idx = has_center ? 1 : 0;

        // Ensure the scale keeps inter-agent displacement close to ~1.4 to 2m
        float radius = 1.0f;

        for (int i = 0; i < perimeter_agents; ++i)
        {
            // Inscribe polygon geometrically
            float angle = static_cast<float>(i) * (2.0f * M_PI / static_cast<float>(perimeter_agents));
            target(start_idx + i, 0) = radius * std::cos(angle);
            target(start_idx + i, 1) = radius * std::sin(angle);
            target(start_idx + i, 2) = 0.0f;
        }

        if (has_center)
        {
            target.row(0).setZero();
        }

        // Re-center against the exact statistical mean to zero-bias the operational centroid
        Eigen::Vector3f center = target.colwise().mean();
        for (int i = 0; i < n_agents; ++i)
        {
            target.row(i) -= center.transpose();
        }

        return target;
    }

    inline float wrap_angle_(float a)
    {
        while (a > M_PI)
            a -= 2.0 * M_PI;
        while (a <= -M_PI)
            a += 2.0 * M_PI;
        return a;
    }
    int num_agents_;
    int act_dim_;
    Eigen::VectorXf action_scaling_;
    std::vector<std::string> state_keys_;
    std::deque<float> action_history_;
    Eigen::Vector3f desired_target_pos_;
    float max_rotor_vel_{0.0f};
    Eigen::MatrixXf target_formation_;

public:
    bool done_ = false;
};
#endif