// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Amal Dev Haridevan

#ifndef PROCESSOR_HH_
#define PROCESSOR_HH_

#include <type_traits>
#include <vector>
#include <string>
#include <Eigen/Dense>
#include "envpool/core/async_envpool.h"
#include <map>

/**
 * @brief Base class for processors. This specifies the interface for all processors.
 * This class should be inherited by all processors.
 * The processor is responsible for processing the observations and actions.
 * The raw state [ pos, vel, quat, omega ] is processed into the observation space.
 * The action from the policy is processed to Feed into the environment.
 * 
 * @tparam EnvSpec  The Spec class of the environment for which this processor is created
 */
template <typename EnvSpec>
class GazeboProcessor
{
public:
    using State =
        Dict<typename EnvSpec::StateKeys,
             typename SpecToTArray<typename EnvSpec::StateSpec::Values>::Type>;
    using Action =
        Dict<typename EnvSpec::ActionKeys,
             typename SpecToTArray<typename EnvSpec::ActionSpec::Values>::Type>;
    using Statef = Eigen::Matrix<float, 13, 1>;
    
    /** @brief ProcessObservation processes the raw observation from the environment
     * into the observation space.
     * @param current_state The current state mapped by string keys of agents
     * @param current_state_dot The current state dot mapped by string keys of agents
     * @param previous_state The previous state mapped by string keys of agents
     * @param previous_state_dot The previous state dot mapped by string keys of agents
     * @param processed_obs The processed observation as a Dict with keys being
     * the observation keys and values being TArray to feed into the environment
     */
    virtual void ProcessObservation(const std::unordered_map<std::string, Statef> &current_state,
                                    const std::unordered_map<std::string, Statef> &current_state_dot,
                                    const std::unordered_map<std::string, Statef> &previous_state,
                                    const std::unordered_map<std::string, Statef> &previous_state_dot,
                                    State &processed_obs) = 0;
    /** @brief ProcessAction processes the action from the policy
     * into the action space of the environment.
     * @param policy_action The action from the policy (Envpool Action type)
     * @param processed_action The  processed action as a map of string keys of agents
     * with values being Eigen::VectorXf to feed into the environment
     */
    virtual void ProcessAction(const Action &policy_action, std::unordered_map<std::string, Eigen::VectorXf> &processed_action) = 0;

    /** @brief Computes the reward based on the current state, previous state, and action
     * @param current_state The current state mapped by string keys of agents
     * @param current_state_dot The current state dot mapped by string keys of agents
     * @param previous_state The previous state mapped by string keys of agents
     * @param previous_state_dot The previous state dot mapped by string keys of agents
     * @param action The action taken (which is the Envpool Action type, see envpool/core/asyn_envpool.h)
     * @param rewards The computed rewards for each agent, mapped by string keys of agents
     */
    virtual void ComputeReward(const std::unordered_map<std::string, Statef> &current_state,
                               const std::unordered_map<std::string, Statef> &current_state_dot,
                               const std::unordered_map<std::string, Statef> &previous_state,
                               const std::unordered_map<std::string, Statef> &previous_state_dot,
                               const Action &action,
                               std::unordered_map<std::string, float> &rewards) = 0;

    virtual ~GazeboProcessor() = default;

    /** @brief Reset method templated */
    template <typename... Args>
    void Reset(Args &&...args)
    {
        throw std::runtime_error("Reset method not implemented for this processor");
    }
    /** @brief Reset method without arguments */
    virtual void Reset()
    {
        throw std::runtime_error("Reset method not implemented for this processor");
    }
};

/** @brief A method to check if the processor is a child of GazeboProcessor */
template <typename T, typename EnvSpec>
using is_gazebo_processor = std::is_base_of<GazeboProcessor<EnvSpec>, T>;

#endif