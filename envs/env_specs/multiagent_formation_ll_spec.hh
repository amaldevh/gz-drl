// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Amal Dev Haridevan

#ifndef MULTIAGENT_FORMATION_LL_HH_
#define MULTIAGENT_FORMATION_LL_HH_

#include "env_specs/gazebo_envpool_spec.hh"
/** @class MultiagentFormationLLSpec
 * @brief Specifies the Hover environment spec
 */
class MultiAgentFormationLLSpec : public GazeboSpec
{
public:
    /** @brief DefaultConfig returns the default config for Hover env
     */
    static decltype(auto) DefaultConfig()
    {
        auto base_config = GazeboSpec::BaseGazeboConfig();
        return ConcatDict(
            base_config,
            MakeDict(
                "base_uav_model_name"_.Bind(std::string("quadrotor")),
                "uav_base_link_name"_.Bind(std::string("quadrotor/base_link")),
                "num_agents"_.Bind(2),
                "max_steps_per_episode"_.Bind(20000),
                "domain_randomization"_.Bind(true),
                "physics_steps_per_control"_.Bind(1),
                "max_rotor_vel"_.Bind(2300.0)));
    }
    /** @brief Returns the observation spec for Hover env
     * Observation consists of:
     * relative position to desired hover pos (3), yaw error (1),
     * quaternion (4), velocity (3), omega (3),
     * last actions (3 if xyz else 4 * action_history_size) = 14 + last_action
     */
    template <typename Config>
    static decltype(auto) StateSpec(const Config &conf)
    {
        float fmax = std::numeric_limits<float>::max();
        const int num_agents = conf["num_agents"_];
        // Self state: Target-relative pos(3), linear vel(3), rotmat(9), angular vel(3) => Total 18
        // Others state: Relative pos(3), pdist(1), rotmat(9), angular vel(3) => Total 16 per other agent (x 4 = 64)
        // Dimension: 18 + (NUM_AGENTS - 1) * 16 = 82
        const int obs_dim = 18 + (num_agents - 1) * 16;
        return MakeDict("obs"_.Bind(Spec<float>({num_agents, obs_dim}, {-fmax, fmax})));
    }

    /** @brief Returns the action spec for Hover env
     * Action space consists of:
     * Outer-loop commands: Fx  Fy Fz yaw_rate (4)
     */
    template <typename Config>
    static decltype(auto) ActionSpec(const Config &conf)
    {
        const int act_dim = 4;
        const int num_agents = conf["num_agents"_];
        return MakeDict("action"_.Bind(Spec<float>({num_agents, act_dim}, {-1.0f, 1.0f})));
    }
};

using MultiAgentFormationLLEnvSpec = EnvSpec<MultiAgentFormationLLSpec>;
#endif