// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Amal Dev Haridevan

#ifndef HOVER_SPEC_HH_
#define HOVER_SPEC_HH_

#include "env_specs/gazebo_envpool_spec.hh"
/** @class HoverSpec
 * @brief Specifies the Hover environment spec
 */
class HoverSpec : public GazeboSpec
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
                "uav_model_name"_.Bind(std::string("quadrotor")),
                "uav_base_link_name"_.Bind(std::string("quadrotor/base_link")),
                "sdf_file"_.Bind(std::string("world_hover.sdf")),
                "action_history_size"_.Bind(2),
                "max_steps_per_episode"_.Bind(20000),
                "domain_randomization"_.Bind(false),
                "physics_steps_per_control"_.Bind(1)));
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
        const int last_action_sz = conf["action_history_size"_] * 4;
        const int obs_dim = 14 + last_action_sz;
        return MakeDict("obs"_.Bind(Spec<float>({obs_dim}, {-fmax, fmax})));
    }

    /** @brief Returns the action spec for Hover env
     * Action space consists of:
     * Direct normalized rotor velocity commands (4)
     */
    template <typename Config>
    static decltype(auto) ActionSpec(const Config &conf)
    {
        const int act_dim = 4;
        return MakeDict("action"_.Bind(Spec<float>(
            {-1, act_dim}, {-1.0f, 1.0f})));
    }
};

using HoverEnvSpec = EnvSpec<HoverSpec>;
#endif
