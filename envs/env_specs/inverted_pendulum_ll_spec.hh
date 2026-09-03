// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Amal Dev Haridevan

#ifndef INVERTED_PENDULUM_LL_SPEC_HH_
#define INVERTED_PENDULUM_LL_SPEC_HH_

#include "env_specs/gazebo_envpool_spec.hh"

/** @class InvertedPendulumLLSpec
 * @brief Specifies the Inverted Pendulum environment spec
 */
class InvertedPendulumLLSpec : public GazeboSpec
{
public:
    static decltype(auto) DefaultConfig()
    {
        auto base_config = GazeboSpec::BaseGazeboConfig();
        return ConcatDict(
            base_config,
            MakeDict(
                "payload_link_name"_.Bind(std::string("payload")),
                "uav_model_name"_.Bind(std::string("quadrotor")),
                "rotor_link_names"_.Bind(std::vector<std::string>{std::string("quadrotor/rotor_0"),
                     std::string("quadrotor/rotor_1"), std::string("quadrotor/rotor_2"), std::string("quadrotor/rotor_3")}),
                "ktau"_.Bind(1.0/68.0),
                "turning_directions"_.Bind(std::vector<int>{1, -1, -1, 1}),
                "uav_base_link_name"_.Bind(std::string("quadrotor/base_link")),
                "sdf_file"_.Bind(std::string("world_inverted_pendulum.sdf")),
                "action_history_size"_.Bind(10),
                "state_history_size"_.Bind(10),
                "max_steps_per_episode"_.Bind(1000),
                "physics_steps_per_control"_.Bind(10),
                "max_rotor_thrust"_.Bind(12.0f),
                "domain_randomization"_.Bind(false),
            "privileged_obs"_.Bind(true)));
    }

    template <typename Config>
    static decltype(auto) StateSpec(const Config &conf)
    {
        float fmax = std::numeric_limits<float>::max();
        const int last_action_sz = conf["action_history_size"_] * 4;
        const int uav_state_history_size = conf["state_history_size"_]*10; // vel. quat, and omega
        const int payload_state_history_size = conf["state_history_size"_]*12; // rel pos to target,  rel pos to uav, vel and omega
        const int obs_dim = uav_state_history_size + payload_state_history_size + last_action_sz; 
        int privileged_info_dim = 6 +  conf["state_history_size"_]*13*2; // mass + inertia idag + time constants + last 13 state dots for payload and uav
        if (!conf["privileged_obs"_]){
            privileged_info_dim = 0;
        }
        return MakeDict("obs"_.Bind(Spec<float>({obs_dim + privileged_info_dim}, {-fmax, fmax})),
                "info:policy_obs_dim"_.Bind(Spec<float>({1}, {-fmax, fmax})),
                "info:privileged_obs_dim"_.Bind(Spec<float>({1}, {-fmax, fmax})));
    }

    template <typename Config>
    static decltype(auto) ActionSpec(const Config &conf)
    {
        return MakeDict("action"_.Bind(Spec<float>({-1, 4}, {-1.0f, 1.0f})));
    }
};

using InvertedPendulumLLEnvSpec = EnvSpec<InvertedPendulumLLSpec>;
#endif
