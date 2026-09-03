// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Amal Dev Haridevan

#ifndef PAYLOAD_TRANSPORT_LL_SPEC_HH_
#define PAYLOAD_TRANSPORT_LL_SPEC_HH_

#include "env_specs/gazebo_envpool_spec.hh"
/** @class PayloadTransportLLSpec
 * @brief Specifies the Payload Transport environment spec
 */
class PayloadTransportLLSpec : public GazeboSpec
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
                "uav_base_link_name"_.Bind(std::string("quadrotor/base_link")),
                "sdf_file"_.Bind(std::string("world_payload.sdf")),
                "action_history_size"_.Bind(1),
                "max_steps_per_episode"_.Bind(1000),
                "physics_steps_per_control"_.Bind(10),
                "max_rotor_vel"_.Bind(2246.0f),
                "domain_randomization"_.Bind(true)));
    }

    template <typename Config>
    static decltype(auto) StateSpec(const Config &conf)
    {
        float fmax = std::numeric_limits<float>::max();
        const int last_action_sz = conf["action_history_size"_] * 4;
        const int obs_dim = 22 + last_action_sz;
        return MakeDict("obs"_.Bind(Spec<float>({obs_dim}, {-fmax, fmax})));
    }

    template <typename Config>
    static decltype(auto) ActionSpec(const Config &conf)
    {
        return MakeDict("action"_.Bind(Spec<float>({-1, 4}, {-1.0f, 1.0f})));
    }
};

using PayloadTransportLLEnvSpec = EnvSpec<PayloadTransportLLSpec>;
#endif
