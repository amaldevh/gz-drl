// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Amal Dev Haridevan

#ifndef PAYLOAD_TRANSPORT_SPEC_HH_
#define PAYLOAD_TRANSPORT_SPEC_HH_

#include "env_specs/gazebo_envpool_spec.hh"
/** @class PayloadTransportSpec
 * @brief Specifies the PayloadTransport environment spec
 */
class PayloadTransportSpec : public GazeboSpec
{
public:
    /** @brief DefaultConfig returns the default config for PayloadTransport env
     */
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
                "domain_randomization"_.Bind(true),
                "physics_steps_per_control"_.Bind(10),
                "force_scaling"_.Bind(std::vector<float>{10.0f, 10.0f, 10.0f})));
    }
    /** @brief Returns the observation spec for PayloadTransport env
     * Observation consists of:
     * relative position of payload to desired pos (3),
     * relative position of payload to UAV (3),
     * quaternion (4), velocity (3), omega (3) of UAV,
     * payload linear vel (3), payload angular vel (3),
     * last actions (3 * action_history_size) = 22 + last_action
     * The action is a normalized residual world-frame force [Fx, Fy, Fz].
     */
    template <typename Config>
    static decltype(auto) StateSpec(const Config &conf)
    {
        float fmax = std::numeric_limits<float>::max();
        const int last_action_sz = conf["action_history_size"_] * 3;
        const int obs_dim = 22 + last_action_sz;
        return MakeDict("obs"_.Bind(Spec<float>({obs_dim}, {-fmax, fmax})));
    }
    /** @brief Returns the action spec for PayloadTransport env
     * Action space consists of:
     *  normalized residual world-frame force [Fx, Fy, Fz] (3)
     */
    template <typename Config>
    static decltype(auto) ActionSpec(const Config &conf)
    {
        return MakeDict("action"_.Bind(Spec<float>(
            {-1, 3}, {-1.0f, 1.0f})));
    }
};

using PayloadTransportEnvSpec = EnvSpec<PayloadTransportSpec>;
#endif
