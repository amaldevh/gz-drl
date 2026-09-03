// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Amal Dev Haridevan

#ifndef TRAJECTORY_TRACKING_LL_SPEC_HH_
#define TRAJECTORY_TRACKING_LL_SPEC_HH_

#include "env_specs/gazebo_envpool_spec.hh"

#include <vector>

/** Direct four-rotor-velocity trajectory-tracking environment. */
class TrajectoryTrackingLLSpec : public GazeboSpec
{
public:
    static constexpr int kActionDimension = 4;

    static decltype(auto) DefaultConfig()
    {
        auto base_config = GazeboSpec::BaseGazeboConfig();
        return ConcatDict(
            base_config,
            MakeDict(
                "uav_model_name"_.Bind(std::string("quadrotor")),
                "uav_base_link_name"_.Bind(
                    std::string("quadrotor/base_link")),
                "sdf_file"_.Bind(
                    std::string("world_trajectory_tracking.sdf")),
                "max_steps_per_episode"_.Bind(3000),
                "domain_randomization"_.Bind(false),
                "use_rotation_matrix"_.Bind(true),
                "state_history_len"_.Bind(20),
                "future_waypoint_num"_.Bind(20),
                "physics_steps_per_control"_.Bind(10),
                "physics_dt"_.Bind(1.0e-3f),
                "trajectory_xy_amplitude_min"_.Bind(1.35f),
                "trajectory_xy_amplitude_max"_.Bind(1.75f),
                "trajectory_z_amplitude_min"_.Bind(0.3f),
                "trajectory_z_amplitude_max"_.Bind(0.4f),
                "trajectory_height_offset_min"_.Bind(0.7f),
                "trajectory_height_offset_max"_.Bind(1.0f),
                "trajectory_speed_min"_.Bind(1.0f),
                "trajectory_speed_max"_.Bind(1.5f),
                "trajectory_max_normal_acceleration"_.Bind(4.0f),
                "trajectory_vertical_position_weight"_.Bind(2.0f),
                "trajectory_vertical_velocity_weight"_.Bind(0.04f),
                "trajectory_fixed_zero_yaw_probability"_.Bind(0.5f),
                "trajectory_sampling_attempts"_.Bind(64),
                "min_rotor_vel"_.Bind(0.0f),
                "max_rotor_vel"_.Bind(2300.0f)));
    }

    template <typename Config>
    static decltype(auto) StateSpec(const Config &conf)
    {
        const float fmax = std::numeric_limits<float>::max();
        const bool use_rotation = conf["use_rotation_matrix"_];
        const int history_features = use_rotation ? 20 : 11;
        const int reference_features = use_rotation ? 15 : 6;
        const int observation_dimension =
            conf["state_history_len"_] * history_features +
            conf["future_waypoint_num"_] * reference_features;
        return MakeDict(
            "obs"_.Bind(Spec<float>(
                {observation_dimension}, {-fmax, fmax})),
            "info:state"_.Bind(Spec<float>({13}, {-fmax, fmax})),
            "info:trajectory"_.Bind(
                Spec<float>({13}, {-fmax, fmax})));
    }

    template <typename Config>
    static decltype(auto) ActionSpec(const Config &conf)
    {
        (void)conf;
        return MakeDict(
            "action"_.Bind(Spec<float>(
                {-1, kActionDimension}, {-1.0f, 1.0f})));
    }

};

using TrajectoryTrackingLLEnvSpec = EnvSpec<TrajectoryTrackingLLSpec>;

#endif // TRAJECTORY_TRACKING_LL_SPEC_HH_
