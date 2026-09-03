// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Amal Dev Haridevan

/** Creates the bindings for envs defined inside
 *  gazebo_envpool directory
 */
#include "gazebo_envpool/gazebo_envpool.hh"
#include "gazebo_hover.hh"
#include "gazebo_trajectory_tracking.hh"
#include "gazebo_trajectory_tracking_ll.hh"
#include "gazebo_payload_transport_ll.hh"
#include "gazebo_payload_transport.hh"
#include "gazebo_inverted_pendulum_ll.hh"
#include "gazebo_multiagent_formation_ll.hh"
#include <pybind11/stl.h>

using hover_env::HoverEnv;
using inverted_pendulum_ll_env::InvertedPendulumLLEnv;
using multiagent_formation_ll_env::MultiAgentFormationLLEnv;
using payload_transport_env::PayloadTransportEnv;
using payload_transport_ll_env::PayloadTransportLLEnv;
using trajectory_tracking_env::TrajectoryTrackingEnv;
using trajectory_tracking_ll_env::TrajectoryTrackingLLEnv;

PYBIND11_MODULE(gazebo_envpool_envs, m)
{
    MAKE_PY_GAZEBO_ENVPOOL(m, HoverEnv, HoverEnvSpec)
    MAKE_PY_GAZEBO_ENVPOOL(m, TrajectoryTrackingEnv, TrajectoryTrackingEnvSpec)
    MAKE_PY_GAZEBO_ENVPOOL(m, TrajectoryTrackingLLEnv, TrajectoryTrackingLLEnvSpec)
    MAKE_PY_GAZEBO_ENVPOOL(m, PayloadTransportLLEnv, PayloadTransportLLEnvSpec)
    MAKE_PY_GAZEBO_ENVPOOL(m, PayloadTransportEnv, PayloadTransportEnvSpec)
    MAKE_PY_GAZEBO_ENVPOOL(m, InvertedPendulumLLEnv, InvertedPendulumLLEnvSpec)
    MAKE_PY_GAZEBO_ENVPOOL(m, MultiAgentFormationLLEnv, MultiAgentFormationLLEnvSpec)
}
