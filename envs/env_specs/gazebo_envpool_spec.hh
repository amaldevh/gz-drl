// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Amal Dev Haridevan

#ifndef GAZEBO_ENVPOOL_SPEC_HH_
#define GAZEBO_ENVPOOL_SPEC_HH_

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include <string>
#include <type_traits>

/**
 * @brief
 *
 * @version 1.0.0
 * @author Amal Dev Haridevan (haridevanamaldev@gmail.com)
 * @date 2026-06-15
 * @copyright Copyright (c) 2026
 */
class GazeboSpec
{
    /** @brief Base Gazebo spec. All specs for Gazebo-Envpool envs must inherit this.
     */
public:
    /**
     * @brief Constructor
     *
     * @return decltype(auto)
     */
    static decltype(auto) BaseGazeboConfig()
    {

        return MakeDict("test_env"_.Bind(false),
                        "test_envid"_.Bind(0),
                        // Python registration supplies the installed package
                        // paths. Native users may pass explicit paths or set
                        // the corresponding Gazebo environment variables.
                        "resources_path"_.Bind(std::string{}),
                        "plugins_path"_.Bind(std::string{}),
                        "gz_partition_offset"_.Bind(0));
    }
};

/** @brief check if a spec class inherits from GazeboSpec   */
template <typename T>
using is_gazebo_spec = std::is_base_of<GazeboSpec, T>;

#endif
