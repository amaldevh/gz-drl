# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Amal Dev Haridevan

# Macro to determine ROS version (1 or 2) based on distribution name
macro(determine_ros_version)
    # Get ROS distribution from environment
    if(DEFINED ENV{ROS_DISTRO})
        set(ROS_DISTRO $ENV{ROS_DISTRO})

        # Define known ROS 1 distributions (complete list - no new ROS 1 releases)
        set(ROS1_DISTROS
            kinetic
            lunar
            melodic
            noetic
        )

        # Check if distribution is ROS 1
        if(ROS_DISTRO IN_LIST ROS1_DISTROS)
            set(ROS_VERSION 1)
            message(STATUS "Detected ROS 1 (${ROS_DISTRO})")
        else()
            # Any other distro is assumed to be ROS 2
            set(ROS_VERSION 2)
            message(STATUS "Detected ROS 2 (${ROS_DISTRO})")
        endif()
    else()
        message(STATUS "ROS_DISTRO environment variable is not set.
         Please source your ROS setup file to build ros components.")
        set(ROS_VERSION 0)
    endif()
endmacro()
