# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Amal Dev Haridevan

cmake_minimum_required(VERSION 3.12)
# This CMakeLists is supposed to be included as a subdirectory in a parent CMake project.
# See ../CMakeLists.txt for usage.
project(sitl_libs VERSION 1.0.0)

# Global include directories
include_directories(include)

# ============================================================================
# Automatic ROS detection and configuration
# ============================================================================
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/determine_ros_version.cmake)
determine_ros_version()

set(GZDRL_ROS_MODE "AUTO" CACHE STRING
    "ROS integration mode: AUTO, ON, or OFF")
set_property(CACHE GZDRL_ROS_MODE PROPERTY STRINGS AUTO ON OFF)
string(TOUPPER "${GZDRL_ROS_MODE}" _gzdrl_ros_mode)
if(NOT _gzdrl_ros_mode MATCHES "^(AUTO|ON|OFF)$")
    message(FATAL_ERROR
        "GZDRL_ROS_MODE must be AUTO, ON, or OFF; got ${GZDRL_ROS_MODE}")
endif()

if(_gzdrl_ros_mode STREQUAL "OFF")
    set(_gzdrl_build_sitl_ros OFF)
    set(_gzdrl_ros_ver 0)
elseif(ROS_VERSION EQUAL 1 OR ROS_VERSION EQUAL 2)
    set(_gzdrl_build_sitl_ros ON)
    set(_gzdrl_ros_ver "${ROS_VERSION}")
elseif(_gzdrl_ros_mode STREQUAL "ON")
    message(FATAL_ERROR
        "GZDRL_ROS_MODE=ON requires a sourced ROS 1 or ROS 2 environment")
else()
    set(_gzdrl_build_sitl_ros OFF)
    set(_gzdrl_ros_ver 0)
endif()

# Derived values are forced on every configure so a reused scikit-build
# directory cannot retain a previous ROS environment.
set(BUILD_SITL_ROS "${_gzdrl_build_sitl_ros}" CACHE INTERNAL
    "Whether GzDRL SITL is built with ROS" FORCE)
set(ROS_VER "${_gzdrl_ros_ver}" CACHE INTERNAL
    "Detected ROS major version" FORCE)

if (BUILD_SITL_ROS)
    # ============================================================================
    # Add compile definitions, so that macros get evaluated
    add_compile_definitions(ROS_VER=${ROS_VER})

    # set pkgs
     set(ROS_PKGS "std_msgs;sensor_msgs;geometry_msgs;nav_msgs;tf2;tf2_ros" 
        CACHE STRING "Semicolon-separated list of ROS packages")
    
        # Package information for package.xml generation
    set(PKG_MAINTAINER_NAME "Amal Dev Haridevan" CACHE STRING "Package maintainer name")
    set(PKG_MAINTAINER_EMAIL "haridevanamaldev@gmail.com" CACHE STRING "Package maintainer email")
    set(PKG_DESCRIPTION "Hybrid ROS package supporting ROS 1 and ROS 2" CACHE STRING "Package description")
    set(PKG_LICENSE "MIT" CACHE STRING "Package license")
    set(GENERATE_PACKAGE_XML ON CACHE BOOL "Automatically generate package.xml")

    # Convert string to list if needed
    if(ROS_PKGS)
        string(REPLACE " " ";" ROS_PKGS "${ROS_PKGS}")
    endif()

    #  Add associated ros client lib for C++
    if(ROS_VER EQUAL 1)

        # add roscpp
        set(ROS_PKGS "roscpp;${ROS_PKGS}")
    elseif(ROS_VER EQUAL 2)
        # add rclcpp
        set(ROS_PKGS "rclcpp;${ROS_PKGS}")
    else()
        message(FATAL_ERROR "Invalid ROS_VER: ${ROS_VER}. Must be 1 or 2.")
    endif()

    message(STATUS "ROS_VER: ${ROS_VER}")
    message(STATUS "ROS_PKGS: ${ROS_PKGS}")

    # ============================================================================
    # Include Package.xml Generation Module
    # ============================================================================
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/generate_package_xml.cmake)

    # ============================================================================
    # Generate package.xml if requested
    # ============================================================================
    if(GENERATE_PACKAGE_XML)
        message(STATUS "Generating package.xml...")
        
        generate_ros_package_xml(
            PACKAGE_NAME "${PROJECT_NAME}"
            VERSION "${PROJECT_VERSION}"
            DESCRIPTION "${PKG_DESCRIPTION}"
            MAINTAINER_NAME "${PKG_MAINTAINER_NAME}"
            MAINTAINER_EMAIL "${PKG_MAINTAINER_EMAIL}"
            LICENSE "${PKG_LICENSE}"
            ROS_VERSION "${ROS_VER}"
            ROS_PACKAGES ${ROS_PKGS}
            # Optional: Add additional dependencies
            # BUILD_DEPENDS "eigen3" "boost"
            # EXEC_DEPENDS "python3-numpy"
            # TEST_DEPENDS "gtest"
        )
    endif()

    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/hybrid_ros.cmake)

endif()
# ============================================================================
# Common targets, regardless of ROS  
# ============================================================================
pybind11_add_module(sitl_core
        python/sitl.cc
        )
set_target_properties(sitl_core PROPERTIES OUTPUT_NAME "_core")
if (BUILD_SITL_ROS)
    # add the rosrl sitl module
    add_library(rosrl_sitl SHARED
        src/ros_rl_server.cc
        )
    # link with rl_server - PUBLIC so consumers get transitive deps
    target_link_libraries(rosrl_sitl PUBLIC 
        rl_server
        )
    # Export include directories for downstream consumers
    target_include_directories(rosrl_sitl PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${INCLUDE_INSTALL_DIR}>
        )
    # link with sitl 
    target_link_libraries(sitl_core PRIVATE
        rosrl_sitl
        )
endif()
# # ROS specific configuration
# if (BUILD_SITL_ROS)

#     # Create test
#     add_executable(test_node ${CMAKE_CURRENT_SOURCE_DIR}/../tests/performance_tests/cpp/sitl_ros_test.cc)

#     # Link with ROS (this handles everything)
#     link_target_with_ros(test_node)
#     target_include_directories(test_node PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
# endif()

# Set the exported target names for use in the parent CMakeLists.txt
set(SITL_PYTHON_TARGETS 
    sitl_core
    )
set(SITL_NATIVE_TARGETS 
    )
if (BUILD_SITL_ROS)
    list(APPEND SITL_NATIVE_TARGETS 
    # test_node
    rosrl_sitl
    )
endif()

# Setup common include directories
# Native targets use PUBLIC so downstream consumers get include dirs
foreach( targeti ${SITL_NATIVE_TARGETS})
    target_include_directories(${targeti} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${INCLUDE_INSTALL_DIR}>
        )
    if (BUILD_SITL_ROS)
        target_compile_definitions(${targeti} PUBLIC ROS_VER=${ROS_VER})
        link_target_with_ros(${targeti})
    endif()
endforeach()
# Python targets use PRIVATE (not intended for C++ linking)
foreach( targeti ${SITL_PYTHON_TARGETS})
    target_include_directories(${targeti} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        )
    if (BUILD_SITL_ROS)
        target_compile_definitions(${targeti} PRIVATE ROS_VER=${ROS_VER})
        link_target_with_ros(${targeti})
    endif()
endforeach()
set(SITL_NATIVE_TARGETS ${SITL_NATIVE_TARGETS} PARENT_SCOPE)
set(SITL_PYTHON_TARGETS ${SITL_PYTHON_TARGETS} PARENT_SCOPE)

# if (BUILD_SITL_ROS) 
#     if(ROS_VER EQUAL 2)
#         # ament_export_dependencies(${ROS_DEPENDENCIES})
#         # Call ament_package() at the end
#         set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/install)
#         ament_package()
#     endif()
# endif()
# ============================================================================
