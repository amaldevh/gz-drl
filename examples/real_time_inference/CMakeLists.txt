# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Amal Dev Haridevan

cmake_minimum_required(VERSION 3.22.1)
project(gzdrl_real_time_inference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 REQUIRED)
find_package(onnxruntime REQUIRED)
find_package(gzdrl 1.0 REQUIRED)

if(NOT TARGET gzdrl::rosrl_sitl)
    message(FATAL_ERROR
        "This example requires a GzDRL installation built with ROS SITL support. "
        "Source ROS before installing GzDRL.")
endif()

add_executable(gzdrl_onnx_inference src/main.cc)
add_executable(GzDRL::onnx_inference ALIAS gzdrl_onnx_inference)

target_include_directories(gzdrl_onnx_inference PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_link_libraries(gzdrl_onnx_inference PRIVATE
    Eigen3::Eigen
    onnxruntime::onnxruntime
    gzdrl::rl_server
    gzdrl::rosrl_sitl
)

# `hybrid_ros.cmake` is loaded by gzdrlConfig.cmake for ROS-enabled builds.
link_target_with_ros(gzdrl_onnx_inference)

get_target_property(
    ONNXRUNTIME_TARGET_LOCATION
    onnxruntime::onnxruntime
    IMPORTED_LOCATION
)
if(NOT ONNXRUNTIME_TARGET_LOCATION)
    get_target_property(
        ONNXRUNTIME_TARGET_LOCATION
        onnxruntime::onnxruntime
        IMPORTED_LOCATION_RELEASE
    )
endif()

set(_example_rpath "${gzdrl_LIBRARY_DIR}")
if(ONNXRUNTIME_TARGET_LOCATION)
    get_filename_component(
        ONNXRUNTIME_LIBRARY_DIR
        "${ONNXRUNTIME_TARGET_LOCATION}"
        DIRECTORY
    )
    list(APPEND _example_rpath "${ONNXRUNTIME_LIBRARY_DIR}")
endif()
set_target_properties(gzdrl_onnx_inference PROPERTIES
    BUILD_RPATH "${_example_rpath}"
)
