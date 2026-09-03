# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Amal Dev Haridevan

cmake_minimum_required(VERSION 3.22.1)

find_program(GCC_11 gcc-11)
find_program(GXX_11 g++-11)
if(NOT GCC_11 OR NOT GXX_11)
    message(FATAL_ERROR "gcc-11 and g++-11 not found. Please install them.")
endif()
set(CMAKE_C_COMPILER ${GCC_11})
set(CMAKE_CXX_COMPILER ${GXX_11})

project(nmpc_controller_example LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mtune=native")
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(USE_XLA OFF CACHE BOOL "Use XLA Features, Requires CUDA")
# Set compile definitions based on options
if(USE_XLA)
    add_compile_definitions(USE_XLA=1)
endif()


# Find required packages
find_package(Eigen3 REQUIRED)

find_package(gz-sim9 REQUIRED 
    COMPONENTS 
    gui 
    gz )
find_package(gz-rendering9 REQUIRED)
find_package(gz-common6 REQUIRED)
find_package(gz-plugin3 REQUIRED 
    COMPONENTS 
    register)
find_package(gz-sensors9 REQUIRED 
    COMPONENTS 
    camera 
    boundingbox_camera 
    depth_camera 
    gpu_lidar lidar 
    rgbd_camera 
    segmentation_camera
    thermal_camera 
    wide_angle_camera)
find_package(sdformat15 REQUIRED)
find_package(Python 3.10...<3.14 REQUIRED
    COMPONENTS 
    Interpreter 
    Development)
find_package(pybind11 CONFIG)
find_package(glog REQUIRED)
if (USE_XLA)
    find_package(CUDAToolkit REQUIRED)
endif()
find_package(yaml-cpp REQUIRED)
find_package(gzdrl REQUIRED)



if ("$ENV{ACADOS_SOURCE_DIR}" STREQUAL "")
    message(FATAL_ERROR "ACADOS_SOURCE_DIR env var not set" )
endif()
# Path to acados installation
set(ACADOS_INSTALL_DIR "$ENV{ACADOS_SOURCE_DIR}")

# Path to generated acados code (from Python script)
set(ACADOS_GENERATED_DIR "/tmp/c_generated_code_ocp")

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${ACADOS_INSTALL_DIR}/include
    ${ACADOS_INSTALL_DIR}/include/blasfeo/include
    ${ACADOS_INSTALL_DIR}/include/hpipm/include
    ${ACADOS_GENERATED_DIR}
    ${EIGEN3_INCLUDE_DIR}
    ${gzdrl_INCLUDE_DIRS}
    
    )
# Note: gzdrl/include and Eigen3 are automatically included via rl_server PUBLIC interface

# Link directories
link_directories(
    ${ACADOS_INSTALL_DIR}/lib
    ${ACADOS_GENERATED_DIR}
    ${gzdrl_LIBRARY_DIR}
)

# Collect all generated acados C files
file(GLOB ACADOS_MODEL_SRC "${ACADOS_GENERATED_DIR}/quadrotor_model/*.c")
file(GLOB ACADOS_COST_SRC "${ACADOS_GENERATED_DIR}/quadrotor_cost/*.c")

# NMPC Controller library
add_library(nmpc_controller SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/src/nmpc_controller.cc
    ${ACADOS_GENERATED_DIR}/acados_solver_quadrotor.c
    ${ACADOS_MODEL_SRC}
    ${ACADOS_COST_SRC}
)

target_link_libraries(nmpc_controller
    acados
    hpipm
    blasfeo
)

# Main executable
add_executable(nmpc_test ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cc)

target_link_libraries(nmpc_test
    nmpc_controller
    gzdrl::uav_controllers
    gzdrl::rl_server
    acados
    hpipm
    blasfeo
    gz-sim9::gz-sim9
    sdformat15::sdformat15
    gz-common6::gz-common6
    gz-plugin3::gz-plugin3
    gz-sensors9::gz-sensors9
    sdformat15::sdformat15
)

# Set RPATH for finding libraries at runtime
set_target_properties(nmpc_test PROPERTIES
    BUILD_RPATH "${gzdrl_LIBRARY_DIR}:${ACADOS_INSTALL_DIR}/lib:${ACADOS_GENERATED_DIR}"
    INSTALL_RPATH "${gzdrl_LIBRARY_DIR}:${ACADOS_INSTALL_DIR}/lib:${ACADOS_GENERATED_DIR}"
)

set_target_properties(nmpc_controller PROPERTIES
    BUILD_RPATH "${ACADOS_INSTALL_DIR}/lib:${ACADOS_GENERATED_DIR}"
    INSTALL_RPATH "${ACADOS_INSTALL_DIR}/lib:${ACADOS_GENERATED_DIR}"
)

message(STATUS "NMPC Controller Example Configuration:")
message(STATUS "  Acados install dir: ${ACADOS_INSTALL_DIR}")
message(STATUS "  Acados generated dir: ${ACADOS_GENERATED_DIR}")
message(STATUS "  Eigen3 include dir: ${EIGEN3_INCLUDE_DIR}")
