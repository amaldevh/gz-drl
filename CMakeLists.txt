# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Amal Dev Haridevan

cmake_minimum_required(VERSION 3.22.1 FATAL_ERROR)

option(GZDRL_FORCE_GCC_11 "Build with the supported GCC 11 toolchain" ON)
if(GZDRL_FORCE_GCC_11)
    find_program(GCC_11 gcc-11)
    find_program(GXX_11 g++-11)
    if(NOT GCC_11 OR NOT GXX_11)
        message(FATAL_ERROR
            "gcc-11 and g++-11 are required by GZDRL_FORCE_GCC_11. "
            "Install them or configure with -DGZDRL_FORCE_GCC_11=OFF.")
    endif()
    set(CMAKE_C_COMPILER "${GCC_11}")
    set(CMAKE_CXX_COMPILER "${GXX_11}")
endif()

project(gzdrl 
    VERSION 1.0.0
    LANGUAGES CXX)

include(CTest)

# Install a complete Python package when built by scikit-build-core, while
# retaining the checkout-local _lib prefix used by the native CMake workflow.
if(SKBUILD)
    set(LIB_INSTALL_DIR "gzdrl/lib")
    set(INCLUDE_INSTALL_DIR "gzdrl/include")
    set(CMAKE_INSTALL_DIR "gzdrl/cmake/gzdrl")
else()
    if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT OR
       "${CMAKE_INSTALL_PREFIX}" MATCHES "/usr/local")
        set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/_lib"
            CACHE PATH "Install prefix" FORCE)
    endif()
    set(LIB_INSTALL_DIR ".")
    set(INCLUDE_INSTALL_DIR "include")
    set(CMAKE_INSTALL_DIR "cmake/gzdrl")
endif()
set(GZDRL_PYTHON_INSTALL_DIR "gzdrl")
set(ENVS_PYTHON_INSTALL_DIR "${GZDRL_PYTHON_INSTALL_DIR}/envs")
set(SITL_PYTHON_INSTALL_DIR "${GZDRL_PYTHON_INSTALL_DIR}/sitl")
set(RESOURCE_INSTALL_DIR "${GZDRL_PYTHON_INSTALL_DIR}/resources")
set(SDF_INSTALL_DIR "${RESOURCE_INSTALL_DIR}/sdf")

set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type (Release or Debug)")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
message(STATUS "Building ${CMAKE_BUILD_TYPE} binary")

if(SKBUILD)
    set(_gzdrl_native_optimizations_default OFF)
else()
    set(_gzdrl_native_optimizations_default ON)
endif()
option(GZDRL_NATIVE_OPTIMIZATIONS
    "Enable host-specific instruction-set optimizations"
    ${_gzdrl_native_optimizations_default})

set(CMAKE_CXX_FLAGS_RELEASE
    "-O3 -DNDEBUG -fopenmp -DEIGEN_NO_DEBUG")
if(GZDRL_NATIVE_OPTIMIZATIONS)
    add_compile_definitions(GZDRL_NATIVE_OPTIMIZATIONS=1)
    string(APPEND CMAKE_CXX_FLAGS_RELEASE
        " -ffast-math -DEIGEN_FAST_MATH"
        " -march=native -mtune=native -flto=auto -fno-fat-lto-objects"
        " -funroll-loops -finline-functions -fomit-frame-pointer"
        " -fstrict-aliasing -ftree-vectorize -fprefetch-loop-arrays"
        " -fno-signed-zeros -fno-trapping-math -fassociative-math"
        " -freciprocal-math -mfpmath=sse -mavx2 -mfma")
endif()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)  # Enable LTO globally

# Cache variable
set(USE_XLA OFF CACHE BOOL "Use XLA Features, Requires CUDA")

# Use relative runtime paths so installations and wheels can be relocated.
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)

# Set compile definitions based on options
if(USE_XLA)
    add_compile_definitions(USE_XLA=1)
endif()


find_package(gz-cmake QUIET)
if (NOT gz-cmake_FOUND)
   # Fallback for older os 
   # pkg-config for older gazebo versions are not properly setup
   # Only gz-ionic is supported for fallback
   find_package(PkgConfig REQUIRED)
   pkg_check_modules(GZ_CMAKE QUIET gz-cmake4)
   if (GZ_CMAKE_FOUND)
     find_package(gz-cmake4 REQUIRED)
   else()
       message(FATAL_ERROR "gz-cmake4 package not found. For Ubuntu 22.04 and less, only gz-ionic is supported")
   endif()
   gz_find_package(gz-sim9 COMPONENTS
      gui
      rendering
      )
   gz_find_package(gz-math8)
   gz_find_package(gz-rendering9)
   gz_find_package(gz-common6)
   gz_find_package(gz-plugin3 COMPONENTS
      register
      )
   gz_find_package(gz-sensors9 COMPONENTS
      camera
      depth_camera
      gpu_lidar
      lidar
      rgbd_camera
      segmentation_camera
      thermal_camera
      wide_angle_camera
      )
   gz_find_package(sdformat15 
      )

    # Set version variables for fallback packages
    set(gz_sim_VERSION_MAJOR 9)
    set(gz_common_VERSION_MAJOR 6)
    set(gz_rendering_VERSION_MAJOR 9)
    set(gz_plugin_VERSION_MAJOR 3)
    set(gz_sensors_VERSION_MAJOR 9)
    set(sdformat_VERSION_MAJOR 15)

else()
   gz_find_package(gz-sim COMPONENTS
   gui
   rendering
   )
   gz_find_package(gz-math)
   gz_find_package(gz-rendering)
   gz_find_package(gz-common)
   gz_find_package(gz-plugin COMPONENTS
      register
      )
   gz_find_package(gz-sensors COMPONENTS
      camera
      depth_camera
      gpu_lidar
      lidar
      rgbd_camera
      segmentation_camera
      thermal_camera
      wide_angle_camera
      )
   gz_find_package(sdformat 
      )
   gz_find_package(gz-transport)
endif()

find_package(Python 3.10...<3.14 REQUIRED
    COMPONENTS 
    Interpreter 
    Development.Module)

find_package(pybind11 2.12.0...3.12.1 CONFIG QUIET)
# Use the pybind11 shipped with the active Python, and require NumPy-2 ABI
# support (>= 2.12). Without this, find_package silently falls back to the
# system pybind11 2.11.x, whose binaries segfault under NumPy 2.x.
if(NOT pybind11_DIR)
  execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
    list(APPEND CMAKE_PREFIX_PATH "${pybind11_DIR}")
endif()
find_package(pybind11 2.12.0...3.12.1 CONFIG REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(glog REQUIRED)
if (USE_XLA)
    find_package(CUDAToolkit REQUIRED)
endif()

# Setup global deps for sensor interfaces

set(SENSOR_INTERFACES 
    gz-sim${gz_sim_VERSION_MAJOR}::rendering 
    gz-common${gz_common_VERSION_MAJOR}::gz-common${gz_common_VERSION_MAJOR}
    gz-rendering${gz_rendering_VERSION_MAJOR}::gz-rendering${gz_rendering_VERSION_MAJOR}
    gz-sensors${gz_sensors_VERSION_MAJOR}::boundingbox_camera
    gz-sensors${gz_sensors_VERSION_MAJOR}::camera
    gz-sensors${gz_sensors_VERSION_MAJOR}::depth_camera
    gz-sensors${gz_sensors_VERSION_MAJOR}::gpu_lidar
    gz-sensors${gz_sensors_VERSION_MAJOR}::gz-sensors${gz_sensors_VERSION_MAJOR}
    gz-sensors${gz_sensors_VERSION_MAJOR}::lidar
    gz-sensors${gz_sensors_VERSION_MAJOR}::rgbd_camera
    gz-sensors${gz_sensors_VERSION_MAJOR}::segmentation_camera
    gz-sensors${gz_sensors_VERSION_MAJOR}::thermal_camera
    gz-sensors${gz_sensors_VERSION_MAJOR}::wide_angle_camera
    )

# Include directories common for all targets
include_directories(${EIGEN3_INCLUDE_DIRS})
if (USE_XLA)
    include_directories(${CUDAToolkit_INCLUDE_DIRS})
endif()
include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(GzDRL/include)
include_directories(sitl/include)
include_directories(envs)
include_directories(envs/envpool)

# Link directories common for all targets
if (USE_XLA)
    link_directories(${CUDAToolkit_LIBRARY_DIRS})
endif()

# Targets

## Uav controller
add_library(uav_controllers SHARED 
    GzDRL/src/controllers/geometric_controller.cc
    GzDRL/src/controllers/inner_loop_controller.cc
    GzDRL/src/controllers/smc_controller.cc
    GzDRL/src/controllers/pid_controller.cc
    )
target_link_libraries(uav_controllers PUBLIC
    Eigen3::Eigen
    )
target_include_directories(uav_controllers PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/GzDRL/include>
    $<INSTALL_INTERFACE:${INCLUDE_INSTALL_DIR}>
    )

## Sensor interface
add_library(sensor_interface SHARED
    GzDRL/src/sensor/sensor_interface.cc
    GzDRL/src/sensor/video_recording_manager.cc
    )
target_link_libraries(sensor_interface PUBLIC
    ${SENSOR_INTERFACES}
)
target_include_directories(sensor_interface PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/gzdrl/include>
    $<INSTALL_INTERFACE:${INCLUDE_INSTALL_DIR}>
    )

## Rl server
add_library(rl_server SHARED  
    GzDRL/src/rl_server.cc
    )
target_link_libraries(rl_server PUBLIC
    gz-sim${gz_sim_VERSION_MAJOR}::gz-sim${gz_sim_VERSION_MAJOR}
    gz-sim${gz_sim_VERSION_MAJOR}::rendering 
    gz-sim${gz_sim_VERSION_MAJOR}::gui
    uav_controllers
    sensor_interface
    # sdformat${sdformat_VERSION_MAJOR}::sdformat${sdformat_VERSION_MAJOR}
    gz-common${gz_common_VERSION_MAJOR}::gz-common${gz_common_VERSION_MAJOR}
    )
target_include_directories(rl_server PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/GzDRL/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/sitl/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/envs>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/envs/envpool>
    $<INSTALL_INTERFACE:${INCLUDE_INSTALL_DIR}>
    $<INSTALL_INTERFACE:${INCLUDE_INSTALL_DIR}/envpool>
    )

# Plugins
add_library(rotor_plugin SHARED 
    GzDRL/src/plugins/rotor_plugin.cc
    )
add_library(ackermann_plugin SHARED 
    GzDRL/src/plugins/ackermann_plugin.cc
    )
add_library(joint_position_controller_plugin SHARED 
    GzDRL/src/plugins/joint_position_controller_plugin.cc
    )

# Python bindings for rl server.  The extension is private so ``gzdrl`` can be
# a normal Python package that also exposes version and resource metadata.
pybind11_add_module(gzdrl_core
    GzDRL/python/rl_server.cc
    GzDRL/python/async_rl_server.cc
    )
set_target_properties(gzdrl_core PROPERTIES OUTPUT_NAME "_core")
target_link_libraries(gzdrl_core PRIVATE
    rl_server
    )

# Python bindings for Gazebo-Envpool envs
pybind11_add_module(gazebo_envpool_envs
    envs/gazebo_envpool/gazebo_envpool_envs.cc
)
target_include_directories(gazebo_envpool_envs PRIVATE  
    envs 
    envs/env_specs
    envs/envpool 
    envs/racing_env/include
    ${CMAKE_CURRENT_SOURCE_DIR}/sitl/include
    )
target_link_libraries(gazebo_envpool_envs PRIVATE  
    glog::glog
    rl_server
    uav_controllers)

if (USE_XLA)
   target_link_libraries(gazebo_envpool_envs PRIVATE  
       CUDA::cudart
       )
endif()

# SITL components, this will define SITL targets
add_subdirectory(sitl)

# Tests
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()

set(NATIVE_TARGETS  
    uav_controllers 
    sensor_interface
    rl_server
    )
set( PYTHON_TARGETS  
    gazebo_envpool_envs
    gzdrl_core
    )
set(PLUGIN_TARGETS 
    rotor_plugin
    ackermann_plugin
    joint_position_controller_plugin
    )
# Set target properties
## Plugins need special properties
foreach(targeti ${PLUGIN_TARGETS})
    target_link_libraries(${targeti} PRIVATE 
        rl_server
        gz-plugin${gz_plugin_VERSION_MAJOR}::gz-plugin${gz_plugin_VERSION_MAJOR}-register
        )
    target_include_directories(${targeti} PRIVATE
        GzDRL/include
        )
endforeach()

foreach( targeti ${NATIVE_TARGETS} ${PLUGIN_TARGETS} 
        ${SITL_NATIVE_TARGETS} ${SITL_PYTHON_TARGETS})
    set_target_properties(${targeti} PROPERTIES 
        POSITION_INDEPENDENT_CODE ON
        )
endforeach()
foreach( targeti ${PYTHON_TARGETS})
    set_target_properties(${targeti} PROPERTIES
    INTERPROCEDURAL_OPTIMIZATION ON
    VISIBILITY_INLINES_HIDDEN ON
    BUILD_WITH_INSTALL_RPATH FALSE
    )
endforeach()

foreach(targeti ${NATIVE_TARGETS} ${PLUGIN_TARGETS} ${SITL_NATIVE_TARGETS})
    set_target_properties(${targeti} PROPERTIES INSTALL_RPATH "$ORIGIN")
endforeach()

if(SKBUILD)
    set_target_properties(gzdrl_core PROPERTIES INSTALL_RPATH "$ORIGIN/lib")
    set_target_properties(gazebo_envpool_envs PROPERTIES
        INSTALL_RPATH "$ORIGIN/../lib")
    set_target_properties(sitl_core PROPERTIES
        INSTALL_RPATH "$ORIGIN/../lib")
else()
    set_target_properties(gzdrl_core gazebo_envpool_envs sitl_core PROPERTIES
        INSTALL_RPATH "$ORIGIN/..")
endif()

# Let sitl targets link with rl_server  
foreach ( targeti ${SITL_NATIVE_TARGETS} ${SITL_PYTHON_TARGETS})
    target_link_libraries(${targeti} PRIVATE 
        rl_server
        )
endforeach()

# ============================================================================
# Installation and Export
# ============================================================================

# Export native targets for downstream CMake projects
set(EXPORT_TARGETS 
    uav_controllers
    sensor_interface
    rl_server
)
if(BUILD_SITL_ROS AND TARGET rosrl_sitl)
    list(APPEND EXPORT_TARGETS rosrl_sitl)
endif()

# Install exportable targets with EXPORT
install(TARGETS ${EXPORT_TARGETS}
    EXPORT gzdrlTargets
    LIBRARY DESTINATION ${LIB_INSTALL_DIR}
    ARCHIVE DESTINATION ${LIB_INSTALL_DIR}
    RUNTIME DESTINATION ${LIB_INSTALL_DIR}
)

# Gazebo plugins are loaded dynamically and are intentionally not exported as
# CMake link targets.
install(TARGETS ${PLUGIN_TARGETS}
    LIBRARY DESTINATION ${LIB_INSTALL_DIR}
    RUNTIME DESTINATION ${LIB_INSTALL_DIR}
)

# Keep extension modules inside their Python package namespaces.
install(TARGETS gzdrl_core
    LIBRARY DESTINATION ${GZDRL_PYTHON_INSTALL_DIR}
    RUNTIME DESTINATION ${GZDRL_PYTHON_INSTALL_DIR}
)
install(TARGETS gazebo_envpool_envs
    LIBRARY DESTINATION ${ENVS_PYTHON_INSTALL_DIR}
    RUNTIME DESTINATION ${ENVS_PYTHON_INSTALL_DIR}
)
install(TARGETS sitl_core
    LIBRARY DESTINATION ${SITL_PYTHON_INSTALL_DIR}
    RUNTIME DESTINATION ${SITL_PYTHON_INSTALL_DIR}
)

# Python sources are installed by CMake as well.  This prevents the wheel
# builder from copying C++ implementation and vendored source directories as
# accidental Python package data.
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/GzDRL/__init__.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/GzDRL/_paths.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/GzDRL/_version.py"
    DESTINATION ${GZDRL_PYTHON_INSTALL_DIR}
)
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/envs/__init__.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/envs/env_adapters.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/envs/registration.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/envs/sb3_custom_callbacks.py"
    DESTINATION ${ENVS_PYTHON_INSTALL_DIR}
)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/envs/python/"
    DESTINATION "${ENVS_PYTHON_INSTALL_DIR}/python"
    FILES_MATCHING PATTERN "*.py"
)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/envs/python_envs/"
    DESTINATION "${ENVS_PYTHON_INSTALL_DIR}/python_envs"
    FILES_MATCHING PATTERN "*.py"
)
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/sitl/__init__.py"
    DESTINATION ${SITL_PYTHON_INSTALL_DIR}
)

# Runtime Gazebo worlds, models, meshes, textures, and plugin configuration.
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/resources/sdf"
    DESTINATION ${RESOURCE_INSTALL_DIR}
)

# Install headers for downstream CMake and standalone C++ consumers.
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/GzDRL/include/
    DESTINATION ${INCLUDE_INSTALL_DIR}
    FILES_MATCHING PATTERN "*.hh" PATTERN "*.hpp" PATTERN "*.h"
)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/sitl/include/
    DESTINATION ${INCLUDE_INSTALL_DIR}
    FILES_MATCHING PATTERN "*.hh" PATTERN "*.hpp" PATTERN "*.h"
)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/envs/
    DESTINATION ${INCLUDE_INSTALL_DIR}
    FILES_MATCHING
    PATTERN "*.hh" PATTERN "*.hpp" PATTERN "*.h"
    PATTERN "python" EXCLUDE
    PATTERN "python_envs" EXCLUDE
    PATTERN "__pycache__" EXCLUDE
)

# Export the targets to a file
install(EXPORT gzdrlTargets
    FILE gzdrlTargets.cmake
    NAMESPACE gzdrl::
    DESTINATION ${CMAKE_INSTALL_DIR}
)

# Generate and install config files
include(CMakePackageConfigHelpers)

# Set variables for config file
set(PACKAGE_INCLUDE_INSTALL_DIR "${INCLUDE_INSTALL_DIR}")
set(PACKAGE_LIB_INSTALL_DIR "${LIB_INSTALL_DIR}")
set(PACKAGE_RESOURCE_INSTALL_DIR "${RESOURCE_INSTALL_DIR}")
set(PACKAGE_SDF_INSTALL_DIR "${SDF_INSTALL_DIR}")

# Determine ROS version for config file
if(BUILD_SITL_ROS)
    # ROS_VER should be set by sitl/CMakeLists.txt
    if(NOT DEFINED ROS_VER)
        set(ROS_VER 0)
    endif()
else()
    set(ROS_VER 0)
endif()

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/gzdrlConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/gzdrlConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_DIR}
    PATH_VARS
        PACKAGE_INCLUDE_INSTALL_DIR
        PACKAGE_LIB_INSTALL_DIR
        PACKAGE_RESOURCE_INSTALL_DIR
        PACKAGE_SDF_INSTALL_DIR
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/gzdrlConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/gzdrlConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/gzdrlConfigVersion.cmake"
    "${CMAKE_CURRENT_SOURCE_DIR}/sitl/cmake/hybrid_ros.cmake"
    "${CMAKE_CURRENT_SOURCE_DIR}/sitl/cmake/determine_ros_version.cmake"
    DESTINATION ${CMAKE_INSTALL_DIR}
)

message(STATUS "CMake config files will be installed to: ${CMAKE_INSTALL_DIR}")
message(STATUS "Downstream projects can use: find_package(gzdrl REQUIRED)")
