# CMake package

The installed CMake package is version 1.0.0 and uses same-major compatibility.
It exports native link targets under the `gzdrl::` namespace, including the
server, controllers, and sensor interface.

```cmake
cmake_minimum_required(VERSION 3.22.1)
project(my_gzdrl_application LANGUAGES CXX)

find_package(gzdrl 1.0 REQUIRED)

add_executable(my_application main.cc)
target_compile_features(my_application PRIVATE cxx_std_20)
target_link_libraries(my_application PRIVATE gzdrl::rl_server)

# Absolute, relocatable install locations exported by gzdrlConfig.cmake:
message(STATUS "GzDRL resources: ${gzdrl_RESOURCE_DIR}")
message(STATUS "GzDRL SDF: ${gzdrl_SDF_DIR}")
message(STATUS "GzDRL plugins: ${gzdrl_PLUGIN_DIR}")
```

From Python, discover the configuration directory without hardcoding a wheel
layout:

```bash
GZDRL_CMAKE_DIR="$(python -c 'import gzdrl; print(gzdrl.get_cmake_path())')"
cmake -S . -B build -Dgzdrl_DIR="$GZDRL_CMAKE_DIR"
```

Gazebo plugins are installed for dynamic loading but intentionally are not
exported as downstream link targets.

Use the three exported directory variables to configure
`GZ_SIM_RESOURCE_PATH` and `GZ_SIM_SYSTEM_PLUGIN_PATH` in a C++ application's
launcher. Do not infer a source checkout from `__FILE__`. The scikit-build wheel
uses relocatable runtime paths and disables host-specific instruction selection
by default; a source-native build may use different optimization settings.
