# SDF resources and Gazebo plugins

The wheel packages the project's SDF worlds, models, meshes, textures, and
plugin libraries. Importing `gzdrl` prepends the packaged locations to
`GZ_SIM_RESOURCE_PATH` and `GZ_SIM_SYSTEM_PLUGIN_PATH` for the current process.

Use the public helpers when an API needs an absolute path:

```python
import gzdrl

world = gzdrl.get_sdf_path("world_hover.sdf")
plugin_dir = gzdrl.get_plugin_path()
headers = gzdrl.get_include_path()
cmake_config = gzdrl.get_cmake_path()
```

`get_resource_path(*parts)` and `get_sdf_path(*parts)` reject paths that escape
their packaged roots. This catches accidental `..` traversal and avoids
depending on the source layout.

## Bundled systems

Native targets include the multirotor rotor plugin, Ackermann command plugin,
and joint-position controller plugin. The server library and sensor interface
are installed alongside them. The SDF's plugin filename and entity/link names
must match those libraries and the command calls.

## Downstream CMake

The wheel and native install include `gzdrlConfig.cmake`. A downstream C++
project can point `gzdrl_DIR` at `gzdrl.get_cmake_path()` or add its parent to
`CMAKE_PREFIX_PATH`, then use:

```cmake
find_package(gzdrl 1.0 REQUIRED)
target_link_libraries(my_target PRIVATE gzdrl::rl_server)
```

The package configuration exports `gzdrl_RESOURCE_DIR`, `gzdrl_SDF_DIR`,
and `gzdrl_PLUGIN_DIR`. Use those values when configuring a C++ launcher or
install-time environment for `GZ_SIM_RESOURCE_PATH` and
`GZ_SIM_SYSTEM_PLUGIN_PATH`.
