# Source layout

The `gz-drl` checkout is organized by runtime responsibility:

| Path | Role | Included in generated API? |
|---|---|---|
| `GzDRL/include`, `GzDRL/src` | server, controllers, sensors, plugins | public headers only |
| `GzDRL/python` | pybind11 bindings for `gzdrl` | Python inventory |
| `envs/env_specs` | task configuration and tensor specs | C++ reference |
| `envs/processors` | task transforms and rewards | C++ reference |
| `envs/gazebo_envpool` | task stepping implementations/binding | curated narrative |
| `envs/envpool` | vendored EnvPool-derived scheduler | excluded |
| `envs/python` | vendored EnvPool-derived adapters | excluded from authored API inventory |
| `sitl` | filter and optional ROS bridge | Python inventory / selected headers |
| `resources/sdf` | worlds, models, meshes, textures | resource guides |
| `examples` | user-facing runnable integrations | curated example catalog |
| `tests` | validation and performance fixtures | excluded |
| `experiments_paper` | research orchestration, results, plotting | excluded except reviewed claims and one architecture asset |
| `build`, `_lib` | generated native artifacts | excluded |

The site header and Python API inventory identify the checkout revision used
to build the reference. Read the Docs builds the documentation and source from
the same branch or tag.
