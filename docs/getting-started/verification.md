# Verify an installation

Check imports and packaged resources before constructing a simulator. Run the
following command outside the source directory, with the Python environment
from the installation guide still active. This prevents local source files
from hiding a packaging problem.

```bash
cd /tmp
python - <<'PYTHON'
import gzdrl
from gzdrl import envs, sitl

assert gzdrl.get_sdf_path("world_simple.sdf").is_file()
assert gzdrl.get_plugin_path().is_dir()
assert "GazeboPoolHoverEnv-v0" in envs.list_all_envs()
print("GzDRL", gzdrl.__version__)
print("SDF root", gzdrl.get_sdf_path())
print("Registered tasks", len(envs.list_all_envs()))
PYTHON
```

The current distribution registers seven tasks. A successful check confirms
that Python can load the native package and locate its resources. The
{doc}`quickstart` checks the next stage: loading a world and advancing physics.

## Diagnose a failure

| Symptom | Check |
|---|---|
| Import fails before a server is constructed | Confirm the active Python environment and inspect the missing shared library named in the error. |
| Gazebo cannot find the world | Print `gzdrl.get_sdf_path()` and check that the requested file exists. |
| Gazebo cannot load a plugin | Check `gzdrl.get_plugin_path()` and the plugin's native dependencies. |
| `RosDRLServer` is missing | Rebuild with ROS sourced and `GZDRL_ROS_MODE=ON`. |

To inspect all packaged directories:

```python
import gzdrl

print(gzdrl.get_resource_path())
print(gzdrl.get_plugin_path())
print(gzdrl.get_include_path())
print(gzdrl.get_cmake_path())
```
