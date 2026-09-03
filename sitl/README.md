# GzDRL software-in-the-loop interface

The `sitl` package shipped with GzDRL 1.0.0 provides reusable
software-in-the-loop components. Import it directly after installing GzDRL:

```python
import numpy as np
import gzdrl.sitl as sitl

filter_3d = sitl.SecondOrderLPFilter3d(20.0, 0.7, 0.001)
filtered = filter_3d.update(np.array([1.0, 0.0, -1.0]))
```

No manual import-path modification is required.

## ROS support

ROS support is detected at build time from the sourced ROS environment. When
`ROS_DISTRO` is unset, GzDRL builds the ROS-independent `sitl` module and
exports `SecondOrderLPFilter3d`. When a supported ROS setup is sourced, CMake
automatically enables the matching ROS 1 or ROS 2 integration and the Python
module also exports `RosDRLServer`.

The validated reference platform is **Ubuntu 24.04, Gazebo Jetty, and ROS 2
Jazzy**. Build and install with Jazzy support as follows:

```bash
source /opt/ros/jazzy/setup.bash
python -m pip install --no-cache-dir .
python -c "import gzdrl.sitl as sitl; print(sitl.RosDRLServer)"
```

The install command must run after sourcing ROS because detection happens while
the native extension is compiled. A wheel built without ROS remains useful for
the standalone filter, but it cannot acquire ROS functionality at import time;
rebuild it in the sourced environment.

`GZDRL_ROS_MODE` defaults to `AUTO`. Set the CMake definition to `OFF` to
deliberately omit ROS even when an environment is sourced, or to `ON` to fail
unless ROS is detected. Detection results are refreshed on every configure, so
reusing scikit-build's build directory does not preserve an earlier no-ROS or
different-ROS result.

`RosDRLServer` wraps the GzDRL simulation server with ROS publishers,
subscribers, synchronous simulation control, and configurable real-time-factor
execution. Use `import gzdrl.sitl as sitl` for its Python interface and `import gzdrl` for
world/plugin path helpers such as `gzdrl.get_sdf_path(...)`.
