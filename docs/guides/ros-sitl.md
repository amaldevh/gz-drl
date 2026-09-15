# Connect a simulation to ROS

<span id="ros-2-and-sitl"></span>

The optional `gzdrl.sitl.RosDRLServer` bridge lets ROS nodes exchange commands
and state with a Gazebo simulation. Use it for software-in-the-loop (SITL)
testing of a controller or deployment stack. It owns a `DRLServer` and runs the
simulation at a requested real-time factor.

## Build the bridge

The `gzdrl.sitl` package always includes `SecondOrderLPFilter3d`. The ROS class
is available only when ROS was detected during compilation. For ROS 2 Jazzy,
source the installation and rebuild from the repository root:

```bash
source /opt/ros/jazzy/setup.bash
python -m pip install . -Ccmake.define.GZDRL_ROS_MODE=ON
python -c 'from gzdrl.sitl import RosDRLServer; print(RosDRLServer)'
```

`GZDRL_ROS_MODE` accepts `AUTO`, `ON`, or `OFF`. `AUTO` detects the sourced ROS
environment, `ON` requires detection to succeed, and `OFF` omits the bridge.
Detection is refreshed on each CMake configure. The source includes ROS 1 and
ROS 2 paths; the current Dockerfile uses Ubuntu 24.04, Jazzy, and Gazebo Jetty.

The bridge integrates directly with ROS and does not use `ros_gz`. For projects
that also use `ros_gz`, follow its [Gazebo–ROS compatibility guidance](https://gazebosim.org/docs/jetty/ros_installation/)
when choosing versions.

## Run the simulation and ROS executor

Construct the bridge with a transport partition, world, tracked models, sensor
flag, real-time factor, and optional model-to-link map:

```python
import gzdrl
from gzdrl.sitl import RosDRLServer

bridge = RosDRLServer(
    "ros-demo",
    str(gzdrl.get_sdf_path("world_simple.sdf")),
    ["quadrotor"],
    False,
    1.0,
    {"quadrotor": ["quadrotor/base_link"]},
)
bridge.run()
print(bridge.get_published_topic_map())
print(bridge.get_subscribed_topic_map())
bridge.spin()
```

A factor of `1.0` requests simulation time to follow wall-clock time when the
hardware can keep up. `spin()` runs the ROS executor in the calling thread;
`spin_async()` is available when the application needs a background executor.
Inspect the returned topic maps when connecting other nodes.

`server()` exposes the underlying direct server. Keep one owner for simulation
stepping: advancing it independently while the bridge loop is running can
cause data races. See {doc}`/examples/ros-sitl/index` for the repository example.
