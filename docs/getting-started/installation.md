# Installation

<span id="validated-project-baseline"></span>
<span id="system-dependencies"></span>
<span id="install-the-python-distribution"></span>
<span id="optional-ros-2sitl-support"></span>
<span id="native-cmake-installation"></span>

GzDRL includes C++ libraries, Python bindings, and Gazebo resources. Installing
from the source repository builds these components and packages them under
`gzdrl`. This guide uses Ubuntu 24.04 and Gazebo Jetty, matching the current
repository Dockerfile.

## Requirements

| Component | Requirement for this guide |
|---|---|
| Operating system | Ubuntu 24.04 |
| Simulator | Gazebo Jetty |
| Compiler | GCC/G++ 11; selected by CMake by default |
| Python | 3.10–3.13, as declared in `pyproject.toml` |
| Build tools | CMake 3.22.1 or newer, Python development headers |
| Optional ROS integration | ROS 2 Jazzy, sourced before building |

The repository also lists Ubuntu 22.04 and Gazebo Harmonic and Ionic among its
supported configurations. Package availability and dependency discovery vary
with that combination; the commands below describe the Ubuntu 24.04 / Jetty
installation. The Python package currently excludes Python 3.14.

## Install Gazebo and build dependencies

Follow the [Gazebo Jetty Ubuntu installation guide](https://gazebosim.org/docs/jetty/install_ubuntu/)
to configure OSRF's package repository. On Ubuntu 24.04:

```bash
sudo apt-get update
sudo apt-get install -y curl lsb-release gnupg
sudo curl https://packages.osrfoundation.org/gazebo.gpg \
  --output /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] https://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" \
  | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null
sudo apt-get update
sudo apt-get install -y gz-jetty
```

Install the compiler and the remaining build dependencies:

```bash
sudo apt-get install -y \
  build-essential cmake gcc-11 g++-11 git pkg-config \
  python3-dev python3-venv libeigen3-dev libgoogle-glog-dev
```

The Gazebo installation supplies its simulation, rendering, sensor, transport,
and SDF libraries. CUDA is only needed for the optional `USE_XLA` build.

## Build the Python package

Clone the repository and create an isolated Python environment:

```bash
git clone https://github.com/amaldevh/gz-drl.git
cd gz-drl
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

`pip` invokes CMake through `scikit-build-core`. The installation contains:

- `gzdrl`, the direct server and controller bindings;
- `gzdrl.envs`, the environment registry and Python adapters;
- `gzdrl.sitl`, the filter and optional ROS bridge;
- native libraries, plugins, SDF resources, public headers, and CMake exports.

Importing `gzdrl` adds the packaged resource and plugin directories to Gazebo's
search paths in the current process. Use `gzdrl.get_sdf_path()` and the other
{doc}`/guides/sdf-and-plugins` helpers when an application needs an absolute path.

For Stable-Baselines3 training, install the RL dependencies from the repository
root:

```bash
python -m pip install '.[rl]'
```

Continue with {doc}`verification` before starting a training run.

## Build with ROS support

For the optional bridge, install ROS 2 Jazzy and source it before building.
From the repository root, with the Python environment active:

```bash
source /opt/ros/jazzy/setup.bash
python -m pip install . -Ccmake.define.GZDRL_ROS_MODE=ON
python -c 'from gzdrl.sitl import RosDRLServer; print(RosDRLServer)'
```

`GZDRL_ROS_MODE=ON` makes a missing ROS installation a build error. The default,
`AUTO`, builds the bridge only when ROS is detected; `OFF` omits it.
GzDRL integrates directly with ROS and does not use `ros_gz`. See
{doc}`/guides/ros-sitl` for construction and runtime behavior.

## Install for C++ development

A native CMake installation provides headers and exported library targets:

```bash
cmake -S . -B build/native -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
cmake --build build/native --parallel
cmake --install build/native --prefix "$PWD/install"
```

Downstream projects use `find_package(gzdrl REQUIRED)` and link
`gzdrl::rl_server`. See {doc}`/reference/cmake` for resource paths and package
discovery from a Python installation.
