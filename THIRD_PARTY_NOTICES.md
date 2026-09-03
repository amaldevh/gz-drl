# Third-party notices

GzDRL is distributed under the MIT License. Some source files and assets are
derived from or redistribute third-party projects under their original terms.
Those files are not relicensed by the project-wide MIT License. File-level
headers take precedence, and the corresponding full texts are in `LICENSES/`.

## EnvPool

The EnvPool compatibility layer in `envs/envpool/core/`, `envs/python/`,
`envs/registration.py`, and `envs/env_adapters.py` is derived from EnvPool,
copyright Garena Online Private Limited, and remains under Apache License 2.0.

License: `LICENSES/Apache-2.0.txt`

## Gazebo and RotorS-derived integrations

The Ackermann, joint-position, rotor, and sensor integration files listed
below retain their original Apache License 2.0 notices and contributor
copyrights:

- `gzdrl/include/plugins/ackermann_plugin.hh`
- `gzdrl/include/plugins/joint_position_controller_plugin.hh`
- `gzdrl/include/plugins/rotor_plugin.hh`
- `gzdrl/src/plugins/ackermann_plugin.cc`
- `gzdrl/src/plugins/joint_position_controller_plugin.cc`
- `gzdrl/src/plugins/rotor_plugin.cc`
- `gzdrl/include/sensor/sensor_interface.hh`
- `gzdrl/src/sensor/sensor_interface.cc`

License: `LICENSES/Apache-2.0.txt`

## ConcurrentQueue and semaphore

`envs/envpool/concurrentqueue.h` and
`envs/envpool/blockingconcurrentqueue.h` are from Cameron Desrochers'
moodycamel ConcurrentQueue and are used under the simplified BSD license.
`envs/envpool/lightweightsemaphore.h` contains Jeff Preshing's semaphore code,
adapted by Cameron Desrochers, under the zlib license.

Licenses: `LICENSES/BSD-2-Clause-ConcurrentQueue.txt` and
`LICENSES/Zlib-Preshing-Semaphore.txt`

## ThreadPool

`envs/envpool/ThreadPool.h` is from Jakob Progsch and Vaclav Zeman's ThreadPool
and remains under the zlib license. Its previously omitted upstream notice has
been restored in the source file.

License: `LICENSES/Zlib-ThreadPool.txt`

## robin-map

`tests/robin-map/` is a git submodule containing Tessil's robin-map, copyright
Thibaut Goetghebuer-Planchon, under the MIT License.

License: `LICENSES/MIT-robin-map.txt`

## Crazyflie model assets

`resources/sdf/crazyflie.sdf` and its Crazyflie mesh assets carry the Bitcraze
MIT notice, copyright 2022 Bitcraze.

License: `LICENSES/MIT-Bitcraze.txt`

## Stable-Baselines3 example-derived code

Parts of `experiments_paper/sb3_custom_callbacks.py` and the policy-extension
pattern in `experiments_paper/benchmark_dr/asymmetric_ppo.py` are adapted from
Stable-Baselines3 examples, copyright Antonin Raffin, under the MIT License.

License: `LICENSES/MIT-Stable-Baselines3.txt`

## Other models and research inputs

The remaining files under `resources/sdf/` include model and world inputs with
mixed or presently undocumented provenance. They are distributed as research
inputs and are not claimed as original MIT-licensed GzDRL source unless a
file-level notice says so. Confirm their redistribution terms before a public
release.
