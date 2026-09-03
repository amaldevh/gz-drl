# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Amal Dev Haridevan

"""Registered Gazebo-EnvPool environments."""

from __future__ import annotations

from types import ModuleType

import numpy as np

import gzdrl

from .python.api import py_env
from .registration import (
    list_all_envs,
    make,
    make_dm,
    make_gym,
    make_gymnasium,
    make_spec,
    register,
)


if int(np.__version__.split(".", maxsplit=1)[0]) not in (1, 2):
    raise ImportError(
        f"gzdrl requires NumPy 1.x or 2.x, but found {np.__version__}"
    )

from . import gazebo_envpool_envs as _gazebo_envpool_envs


_registered_envs: list[str] = []


def env_loader(module: ModuleType = _gazebo_envpool_envs) -> None:
    """Register every Gazebo environment exported by a native binding module."""

    attrs = dir(module)
    pool_names = sorted(name for name in attrs if name.startswith("_GazeboPool"))
    spec_names = sorted(name for name in attrs if name.startswith("_GazeboSpec"))
    if len(pool_names) != len(spec_names):
        raise RuntimeError(f"Pool and spec names do not match in module {module.__name__}")

    for pool_name, spec_name in zip(pool_names, spec_names):
        env_name = pool_name.removeprefix("_") + "-v0"
        base_name = env_name
        index = 1
        while env_name in _registered_envs:
            env_name = f"{base_name[:-1]}{index}"
            index += 1
        _registered_envs.append(env_name)

        env_spec, dm_pool, gym_pool, gymnasium_cls = py_env(
            getattr(module, spec_name), getattr(module, pool_name)
        )
        config_name = env_name.replace("-", "_").replace(" ", "")
        generated = {
            f"{config_name}Spec": env_spec,
            f"{config_name}DMEnvPool": dm_pool,
            f"{config_name}GymEnvPool": gym_pool,
            f"{config_name}GymnasiumEnvPool": gymnasium_cls,
        }
        globals().update(generated)
        register(
            task_id=env_name,
            import_path=__name__,
            spec_cls=f"{config_name}Spec",
            dm_cls=f"{config_name}DMEnvPool",
            gym_cls=f"{config_name}GymEnvPool",
            gymnasium_cls=f"{config_name}GymnasiumEnvPool",
            resources_path=str(gzdrl.SDF_PATH),
            plugins_path=str(gzdrl.PLUGIN_PATH),
        )


env_loader()
