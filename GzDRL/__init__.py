# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Amal Dev Haridevan

"""Python interface for GzDRL."""

from __future__ import annotations

from ._version import __version__

from . import _core as _core
from ._paths import (
    CMAKE_PATH,
    INCLUDE_PATH,
    PLUGIN_PATH,
    RESOURCE_PATH,
    SDF_PATH,
    configure_gazebo_paths,
    get_cmake_path,
    get_include_path,
    get_plugin_path,
    get_resource_path,
    get_sdf_path,
)


configure_gazebo_paths()

_native_names = tuple(name for name in dir(_core) if not name.startswith("_"))
globals().update({name: getattr(_core, name) for name in _native_names})

__all__ = [
    "CMAKE_PATH",
    "INCLUDE_PATH",
    "PLUGIN_PATH",
    "RESOURCE_PATH",
    "SDF_PATH",
    "__version__",
    "configure_gazebo_paths",
    "get_cmake_path",
    "get_include_path",
    "get_plugin_path",
    "get_resource_path",
    "get_sdf_path",
    *_native_names,
]

del _native_names
