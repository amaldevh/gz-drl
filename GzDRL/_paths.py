# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Amal Dev Haridevan

"""Locations of the data and native development files shipped with GzDRL."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from . import _core


_PACKAGE_DIR = Path(__file__).resolve().parent
_CORE_DIR = Path(_core.__file__).resolve().parent


def _first_directory(candidates: Iterable[Path], fallback: Path) -> Path:
    """Return the first existing candidate, resolving paths for stable output."""

    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    return fallback.resolve()


RESOURCE_PATH = _first_directory(
    (
        _CORE_DIR / "resources",
        _PACKAGE_DIR / "resources",
        _PACKAGE_DIR.parent / "resources",
    ),
    _PACKAGE_DIR / "resources",
)
SDF_PATH = RESOURCE_PATH / "sdf"

PLUGIN_PATH = _first_directory(
    (
        _CORE_DIR / "lib",
        _PACKAGE_DIR / "lib",
        _CORE_DIR.parent,
    ),
    _PACKAGE_DIR / "lib",
)

INCLUDE_PATH = _first_directory(
    (
        _CORE_DIR / "include",
        _PACKAGE_DIR / "include",
        _CORE_DIR.parent / "include",
    ),
    _PACKAGE_DIR / "include",
)

CMAKE_PATH = _first_directory(
    (
        _CORE_DIR / "cmake" / "gzdrl",
        _PACKAGE_DIR / "cmake" / "gzdrl",
        _CORE_DIR.parent / "cmake" / "gzdrl",
    ),
    _PACKAGE_DIR / "cmake" / "gzdrl",
)


def _safe_child(root: Path, parts: tuple[os.PathLike[str] | str, ...]) -> Path:
    path = root.joinpath(*(os.fspath(part) for part in parts)).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes the packaged GzDRL directory: {path}")
    return path


def get_resource_path(*parts: os.PathLike[str] | str) -> Path:
    """Return a path below GzDRL's packaged ``resources`` directory."""

    return _safe_child(RESOURCE_PATH, parts)


def get_sdf_path(*parts: os.PathLike[str] | str) -> Path:
    """Return a path below GzDRL's packaged Gazebo SDF directory."""

    return _safe_child(SDF_PATH, parts)


def get_plugin_path() -> Path:
    """Return the directory containing GzDRL's Gazebo system plugins."""

    return PLUGIN_PATH


def get_include_path() -> Path:
    """Return the directory containing the installed GzDRL C++ headers."""

    return INCLUDE_PATH


def get_cmake_path() -> Path:
    """Return the directory containing ``gzdrlConfig.cmake``."""

    return CMAKE_PATH


def _prepend_environment_path(variable: str, path: Path) -> None:
    value = str(path)
    entries = [entry for entry in os.environ.get(variable, "").split(os.pathsep) if entry]
    if value not in entries:
        os.environ[variable] = os.pathsep.join((value, *entries))


def configure_gazebo_paths() -> None:
    """Add packaged models and plugins to Gazebo's process-local search paths."""

    _prepend_environment_path("GZ_SIM_RESOURCE_PATH", SDF_PATH)
    _prepend_environment_path("GZ_SIM_SYSTEM_PLUGIN_PATH", PLUGIN_PATH)
