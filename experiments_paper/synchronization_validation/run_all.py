#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Amal Dev Haridevan

"""Run both synchronization backends and generate the paper artifacts."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from common import CONTROL_RATE_HZ, DEFAULT_RESULTS_DIR, DEFAULT_SDF


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=("gzdrl", "ros2_gazebo"),
        default=("gzdrl", "ros2_gazebo"),
    )
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--actions", type=int, default=200)
    parser.add_argument("--control-rate", type=float, default=CONTROL_RATE_HZ)
    parser.add_argument("--sdf-file", type=Path, default=DEFAULT_SDF)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("pdf", "svg", "png"),
        default=("pdf", "png"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    script_directory = Path(__file__).resolve().parent
    results_dir = args.results_dir.expanduser().resolve()
    common = [
        "--trials",
        str(args.trials),
        "--actions",
        str(args.actions),
        "--control-rate",
        str(args.control_rate),
        "--sdf-file",
        str(args.sdf_file.expanduser().resolve()),
    ]
    if "gzdrl" in args.backends:
        subprocess.run(
            [
                sys.executable,
                str(script_directory / "run_gzdrl.py"),
                *common,
                "--output-dir",
                str(results_dir / "raw" / "gzdrl"),
            ],
            check=True,
        )
    if "ros2_gazebo" in args.backends:
        subprocess.run(
            [
                sys.executable,
                str(script_directory / "run_ros2.py"),
                *common,
                "--output-dir",
                str(results_dir / "raw" / "ros2_gazebo"),
            ],
            check=True,
        )
    subprocess.run(
        [
            sys.executable,
            str(script_directory / "analyze_results.py"),
            "--results-dir",
            str(results_dir),
            "--formats",
            *args.formats,
        ],
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
