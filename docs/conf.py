# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Amal Dev Haridevan
"""Sphinx configuration for the GzDRL documentation."""

from __future__ import annotations

import os
import sys
from pathlib import Path


DOCS_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = DOCS_DIR.parent
sys.path.insert(0, str(DOCS_DIR / "_ext"))

from gzdrl_docs import prepare  # noqa: E402


generated = prepare(DOCS_DIR, REPOSITORY_ROOT)

project = "GzDRL"
author = "Amal Dev Haridevan, Junjie Kang, and Jinjun Shan"
copyright = "2026, Amal Dev Haridevan"
version = generated["version"]
release = generated["version"]

extensions = [
    "breathe",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx.ext.mathjax",
]

source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
root_doc = "index"
exclude_patterns = ["_generated/*", "_build/**", "README.md"]

myst_enable_extensions = [
    "attrs_block",
    "colon_fence",
    "deflist",
    "fieldlist",
    "substitution",
    "tasklist",
    "dollarmath",
    "amsmath",
]
myst_heading_anchors = 3

breathe_projects = {"GzDRL": generated["doxygen_xml"]}
breathe_default_project = "GzDRL"
breathe_show_define_initializer = True

html_theme = "furo"
html_title = f"GzDRL {release}"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#235b87",
        "color-brand-content": "#235b87",
        "color-admonition-background": "#f5f8fb",
    },
    "dark_css_variables": {
        "color-brand-primary": "#77b7e5",
        "color-brand-content": "#77b7e5",
    },
    "announcement": (
        f"GzDRL {release} · source revision "
        f"{generated['source_ref'][:12]}"
    ),
}
html_show_sourcelink = False
html_context = generated

copybutton_prompt_text = r">>> |\.\.\. |\$ |# "
copybutton_prompt_is_regexp = True

nitpicky = True
nitpick_ignore = [
    ("cpp:identifier", "Eigen"),
    ("cpp:identifier", "std"),
    ("cpp:identifier", "Env<EnvSpec<SpecCls>>"),
    ("cpp:identifier", "EnvSpec<SpecCls>"),
    ("cpp:identifier", "size_t"),
    # Internal Doxygen links referenced by class summaries but not expanded in
    # the deliberately small C++ foundation page.
    ("std:ref", "classDRLHelperSystem"),
    ("std:ref", "classGazeboProcessor"),
    ("std:ref", "classDRLServer_1a986acf9c582ab63cf1882ee025a686f8"),
    ("std:ref", "classDRLServer_1a362111526e11af96a978a347a7764ef1"),
    ("std:ref", "classDRLServer_1aff0ebcfb24c2d534d17e173678ca0d3b"),
    ("std:ref", "gazebo__envpool__spec_8hh"),
    ("std:ref", "processor_8hh"),
]
suppress_warnings = [
    # Doxygen documents overloaded C++ names repeatedly; Breathe owns these IDs.
    "breathe.duplicate_declaration",
]

linkcheck_timeout = 20
linkcheck_retries = 2
linkcheck_anchors = True
linkcheck_ignore = [
    # The generated API creates hundreds of per-symbol GitHub links. Checking
    # them individually triggers GitHub's unauthenticated request rate limit.
    rf"https://github\.com/amaldevh/gz-drl/(?:blob|tree)/{generated['source_ref']}(?:/.*)?",
]
rst_epilog = f"""
.. |gzdrl-version| replace:: {release}
.. |source-ref| replace:: {generated['source_ref'][:12]}
"""

# Read the Docs supplies the canonical URL for each project/version.
html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "")
if os.environ.get("READTHEDOCS") == "True":
    html_context["READTHEDOCS"] = True
