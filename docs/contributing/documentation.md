# Build and edit the documentation

<span id="documentation-standards"></span>
<span id="source-of-truth"></span>
<span id="claims"></span>
<span id="local-checks"></span>

The site uses Sphinx, MyST Markdown, and the Furo theme. Doxygen and Breathe
render the C++ reference; a local extension extracts the Python binding
inventory and creates the environment, example, and module catalogs.
The documentation build reads source files without compiling or importing the
native package, so Gazebo and ROS are not required to build the site.

## Build locally

From a Git checkout of `gz-drl`, install Doxygen and the documentation tools:

```bash
sudo apt-get install doxygen
python3 -m venv /tmp/gzdrl-docs-venv
source /tmp/gzdrl-docs-venv/bin/activate
python -m pip install -r docs/requirements.txt
python -m sphinx -b html -n -W --keep-going docs docs/_build/html
```

Open `docs/_build/html/index.html` in a browser, or serve the output locally:

```bash
python -m http.server --directory docs/_build/html 8000
```

The strict build fails on warnings, including broken internal references. To
check external links separately:

```bash
python -m sphinx -b linkcheck docs docs/_build/linkcheck
```

Generated API source is written to `docs/_generated/`; Doxygen XML and HTML
are written under `docs/_build/`. Both directories are ignored by Git.
The source revision shown on the site comes from the checkout's commit, and
the version comes from `pyproject.toml`, checked against CMake and `_version.py`.

## Read the Docs setup

The root `.readthedocs.yaml` selects Ubuntu 24.04, Python 3.12, Doxygen, and
`docs/requirements.txt`. It points Sphinx at `docs/conf.py` and fails on build
warnings. See the [Read the Docs configuration reference](https://docs.readthedocs.com/platform/stable/config-file/v2.html)
for the supported settings.

After the files are committed and pushed, import the `gz-drl` repository into
Read the Docs and select the branch containing this configuration. Trigger a
build and inspect the generated pages and logs. Enable release tags in the
project's Versions settings when versioned documentation is needed.

Each branch or tag supplies both the docs and the source used for its API
reference. There is no separate source download or moving `main` dependency.
The hosted build requires no Gazebo installation and no simulation display.

## Write pages that help readers complete a task

Begin with the problem and explain where the topic fits in the
{doc}`/concepts/execution-semantics` flow. Introduce prerequisites before the
first command, then explain the expected result and the next useful step.
Use stable terms from aerial robotics and reinforcement learning, and keep
method names, units, frames, and shapes precise.

Verify behavior against the current implementation and bindings. Use the paper
for experimental results, with the relevant hardware and configuration.
Prefer source excerpts through `literalinclude` when explaining a working
implementation; avoid incomplete code presented as a runnable example.

Preserve existing page paths and anchors when revising content. Catalog pages
use YAML metadata described in the contributor templates. Review the rendered
page as well as the source before publishing.

## Licensing

The migrated guides and architecture assets retain the MIT license from the
original documentation repository, included in `docs/LICENSE`. GzDRL source
and extracted API text retain their original notices. Consult the repository's
`THIRD_PARTY_NOTICES.md` and `LICENSES/` for vendored code.
