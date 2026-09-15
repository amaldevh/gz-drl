# GzDRL documentation

This directory contains the Sphinx/MyST documentation for the adjacent GzDRL
source. Read the Docs uses the root `.readthedocs.yaml` configuration.

From the repository root:

```bash
sudo apt-get install doxygen
python3 -m venv /tmp/gzdrl-docs-venv
source /tmp/gzdrl-docs-venv/bin/activate
python -m pip install -r docs/requirements.txt
python -m sphinx -b html -n -W --keep-going docs docs/_build/html
python -m sphinx -b linkcheck docs docs/_build/linkcheck
```

Open `docs/_build/html/index.html` to review the site. No Gazebo or ROS
installation is needed for the documentation build. Build from a Git checkout;
API links refer to its commit and version metadata is read from its source.

The contributor guide at `contributing/documentation.md` covers editing,
catalog metadata, generated files, and Read the Docs project setup. The
simulation flow diagram is in `concepts/execution-semantics.md`, with the
original architecture figure in `modules/index.md`.
