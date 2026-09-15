# Add an example page

Create `docs/examples/<slug>/index.md` with a narrow, maintained workflow:

```markdown
---
title: My example
summary: The user outcome demonstrated by this example.
order: 80
tags: [beginner, GazeboPool]
---

# My example

Source example: `examples/my_example/run.py`.

List prerequisites, the command to run, expected output category, and the
important API contract it demonstrates.
```

Use installed imports (`import gzdrl`, `from gzdrl import envs`, or `from gzdrl import sitl`). Never
teach users to append a checkout, `gzdrl`, or `_lib` directory to `sys.path`.
Resolve packaged files through GzDRL path helpers.

Do not add research plot/table generators to the user example catalog. A paper
harness may be described on a research page when needed for reproducibility,
but it is not a library API.
