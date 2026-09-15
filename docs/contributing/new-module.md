# Add a module page

Create `docs/modules/<slug>/index.md`. No catalog or navigation file needs a
manual edit.

```markdown
---
title: My module
summary: One sentence explaining its public responsibility.
order: 80
tags: [Python, C++]
---

# My module

Describe its boundary, public entry points, ownership, and links to relevant
guides/reference pages.
```

The build extension scans `modules/*/index.md`, validates `title` and `summary`,
sorts by `order`, and creates cards plus a hidden toctree. Set `catalog: false`
only for a deliberately hidden compatibility page.

If the module adds pybind11 classes or functions in `GzDRL/python` or
`sitl/python`, the generated Python inventory discovers them from the configured
source. If it adds public C++ headers below a documented Doxygen input, add a
focused `doxygenfile` directive only when the header should appear in the
curated C++ foundation.
