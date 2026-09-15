# Check a documentation release

<span id="documentation-release-checklist"></span>

Before publishing a branch or tag on Read the Docs, check that the guides and
reference describe that version of GzDRL.

1. Confirm that `pyproject.toml`, CMake, and `GzDRL/_version.py` agree on the
   version. The documentation build checks this automatically.
2. Verify registered IDs, default spaces, and any configuration-dependent
   shapes. Run a reset and a short rollout for changed task examples.
3. Check installation commands against the target platform and source build.
   Distinguish declared compatibility from configurations tested for the release.
4. Run the strict HTML build and external link check in
   {doc}`documentation`. Inspect any unavailable external links individually.
5. Review navigation, code blocks, tables, figures, and search in the rendered
   site. Check that source links point to the intended commit.
6. Check example output paths, optional dependencies, and known limitations.
   Verify citation metadata against the public paper.
7. Review license notices for any copied text, code, or assets.
8. Push the reviewed revision, build that version on Read the Docs, and inspect
   its build log and hosted pages before making it the default version.

If a hosted build fails, retain the previous working version while correcting
the failing revision. Generated build directories do not belong in the commit.
