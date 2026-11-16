# Release Guide

Nothing is published from this repository. Each binding is released from its
own repository, with its own version, by pushing a bare version tag (no `v`
prefix) there:

| Artifact | Repository | Workflow | Guide |
| --- | --- | --- | --- |
| Python wheels on PyPI | [nlolib-python](https://github.com/kinsky-main/nlolib-python) | `release-pypi.yml` | [python/docs/release.md](../python/docs/release.md) |
| `nlolib.mltbx` on GitHub releases and File Exchange | [nlolib-matlab](https://github.com/kinsky-main/nlolib-matlab) | `release-mltbx.yml` | [matlab/docs/development.md](../matlab/docs/development.md) |
| Julia artifacts and registry entry | [NLOLib.jl](https://github.com/kinsky-main/NLOLib.jl) | `release-julia.yml` | [julia/docs/release.md](../julia/docs/release.md) |

## What a binding release is built from

Each binding repository has a `CORE_REF` file holding one full commit SHA of
this repository. Its release workflow checks out that commit, configures it
with `-DNLOLIB_<LANG>_DIR` pointing at the binding, and builds the native
library from it. So a binding tag always names one binding tree and one core
tree, and a binding-only fix can ship without touching this repository.

## Releasing a core change

1. Merge the change here. Bump the core version if the C API or behaviour
   changed:

   ```bash
   python tools/release/check_versions.py --bump minor
   ```

2. In each binding that should ship it: set `CORE_REF` to the merged commit,
   bump the binding's own version (`pyproject.toml`, `VERSION` or
   `Project.toml`), merge, and push the version tag. The binding's CI runs its
   tests against the new core before the tag exists.
3. Back here, move the submodule pointers to the released binding commits:

   ```bash
   git submodule update --remote python matlab julia
   git commit -am "Update the binding submodules"
   ```

The binding versions need not match each other or the core version. The core
version is recorded in `CMakeLists.txt` and `sbom.cdx.json`; `CORE_REF` is what
ties a published binding to a core tree.

Note: the release builds configure CMake with `SQLITE_USE_FETCHCONTENT=ON`,
which statically links SQLite into `nlolib` and avoids external SQLite runtime
DLL dependencies.
