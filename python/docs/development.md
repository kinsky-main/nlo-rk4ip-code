# Developing the Python Package

This repository holds the Python package, its tests, its examples and its
release pipeline. The native library it wraps is built from the core
repository, [nlo-rk4ip-code](https://github.com/kinsky-main/nlo-rk4ip-code).

## Layout

| Path | Contents |
| --- | --- |
| `nlolib/` | the package (ctypes wrapper; no compiled extension) |
| `tests/` | the `test_python_*` suite, registered with CTest by `tests/CMakeLists.txt` |
| `examples/` | runnable scripts and their shared `backend/` helpers |
| `tools/` | wheel build scripts and version check |
| `CORE_REF` | the core commit the wheels and CI are built from |
| `pyproject.toml` | package metadata; the source of truth for the version |

## Two ways to work

**As a submodule of the core repository.** Clone the core repository with
`--recurse-submodules`; this repository is its `python/` directory. A normal
core build then builds the library next to the package and registers the
Python tests:

```bash
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build --config Release
ctest --test-dir build -C Release -R "^test_python_"
```

**Standalone.** Clone this repository alone and fetch the pinned core revision
into `./core`:

```bash
bash tools/fetch_core.sh
cmake -S core -B build -DBUILD_TESTING=ON -DNLOLIB_PYTHON_DIR="$PWD"
cmake --build build --config Release
ctest --test-dir build -C Release -R "^test_python_"
```

`NLOLIB_PYTHON_DIR` points the core build at this checkout. Either way the
shared library lands beside the package, where `nlolib.load()` finds it; set
`NLOLIB_LIBRARY` to load a different one.

## The core pin

`CORE_REF` holds one full commit SHA of the core repository. CI and the release
workflow check out exactly that commit, so a tag here always builds the same
native library. To move to a newer core, replace the SHA in a pull request; CI
then runs the tests against it. Inside the core repository the pin is not used
— the submodule builds against the tree it is mounted in.

## Versioning and release

The package version in `pyproject.toml` is independent of the core version.
See [release.md](release.md).
