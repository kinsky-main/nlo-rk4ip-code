# Developing the MATLAB Toolbox

This repository holds the MATLAB wrapper, its tests, its examples and its
release pipeline. The native library it wraps is built from the core
repository, [nlo-rk4ip-code](https://github.com/kinsky-main/nlo-rk4ip-code).

## Layout

| Path | Contents |
| --- | --- |
| `+nlolib/`, `nlolib_setup.m` | the wrapper |
| `mex/` | the progress-callback MEX source |
| `tests/` | MATLAB unit tests, run by the `test_matlab_bindings` CTest case |
| `examples/` | example scripts and their `+backend` helpers |
| `package_mltbx.m`, `merge_mltbx.m`, `tools/` | toolbox packaging |
| `VERSION` | the toolbox version; `package_mltbx` stamps the bundle from it |
| `CORE_REF` | the core commit the toolbox and CI are built from |

The staged and installed toolbox has a different layout from this repository:
the examples are under `examples/matlab` and the user guide is
`docs/matlab_user_guide.md`. The `matlab_stage` target does that mapping.

## Two ways to work

**As a submodule of the core repository.** Clone the core repository with
`--recurse-submodules`; this repository is its `matlab/` directory.

```bash
cmake -S . -B build -DBUILD_TESTING=ON -DNLOLIB_BUILD_MATLAB_TESTS=ON
cmake --build build --config Release --target matlab_stage
ctest --test-dir build -C Release -R "^test_matlab_"
```

**Standalone.** Clone this repository alone and fetch the pinned core revision
into `./core`:

```bash
bash tools/fetch_core.sh
cmake -S core -B build -DBUILD_TESTING=ON -DNLOLIB_BUILD_MATLAB_TESTS=ON -DNLOLIB_MATLAB_DIR="$PWD"
cmake --build build --config Release --target matlab_stage
ctest --test-dir build -C Release -R "^test_matlab_"
```

`NLOLIB_MATLAB_DIR` points the core build at this checkout. Then, in MATLAB:

```matlab
addpath('build/matlab_toolbox');
nlolib_setup();
api = nlolib.NLolib();
```

## The core pin

`CORE_REF` holds one full commit SHA of the core repository. CI and the release
workflow check out exactly that commit, so a tag here always builds the same
native library. To move to a newer core, replace the SHA in a pull request.
Inside the core repository the pin is not used — the submodule builds against
the tree it is mounted in.

## Releasing

The toolbox version in `VERSION` is independent of the core version.

1. Set `VERSION` (and `CORE_REF`, if the release needs a newer core) in a pull
   request and merge it.
2. Tag the merge commit with the bare version and push the tag:

   ```bash
   git tag "$(python tools/check_version.py)"
   git push origin "$(python tools/check_version.py)"
   ```

`.github/workflows/release-mltbx.yml` then builds the toolbox on Linux and
Windows, merges the two bundles into one `nlolib.mltbx`, and publishes it as a
GitHub release. File Exchange is linked to this repository's releases and
picks the new version up from there. Dispatching the workflow by hand is a dry
run: it builds and checks everything and publishes nothing.

To build a bundle locally, use `tools/build_mltbx_linux.sh` or
`tools/build_mltbx_windows.ps1`; see [binding_guide.md](binding_guide.md) for
what `package_mltbx` and `merge_mltbx` do.
