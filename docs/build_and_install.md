# Build and Install

## Prerequisites

### Required for core build

- CMake `3.22.1+`
- C99 compiler toolchain
- FFTW build prerequisites handled by CMake
- Vulkan toolchain only when `ENABLE_VULKAN_BACKEND=ON`
- glslang when `ENABLE_VULKAN_BACKEND=ON`: VkFFT and the operator JIT both
  compile GLSL at runtime. The default `NLOLIB_GLSLANG_PROVIDER=FETCH` builds
  it through CMake, so development files are needed only for the `SYSTEM` and
  `AUTO` providers

### Optional but commonly needed

- Python 3 when `BUILD_TESTING=ON`
- Doxygen for the `docs` target
- Graphviz for call/include graphs

## Quick Start

### Windows

```powershell
cmake -S . -B build
cmake --build build --config Debug
ctest --test-dir build --build-config Debug --output-on-failure
```

### Linux

Ubuntu package prerequisites for the Vulkan/VkFFT path:

```bash
sudo apt install libvulkan-dev glslang-dev glslang-tools spirv-tools
```

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

## Build Configuration

Important top-level options:

- `NLOLIB_BUILD_DOCS`
- `NLOLIB_BUILD_BENCHMARKS`
- `NLOLIB_BUILD_EXAMPLES`
- `NLOLIB_BUILD_MATLAB_TESTS`
- `ENABLE_VULKAN_BACKEND`
- `ENABLE_VKFFT`
- `NLOLIB_GLSLANG_PROVIDER`
- `GLSLANG_GIT_TAG`
- `NLOLIB_FFTW_PROVIDER`
- `NLOLIB_SQLITE_PROVIDER`
- `NLOLIB_STATIC_LIBSTDCXX`
- `BUILD_TESTING`

`NLOLIB_GLSLANG_PROVIDER` controls how the runtime glslang dependency, shared
by VkFFT and the Vulkan operator JIT, is resolved:

- `AUTO` first uses an installed glslang CMake package (`glslang-config.cmake`,
  found through `CMAKE_PREFIX_PATH`), then a Vulkan/glslang development
  package, then fetches glslang if needed.
- `SYSTEM` requires installed glslang development files and fails with an
  actionable error if they are missing.
- `FETCH` always builds glslang through CMake `FetchContent`.

`GLSLANG_GIT_TAG` defaults to `12.3.1` and is used only by the fetch provider.

`NLOLIB_FFTW_PROVIDER` and `NLOLIB_SQLITE_PROVIDER` take `FETCH` (the default)
or `SYSTEM`. `FETCH` builds the release pinned in `cmake/NLODependencies.cmake`
into nlolib statically. `SYSTEM` links an installed `libfftw3` (found through
`cmake/FindFFTW3.cmake`; `FFTW3_ROOT` points it at a prefix) or `libsqlite3`
(CMake's `FindSQLite3`) instead, and the library then needs that shared
library at runtime. Package builds that supply both from their own dependency
tree, such as the `nlolib_jll` recipe, use `SYSTEM`.

`NLOLIB_STATIC_LIBSTDCXX` (default `ON`) embeds libstdc++ and libgcc in nlolib
on Linux with GCC and hides their symbols, so a host that preloads an older C++
runtime, such as MATLAB, can still load the library. Turn it off when the
package ships a matching C++ runtime alongside the library.

## Bindings {#bindings}

The Python, MATLAB and Julia bindings are git submodules of this repository
(`python/`, `matlab/`, `julia/`), each backed by a repository of its own:

```bash
git submodule update --init
```

A binding that is checked out is added to the build, with its targets
(`matlab_stage`, `julia_stage`) and, when `BUILD_TESTING=ON`, its tests
(`test_python_*`, `test_julia_*`, `test_matlab_bindings`). One that is not is
skipped. `NLOLIB_PYTHON_DIR`, `NLOLIB_MATLAB_DIR` and `NLOLIB_JULIA_DIR` name a
binding checkout elsewhere; a binding repository uses them to build itself
against the core commit its `CORE_REF` file pins.

## The Project Version

`CMakeLists.txt` holds the core library version, and `sbom.cdx.json` stamps a
copy of it. They must agree at every commit; `test_sbom_pins_match` fails if
they drift. The bindings are separate repositories with versions of their own
(see [Bindings](#bindings)).

Nothing advances the version by itself. A release bumps it in one step:

```bash
python tools/release/check_versions.py --bump minor
```

That rewrites both together. `--write` regenerates the SBOM without bumping,
and the bare command prints the version.

Two options relate to this, both off by default:

- `INSTALL_GIT_HOOKS=ON` installs a pre-commit hook that runs the SBOM check.
  It changes no file; it only refuses a commit whose SBOM is stale.
- `BUMP_PATCH_ON_BUILD=ON` rewrites the version in `CMakeLists.txt` after a
  successful build. It is a development convenience and must stay off for a
  release build, whose packaged version is read back out of `CMakeLists.txt`
  after the build.

## Third-Party Pins and the SBOM

Every third-party source that is compiled into nlolib is pinned in one file,
`cmake/NLODependencies.cmake`: release version, archive URL and SHA-256. The
`FetchContent` calls verify the hash, so a build either uses exactly that
archive or fails at configure time. To bump a dependency, edit the version and
hash there and regenerate the bill of materials:

```bash
python tools/write_sbom.py
```

To confirm every pinned archive is still reachable and matches its hash
without configuring the project (needs network, takes a few seconds):

```bash
cmake -P cmake/check_pins.cmake
```

`sbom.cdx.json` is the machine-readable dependency list in CycloneDX 1.6
format (ECMA-424). It records, for every component, the exact version,
licence, source archive and hash, and whether it is compiled in, a runtime
requirement on the target machine, or build/test tooling that is never
shipped. `ctest -R test_sbom_pins_match` fails when the SBOM is stale.

`conanfile.txt` lists only the host-side pieces a package manager can supply
(Vulkan headers and loader, `glslangValidator`); it is optional and unused when
a Vulkan SDK or the distro development packages are installed.
The project supports CMake `3.22.1+`; newer CMake versions may expose
`Vulkan::glslang` directly, but that target is not required.

## Doxygen Docs

Enable docs and configure:

```bash
cmake -S . -B build -DNLOLIB_BUILD_DOCS=ON
```

Build docs:

```bash
cmake --build build --target docs --config Release
```

Generated output:

- `build/docs/html/index.html`

Docs notes:

- If Doxygen is not found, the `docs` target is not added.
- HTML formulas render through MathJax; local LaTeX is not required.
- The rendered landing page comes from `docs/index.md`.

## Related Guides

- [Vulkan Backend Behaviour](gpu_backend.md) for what the Vulkan backend does
  at runtime and the environment switches that change it
- [Python Binding Guide](../python/docs/guide.md)
- [MATLAB Binding Guide](../matlab/docs/binding_guide.md)
- [Julia Binding Guide](../julia/docs/guide.md)
