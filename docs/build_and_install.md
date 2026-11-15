# Build and Install

## Prerequisites

### Required for core build

- CMake `3.22.1+`
- C99 compiler toolchain
- FFTW build prerequisites handled by CMake
- Vulkan toolchain only when `ENABLE_VULKAN_BACKEND=ON`
- glslang development libraries when `ENABLE_VKFFT=ON`; CMake can fetch
  glslang when system development files are unavailable

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
- `BUILD_TESTING`

`NLOLIB_GLSLANG_PROVIDER` controls how VkFFT's runtime glslang dependency is
resolved:

- `AUTO` first uses an installed Vulkan/glslang development package, then
  fetches glslang if needed.
- `SYSTEM` requires installed glslang development files and fails with an
  actionable error if they are missing.
- `FETCH` always builds glslang through CMake `FetchContent`.

`GLSLANG_GIT_TAG` defaults to `12.3.1` and is used only by the fetch provider.

## The Project Version

`CMakeLists.txt` holds the version; `pyproject.toml` and `julia/Project.toml`
carry copies, because a wheel and a Julia package each declare a literal
version of their own, and `sbom.cdx.json` stamps it too. They must agree at
every commit, so a release tag can be cut from any commit and the published
artifacts carry the version their tag names. `test_version_copies_match` and
`test_sbom_pins_match` fail if they drift.

Nothing advances the version by itself. A release bumps it in one step:

```bash
python tools/release/check_versions.py --bump minor
```

That rewrites all three together. `--write` syncs the copies without bumping,
and the bare command verifies them.

Two options relate to this, both off by default:

- `INSTALL_GIT_HOOKS=ON` installs a pre-commit hook that runs the same check.
  It changes no file; it only refuses a commit whose versions disagree.
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

- [Python Binding Guide](../python/README.md)
- [MATLAB Binding Guide](../matlab/README.md)
- [Julia Binding Guide](../julia/README.md)
