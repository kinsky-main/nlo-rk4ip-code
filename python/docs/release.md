# Python Wheel Release Guide

How `nlolib` wheels reach PyPI, and how to build the same wheels locally.

The automated path is `.github/workflows/release-pypi.yml`. The local path is
the three `tools/build_wheel_*` scripts, which the workflow itself
calls — so a release never runs a code path nobody exercises by hand.

## No published wheel was installable before this pipeline

Worth knowing, because it shapes what the smoke test is for. `load()` in
`nlolib/_binding.py` used to resolve exactly one path after the
`NLOLIB_LIBRARY` override — `<parent-of-package>/Release/nlolib.dll` on
Windows, `<parent-of-package>/libnlolib.so` elsewhere — and raise if it was
missing. But `[tool.setuptools.package-data]` and the build scripts put the
shared library *inside* the package. In an installed wheel that parent
directory is `site-packages`, so `nlolib.load()` raised `OSError` every time.
No wheel ever published could be used without the user pointing
`NLOLIB_LIBRARY` at a library by hand.

It went unnoticed because a local build leaves a second copy at
`Release/nlolib.dll` beside the package directory, outside the package, which is exactly where the
old lookup pointed — so in-tree development worked while the installed
artifact did not. Neither copy is tracked in the repository.

`load()` now tries the package directory first and keeps the in-tree layouts
below it. The smoke test exists to keep this fixed: it installs the wheel into
a clean virtualenv on a runner with no checkout, **unsets `NLOLIB_LIBRARY`**
(which would otherwise let the test pass for the wrong reason) and calls
`nlolib.load()` before running anything.

## What a wheel contains

The Python package wraps the nlolib shared library through `ctypes`. There is
no compiled CPython extension, so the wheel carries `libnlolib.so` /
`nlolib.dll` as package data inside the `nlolib/` package directory and is
ABI-agnostic across interpreter versions.

That combination needs a wheel tag setuptools will not pick on its own, and
`setup.py` exists only to set it:

- Untouched, the distribution looks pure and is tagged `py3-none-any`. One
  such wheel per platform collides on the index: the second upload is rejected
  as a duplicate filename, and whichever won would hand its own platform's
  shared library to every user.
- Declaring extension modules alone tags it `cp3X-cp3X-<plat>`, which is
  platform-correct but pins each wheel to one CPython minor version — N
  identical wheels per platform, for nothing.

So `setup.py` reports the distribution as impure (which puts the package in
platlib at the wheel root, where `auditwheel` and `delvewheel` can find it)
and then overrides `get_tag` back to the generic `py3`/`none`. The result is
one `py3-none-<plat>` wheel per platform.

The platform tag is left at whatever the build machine reports
(`linux_x86_64`, `win_amd64`). `auditwheel repair` and `delvewheel repair`
rewrite it afterwards to the tag the binary has actually earned, rather than a
floor asserted in `setup.py` and never checked.

No sdist is published. A source build needs CMake, a C/C++ toolchain, the
Vulkan loader and `glslangValidator`; shipping an sdist would turn
`pip install nlolib` on an unsupported platform into a long source build that
fails confusingly, instead of a clean "no matching distribution".

## Platforms

| Platform | Runner | Wheel tag | Repair |
| --- | --- | --- | --- |
| Linux x86-64 | `ubuntu-24.04` | `py3-none-manylinux_2_38_x86_64` | `auditwheel` |
| Windows x86-64 | `windows-2025` | `py3-none-win_amd64` | `delvewheel` |

### Why `ubuntu-24.04` and not a manylinux container

`ubuntu-24.04` ships glibc 2.39, and `auditwheel` certifies the built library
as `manylinux_2_38` — read the tag off the build log rather than deriving it
from the image's glibc, which is what the `Check the wheel tags` step exists
for. Either way it is a high floor: `pip install nlolib` will not find a wheel
on Ubuntu 22.04, Debian 12 or RHEL 9. It was still the right trade here:

- The build needs a real `libvulkan.so` on disk and `glslangValidator` on
  `PATH` (see `cmake/ResolveVulkan.cmake` and `cmake/NLOVulkanBackend.cmake` in the core repository).
  On Ubuntu both arrive as `libvulkan-dev` and `glslang-tools`, exactly as in
  the MATLAB toolbox release, which has been building this library for a while.
- A `manylinux_2_28` container (AlmaLinux 8) packages neither reliably, so it
  would mean building the Vulkan loader and glslang from source inside the
  container and pre-seeding `GLSLANG_VALIDATOR` and `vulkan_loader_lib` into
  the CMake cache. That is a lot of bespoke bootstrapping for a release
  pipeline, and nothing has asked for the lower floor yet.
- `ubuntu-22.04` would have given glibc 2.35 for free, but it entered
  deprecation in September 2026 and is removed on 17 April 2027, with 24-hour
  brownouts before that. A release workflow that only runs on tags is the
  worst place for an image that disappears for a day at a time.

**To lower the floor later:** replace the Linux matrix entry with a
`container: quay.io/pypa/manylinux_2_28_x86_64` job, build Vulkan-Loader
(`-DBUILD_WSI_*_SUPPORT=OFF`) and glslang from source in a prior step, pass
`-DGLSLANG_VALIDATOR=<path>` and `-Dvulkan_loader_lib=<path>`, and leave the
rest of the job as it is. `auditwheel` will then certify `manylinux_2_28`
without any change to `setup.py`.

### The Vulkan loader in the wheel

On Linux the shared library has a hard `DT_NEEDED` on `libvulkan.so.1`, so
`auditwheel` vendors the loader into the wheel. That is deliberate: without it
`import nlolib` would fail outright on any machine with no Vulkan runtime,
including CPU-only hosts and the CI runners. A vendored loader still reads
`/usr/share/vulkan/icd.d`, so system GPU drivers are found as usual.

On Windows `vulkan-1.dll` is delay-loaded (`cmake/NLOVulkanBackend.cmake`) and
belongs to the user's graphics driver, so `delvewheel` is given
`--exclude vulkan-1.dll`. `vk_auto_context.c` probes for the loader before
touching any entry point, so its absence stays benign.

### macOS is out of scope

The MATLAB toolbox ships only `glnxa64` and `win64`, and this pipeline
matches it. Two further reasons:

- `tools/build_wheel_macos.sh` configures
  `ENABLE_VULKAN_BACKEND=OFF`, so a macOS wheel would be a CPU-only build with
  different capabilities from the Linux and Windows wheels, published under the
  same version. That is a worse outcome than no macOS wheel.
- GitHub's macOS runners are arm64, so shipping macOS properly means deciding
  arm64 vs x86_64 vs universal2 and a `MACOSX_DEPLOYMENT_TARGET` floor.

`build_wheel_macos.sh` is unchanged and still works for local builds.

## Automated release

The package version lives in `pyproject.toml` and is independent of the core
library version. `CORE_REF` names the core commit the wheels are built from.

1. In a pull request, set the new version in `pyproject.toml` and, if the
   release needs a newer core, the new commit SHA in `CORE_REF`. CI builds that
   core with this package and runs the tests.
2. Merge it. Nothing advances the version on its own, so every commit until
   the next bump keeps naming this release.
3. Tag the merge commit with the bare version — no `v` prefix — and push the
   tag:

```bash
git tag "$(python tools/check_version.py)"
git push origin "$(python tools/check_version.py)"
```

PyPI never accepts the same version twice, so the first release from this
repository has to be higher than the last one published from the core
repository.

The workflow then runs:

- **verify** — `tools/check_version.py`, which fails if the tag name is not
  the version in `pyproject.toml`: the wheel filename carries that version, so
  a mismatch would publish something the tag does not describe. It then checks
  out the core at `CORE_REF` and runs the core's SBOM and API-constant checks
  against this package.
- **build** (per platform) — installs the Vulkan loader and glslang, pins the
  Visual Studio 2022 toolchain on Windows, runs the `build_wheel_*` script,
  then refuses any wheel tagged `-any` or carrying an unpublishable platform
  tag. The core is checked out at `CORE_REF` into `./core` first.
- **smoke** (per platform) — on a clean runner with no checkout and no build
  tooling, installs the wheel into a fresh virtualenv, calls `nlolib.load()`,
  and runs one small propagation on the CPU backend
  (`VECTOR_BACKEND_CPU == 0`). Never Vulkan: the runners have no GPU.
- **publish-pypi** — only on a tag. Uploads every platform's wheel in one call
  via PyPI Trusted Publishing.

### Dry runs

Run the workflow manually from the Actions tab (`workflow_dispatch`). It
builds, checks the tags and smoke-tests both wheels, uploads them as
artifacts, and publishes nothing. Download the artifacts to confirm the
filenames carry real platform tags.

Dispatch with **`testpypi: true`** to additionally publish to TestPyPI. It is
opt-in so that an ordinary dry run has no outward effect at all.

### One-time setup

Trusted Publishing uses OIDC, so there is no API token to store. It does need
configuring once on each index:

- On PyPI, add a trusted publisher for the project: owner `kinsky-main`,
  repository `nlolib-python`, workflow `release-pypi.yml`, environment `pypi`.
- On TestPyPI, the same with environment `testpypi`.

The environment names must match the `environment:` keys in the workflow or
the OIDC claim will not be accepted. The `pypi` and `testpypi` environments
also have to exist in this repository's settings.

## Local build

Needed: Python 3.9+, CMake and a C/C++ toolchain, the Vulkan loader
(`libvulkan-dev` / LunarG SDK) and `glslangValidator` on `PATH`, and a core
checkout: `bash tools/fetch_core.sh` fetches the pinned one into `./core`. When
this repository is the `python/` submodule of the core repository, the scripts
use the surrounding tree instead.

```bash
python -m pip install build auditwheel   # delvewheel on Windows
```

Linux:

```bash
./tools/build_wheel_linux.sh build-wheel-linux Release
```

Windows:

```powershell
.\tools\build_wheel_windows.ps1 -BuildDir build-wheel-win -Config Release
```

macOS (CPU-only, not published):

```bash
./tools/build_wheel_macos.sh build-wheel-macos Release
```

Each script configures with `BUMP_PATCH_ON_BUILD=OFF` and
`INSTALL_GIT_HOOKS=OFF`, and none of them touches a version. A wheel built
from a checkout carries whatever `pyproject.toml` says.

On Linux the unrepaired `linux_x86_64` wheel is deleted after a successful
`auditwheel repair`, so everything left in `dist/` is publishable. Prefer the
automated path for anything that actually ships; if you must upload by hand:

```bash
python -m twine upload dist/*.whl
```

## Known gaps

- Linux and Windows x86-64 only. No aarch64 wheels on either platform.
