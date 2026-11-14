# Manual Release Guide

This document describes manual package publication for Python wheels and MATLAB toolbox bundles.

Version policy for distribution builds:

- Patch component (`x.y.Z`) is bumped by CMake on each distribution build.
  Wheel builds no longer do this -- see [release_python.md](release_python.md).
- Minor/major are not bumped by release scripts.
- Python package metadata is synced from `CMakeLists.txt` after CMake build so wheel and toolbox versions match.

## Prerequisites

- Python 3.9+
- CMake + compiler toolchain
- `python -m pip install build twine`
- Optional wheel repair tools:
  - Windows: `delvewheel`
  - Linux: `auditwheel`
  - macOS: `delocate`
- MATLAB (for `.mltbx` generation)

## Python Package Publication

Wheels are published from CI on a version tag, and the manual commands have
moved along with the rest of that story to
[release_python.md](release_python.md). Read it rather than this section: the
wheel tagging, the platform matrix, and why uploading `dist/*` by hand is no
longer the right default are all covered there.

## MATLAB Toolbox Publication

### Windows

```powershell
.\tools\release\build_mltbx_windows.ps1 -BuildDir build-wheel-win -Config Release
```

### Linux

```bash
./tools/release/build_mltbx_linux.sh build-wheel-linux Release
```

### macOS (CPU-only packaging path)

```bash
./tools/release/build_mltbx_macos.sh build-wheel-macos Release
```

The script writes toolbox artifacts to `dist/` with platform suffixes:

- `nlolib-<version>-win64.mltbx`
- `nlolib-<version>-glnxa64.mltbx`
- `nlolib-<version>-maci64.mltbx`

Upload these files to the corresponding GitHub Release.

Note: the release scripts configure CMake with `SQLITE_USE_FETCHCONTENT=ON`,
which statically links SQLite into `nlolib` and avoids external SQLite runtime DLL dependencies.
