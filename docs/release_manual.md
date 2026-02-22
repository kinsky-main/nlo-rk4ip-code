# Manual Release Guide

This document describes manual package publication for Python wheels and MATLAB toolbox bundles.

Version policy for distribution builds:

- Patch component (`x.y.Z`) is bumped by CMake on each distribution build.
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

### Windows

```powershell
.\tools\release\build_wheel_windows.ps1 -BuildDir build-wheel-win -Config Release
python -m twine upload dist\*
```

### Linux

```bash
./tools/release/build_wheel_linux.sh build-wheel-linux Release
python3 -m twine upload dist/*
```

### macOS (CPU-only packaging path)

```bash
./tools/release/build_wheel_macos.sh build-wheel-macos Release
python3 -m twine upload dist/*
```

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

Note: the release scripts configure CMake with `NLO_SQLITE_USE_FETCHCONTENT=ON`,
which statically links SQLite into `nlolib` and avoids external SQLite runtime DLL dependencies.
