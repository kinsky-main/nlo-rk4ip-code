# AGENTS.md

## Scope
- Repo: `nlolib`
- Goal: concise, editable agent guidance for this repository

## Repo Layout
- Core library: `src/`
- Bindings and package surfaces: `python/`, `matlab/`, `julia/`
- Tests: `tests/core/`, `tests/python/`, `tests/julia/`, `tests/matlab/`
- Executables, demos, and benchmark code: `examples/`, `benchmarks/`
- Docs and release helpers: `docs/`, `tools/`
- Build/package entry points: `CMakeLists.txt`, `README.md`, `pyproject.toml`

## Build Notes
- Top-level build is CMake; root `CMakeLists.txt` adds `src/`, `python/`, `matlab/`, `julia/`, `tests/`, `benchmarks/`, and `examples/`.
- Both repo-mutating options now default to `OFF`; pass them explicitly only to
  turn one on:
  - `-DBUMP_PATCH_ON_BUILD=ON` rewrites the version in `CMakeLists.txt` after a
    successful build. Never use it for a release build: the packaged version is
    read back out of `CMakeLists.txt`, so a mid-build bump publishes a version
    that does not match its tag.
  - `-DINSTALL_GIT_HOOKS=ON` installs a pre-commit hook that only checks the
    version copies agree. It does not change any file.
- The version is advanced deliberately, by one command:
  `python tools/release/check_versions.py --bump minor`. It rewrites
  `CMakeLists.txt`, the `pyproject.toml` / `julia/Project.toml` copies and
  `sbom.cdx.json` together, so every commit names one self-consistent version
  and a release tag can be cut from any commit.
- Docs target is conditional: `NLOLIB_BUILD_DOCS=ON` and Doxygen available, otherwise no `docs` target is created.
- Benchmarks depend on the Vulkan backend; `NLOLIB_BUILD_BENCHMARKS` is forced off when `ENABLE_VULKAN_BACKEND=OFF`.

## Do / Don't
- Do keep changes minimal and focused.
- Do preserve existing style and conventions.
- Do prefer the GPU/Vulkan backend path when touching shared execution logic.
- Do keep GPU-to-host transfers minimized and efficient.
- Do keep allocations static in hot paths; do not introduce per-step allocations.
- Do keep C API, Python, MATLAB, and Julia surfaces aligned when changing shared behavior.
- Do update `README.md` build/test/docs or binding usage sections when CMake options, targets, staging flows, or public workflows change.
- Don't reformat unrelated code.
- Don't change public APIs without explicit request.
- Don't add new dependencies or large abstractions without clear need.

## Formatting
- Put Doxygen docstrings on public function declarations in header files.
- Keep public API Doxygen comments actionable: document parameters, return values, units, and ownership/lifetime where relevant.
- Prefer short, precise Doxygen blocks over duplicated prose in source definitions.
- Follow hanging indent with vertical alignment, not column alignment.

## Workflow
- Read relevant files before editing.
- Prefer small, reviewable patches.
- Prioritize subtractive edits over additive edits when possible.
- Match existing local patterns before introducing a new helper or abstraction.
- When correcting a previous change, state the current behavior and the requested replacement so the intended delta is explicit.
- For docs updates, verify commands and option names against current CMake targets and README examples before finalizing.

## Tests
- Run relevant tests when available.
- Add or update tests when behavior or public-facing outputs change.
- Keep tests deterministic and independent of external state.
- Prefer the smallest test slice that covers the changed surface:
  - core C/CMake changes: targeted `ctest` cases under `tests/core`
  - Python binding/example changes: targeted `ctest -R "^test_python_.*"` or the specific Python test
  - Julia changes: targeted Julia `ctest` cases
  - MATLAB changes: `test_matlab_runtime_handle_parser` only when enabled and MATLAB is available
- Note when tests were not run.

## Common Issues
- Python tests are part of the CTest surface when `BUILD_TESTING=ON`, so Python availability matters for full test runs.
- The docs target is optional even when `NLOLIB_BUILD_DOCS=ON`; missing Doxygen means no `docs` target is generated.
- Only `BUMP_PATCH_ON_BUILD=ON` makes a build rewrite the repo version; it is off by default.
- `test_version_copies_match` fails when `CMakeLists.txt`, `pyproject.toml` and `julia/Project.toml` disagree, and `test_sbom_pins_match` fails when `sbom.cdx.json` is behind. Fix both with `python tools/release/check_versions.py --write`, not by editing one file by hand.
