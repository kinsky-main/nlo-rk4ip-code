# AGENTS.md

## Scope
- Repo: `nlolib`
- Goal: concise, editable agent guidance for this repository

## Repo Layout
- Core library: `src/`
- Bindings: `python/`, `matlab/`, `julia/` are git submodules, one repository
  each (`nlolib-python`, `nlolib-matlab`, `NLOLib.jl`). Each holds its own
  package, tests, examples, README, version, and release workflow.
- Core tests: `tests/core/`, `tests/fft/`, `tests/numerics/`, `tests/abi/`
- Executables, demos, and benchmark code: `examples/c/`, `benchmarks/`
- Docs and release helpers: `docs/`, `tools/`
- Build entry points: `CMakeLists.txt`, `README.md`

## Submodules
- Run `git submodule update --init` after cloning; a binding that is not checked
  out is skipped by CMake along with its targets and tests.
- A change inside `python/`, `matlab/` or `julia/` is a commit in that
  submodule's repository, not in this one. Commit and push it there first, then
  commit the moved submodule pointer here.
- Each binding pins the core commit it is released from in its `CORE_REF` file.
  A cross-cutting change lands core first, then the binding (with `CORE_REF`
  moved), then the submodule pointer here.
- Binding versions are independent of the core version and live in the binding
  (`pyproject.toml`, `VERSION`, `Project.toml`). Release tags are pushed to the
  binding repository, not here.

## Build Notes
- Top-level build is CMake; root `CMakeLists.txt` adds `src/`, each checked-out binding (`python/`, `matlab/`, `julia/`, or the directory named by `NLOLIB_PYTHON_DIR` / `NLOLIB_MATLAB_DIR` / `NLOLIB_JULIA_DIR`), `tests/`, `benchmarks/`, and `examples/`.
- Both repo-mutating options now default to `OFF`; pass them explicitly only to
  turn one on:
  - `-DBUMP_PATCH_ON_BUILD=ON` rewrites the version in `CMakeLists.txt` after a
    successful build. Never use it for a release build: the packaged version is
    read back out of `CMakeLists.txt`, so a mid-build bump publishes a version
    that does not match its tag.
  - `-DINSTALL_GIT_HOOKS=ON` installs a pre-commit hook that only checks the
    SBOM agrees with the version and dependency pins. It does not change any file.
- The core version is advanced deliberately, by one command:
  `python tools/release/check_versions.py --bump minor`. It rewrites
  `CMakeLists.txt` and `sbom.cdx.json` together, so every commit names one
  self-consistent version.
- Docs target is conditional: `NLOLIB_BUILD_DOCS=ON` and Doxygen available, otherwise no `docs` target is created.
- Benchmarks depend on the Vulkan backend; `NLOLIB_BUILD_BENCHMARKS` is forced off when `ENABLE_VULKAN_BACKEND=OFF`.

## Do / Don't
- Do keep changes minimal and focused.
- Do preserve existing style and conventions.
- Do prefer the GPU/Vulkan backend path when touching shared execution logic.
- Do keep GPU-to-host transfers minimized and efficient.
- Do keep allocations static in hot paths; do not introduce per-step allocations.
- Do keep C API, Python, MATLAB, and Julia surfaces aligned when changing shared behavior.
- Do change the bindings' public surface through `tests/api_surface.json` and `docs/binding_api_vocabulary.md` first: the three bindings share one vocabulary (one `propagate` entry point, plain-function operators, lowercase string enum values, no client class) and `tools/check_api_parity.py` fails on any difference.
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
  - MATLAB changes: `test_matlab_bindings` only when `NLOLIB_BUILD_MATLAB_TESTS=ON` and MATLAB is available
- Note when tests were not run.

## Common Issues
- Python tests are part of the CTest surface when `BUILD_TESTING=ON`, so Python availability matters for full test runs.
- The docs target is optional even when `NLOLIB_BUILD_DOCS=ON`; missing Doxygen means no `docs` target is generated.
- Only `BUMP_PATCH_ON_BUILD=ON` makes a build rewrite the repo version; it is off by default.
- `test_sbom_pins_match` fails when `sbom.cdx.json` is behind `CMakeLists.txt` or the dependency pins. Fix it with `python tools/release/check_versions.py --write`, not by editing the file by hand.
- `tools/check_api_parity.py` fails with a missing-source error when a binding submodule is not checked out; run `git submodule update --init`.
- `test_api_parity` fails when a mounted binding differs from `tests/api_surface.json`; the output names each difference. `--surface report` prints them without failing.
