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
- Prefer configuring with repo-mutating options disabled unless the task explicitly needs them:
  - `-DNLO_INSTALL_GIT_HOOKS=OFF`
  - `-DNLO_BUMP_PATCH_ON_BUILD=OFF`
- Docs target is conditional: `NLOLIB_BUILD_DOCS=ON` and Doxygen available, otherwise no `docs` target is created.
- Benchmarks depend on the Vulkan backend; `NLOLIB_BUILD_BENCHMARKS` is forced off when `NLO_ENABLE_VULKAN_BACKEND=OFF`.

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
- Build steps can modify the repo version or install hooks unless `NLO_BUMP_PATCH_ON_BUILD` and `NLO_INSTALL_GIT_HOOKS` are disabled.
