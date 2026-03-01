# AGENTS.md

## Scope
- Repo: nlolib
- Goal: concise, editable agent guidance

## Repo layout
- Key dirs: src/, tests/, examples/
- Build files: CMakeLists.txt

## Do / Don’t
- Do keep changes minimal and focused
- Do preserve existing style and conventions
- Don’t reformat unrelated code
- Don’t change public APIs without explicit request
- Don't make per step memory allocations; keep all allocations static
- Do ensure GPU to host transfers are minimized and efficient
- Do check API changes are carried across all language bindings
- Do prioritize the GPU backend
- Do update README build/test/docs sections when CMake options or targets change
- Do keep Windows and Linux command examples in sync for setup/build/test paths

## Formatting
- Put doxygen docstrings in function declarations in header files
- Follow hanging indent with vertical alignment not column alignment
- Keep public API doxygen comments actionable: describe parameters, return values, units, and ownership/lifetime where relevant
- Prefer short, precise doxygen blocks over duplicated prose in source definitions

## Workflow
- Read relevant files before editing
- Prefer small, reviewable patches
- Explain what and why in responses
- For doc updates, verify commands against current CMake targets/options before finalizing

## Tests
- Run relevant tests if available
- Create new tests if necessary to cover new functionality or edge cases
- Ensure tests are deterministic and do not rely on external state
- Ensure tests run within a reasonable time frame
- Note when tests were not run
- When editing test/build docs, validate expected tests with `ctest --test-dir <build-dir> -N`

## PR / review notes
- Call out risks, behavior changes, and follow-ups
- Call out documentation behavior changes (new options, defaults, or targets)

## Common Issues
- When running directory commands, check directory location path as many calls are made with invalid paths
- Docs target is conditional: `NLOLIB_BUILD_DOCS=ON` plus Doxygen available, otherwise no `docs` target is created
