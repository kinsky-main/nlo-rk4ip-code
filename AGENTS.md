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
- Do produce a git command that can be copy-pasted to isolate the change
- Do ensure GPU to host transfers are minimized and efficient
- Do check API changes are carried across all language bindings

## Formatting
- Put doxygen docstrings in function declarations in header files
- Follow hanging indent with vertical alignment not column alignment

## Workflow
- Read relevant files before editing
- Prefer small, reviewable patches
- Explain what and why in responses

## Tests
- Run relevant tests if available
- Note when tests were not run

## PR / review notes
- Call out risks, behavior changes, and follow-ups

## Common Issues
- When running directory commands, check directory location path as many calls are made with invalid paths
