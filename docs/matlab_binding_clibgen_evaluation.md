# Evaluating clibgen for the MATLAB binding

**Question.** Can MATLAB's `clibgen` interface generator replace the hand-written
`loadlibrary`/`calllib` wrapper in `matlab/+nlolib`, removing the need to
hand-maintain the flattened FFI header `src/nlolib_matlab.h`?

**Answer: no, not without restructuring the C entry points.** The blocking
constraint is structural, not cosmetic. Details and reproduction below.

## Method

Run against the real library on the development machine:

- MATLAB **R2025b**, `clibgen` present
- Compiler: **Microsoft Visual C++ 2022 17.0** (the supported MEX C++ compiler)
- Header: `src/nlolib_matlab.h`; import library: `python/Release/nlolib.lib`

```matlab
clibgen.generateLibraryDefinition( ...
    'src/nlolib_matlab.h', ...
    'Libraries', 'python/Release/nlolib.lib', ...
    'PackageName', 'nlolibclib', ...
    'OutputFolder', outDir, ...
    'Verbose', true);
```

## Findings

**1. Generation does not complete cleanly.** The call raises an internal
MathWorks error while formatting its own diagnostic:

```
MATLAB:builtins:IncorrectHoleCount
Incorrect number of parameters supplied for 'MATLAB:CPP:AddMemberFailed_link'.
Expected 5 but found 4.
```

A definition file is still written, but at least one member failed to be added
and MATLAB could not report which.

**2. The functions that matter are auto-commented as undefinable.** Of 14
exported functions, `clibgen` emitted 5 commented out, requiring manual
annotation — and they are the core of the API:

| Function | Auto-defined? |
|---|---|
| `nlolib_propagate` | **no** |
| `nlolib_query_runtime_limits` | **no** |
| `nlolib_perf_profile_read` | **no** |
| `nlolib_set_log_file` | **no** |
| `nlolib_read_log_buffer` | **no** |
| `nlolib_set_log_level`, `nlolib_set_progress_options`, `nlolib_perf_profile_reset`, … | yes (scalar-only signatures) |

The generated definition contains **95** `<MLTYPE>`/`<SHAPE>`/`<DIRECTION>`
placeholders that a human must fill in.

**3. The structural blocker: caller-allocated output buffers.** Every pointer
member of `propagate_output` is commented out pending a compile-time `SHAPE`:

```matlab
%addProperty(propagate_outputDefinition, "output_records", "clib.array.nlolibclib.Double", <SHAPE>, ...
%addProperty(propagate_outputDefinition, "records_written", "clib.array.nlolibclib.UnsignedLongLong", <SHAPE>, ...
%addProperty(propagate_outputDefinition, "storage_result", "clib.nlolibclib.storage_result", <SHAPE>, ...
%addProperty(propagate_outputDefinition, "output_step_events", "clib.nlolibclib.step_event", <SHAPE>, ...
```

`output_records` holds `records_written × num_time_samples` complex values.
That length is a **runtime product of a sibling struct field and a function
argument**. A `clibgen` SHAPE may be a literal, a sibling *parameter* of the
same function, or a sibling *property* of the same class — it cannot be a
product spanning an argument and a member of a different struct. The
"caller allocates, library fills, library reports how much it filled" pattern
that `nlolib_propagate` is built around is therefore not expressible in the
`clib` object model.

`loadlibrary` has no such restriction: MATLAB passes a raw typed pointer and
the wrapper interprets the written length itself.

**4. It would not remove the maintenance burden.** `clibgen` replaces one
hand-maintained artifact (the flattened header) with another (a generated
definition file plus ~95 hand-filled placeholders, regenerated and re-edited
on every ABI change). The desync class of bug that motivated this evaluation
would move, not disappear.

**5. Distribution regression.** `clib` interfaces are compiled artifacts tied
to a specific MATLAB release *and* platform, and the documented workflow
requires R2023a+. The toolbox currently declares `MinimumMatlabRelease =
R2019b` and ships one shared library that works across releases. Adopting
`clibgen` means publishing one built interface per supported MATLAB release.

## Decision

Keep `loadlibrary`/`calllib`. Address the actual failure mode — silent drift
between the wrapper's type-name strings, the flattened header, and the
canonical headers — with automated checks instead:

- `cmake/check_matlab_binding_types.cmake` (CTest: `test_matlab_binding_types`)
  verifies every `libstruct('T')` / `libpointer('TPtr')` name in `matlab/**.m`
  against the typedefs in `src/nlolib_matlab.h`. Needs no MATLAB. This is the
  check that catches the `nlo_complex` → `complex` class of regression.
- `tests/matlab/test_ffi_smoke.m` (CTest: `test_matlab_bindings`) loads the
  library and exercises propagate, step-history capture, runtime limits, log
  buffer, and perf counters end to end.

## What would make clibgen viable later

A thin C shim exposing flat, shape-expressible signatures, e.g.

```c
nlolib_status nlolib_propagate_flat(
    const simulation_config* sim, const physics_config* phys,
    size_t num_time_samples, const nlo_complex* input_field,
    size_t num_records,
    nlo_complex* out_records,      /* SHAPE: [num_records, num_time_samples] */
    size_t* out_records_written);  /* SHAPE: 1 */
```

Multi-dimensional shapes referencing two sibling *parameters* are expressible,
so this form can be annotated. It is a new public C surface plus a second
MATLAB API, and each optional feature (storage, step events, progress,
explicit-z schedules) needs its own flattened entry point. Worth revisiting
only if a `clib`-based binding becomes a requirement in its own right.
