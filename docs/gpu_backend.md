# Vulkan Backend Behaviour

This page covers what the Vulkan backend does differently from the CPU backend
at runtime: the operator JIT, the fused RK algebra, the compact tensor axes,
and the switches that turn each of them off. None of it changes the public C,
Python, MATLAB or Julia API, and none of it affects the CPU backend.

## The Operator JIT

`operator_program_compile` lowers every runtime-operator expression to a value
DAG, with constants folded and repeated subexpressions shared. On Vulkan each
lowered program is then compiled to one compute kernel that evaluates the whole
expression per element, instead of one dispatch per instruction over stack
buffers.

It covers the expressions compiled at init: the potential, linear-factor and
linear expressions in tensor mode, the dispersion-factor and dispersion
expressions otherwise, and the nonlinear expression in both. Compiled SPIR-V is
cached for the life of the process, keyed by the lowered program, so repeated
`propagate` calls with the same operators do not recompile.

The interpreter stays as the fallback and is used whenever the JIT does not
apply: the CPU backend, a program that did not lower, a glslang failure, or
`NLO_OPERATOR_JIT=0`. A compilation failure is not fatal; it logs

```
[nlolib] Vulkan operator JIT unavailable for <label>; using the interpreter.
```

once per program at warning level and the run continues on the interpreter.

### Checking whether the JIT ran

Two counters in `nlo_perf_profile_snapshot`, read back through
`nlolib_perf_profile_read`, report what the JIT actually did:

- `operator_jit_programs` — operator programs bound to a JIT-compiled GPU
  kernel, whether freshly compiled or taken from the cache.
- `operator_jit_fallbacks` — programs left on the interpreter, because the
  program did not lower, because `NLO_OPERATOR_JIT=0` turned the JIT off, or
  because compilation failed.

Unlike every other counter in that struct, these two accumulate
unconditionally: they do not need `nlolib_perf_profile_set_enabled`, so an
ordinary Release build can tell a real JIT run from a silent fallback to the
interpreter. They are counted once per active operator program as a simulation
state is built, not per step, and are cumulative until
`nlolib_perf_profile_reset`, so a process that builds several states
accumulates over all of them.

Both stay zero wherever the JIT is never reached: on the CPU backend, and in a
build configured with `ENABLE_VULKAN_BACKEND=OFF`.

The C API, Python (`perf_profile_read().operator_jit_programs`) and MATLAB all
expose them. Julia has no perf-profile binding.

Measured solve-time speedup against the interpreter, Linux / Quadro P4000 /
NVIDIA 535, Release, fixed step: ×2.10 on the 1-D expression case, ×1.58–1.60
on the (2+1)D GRIN cases, and ×1.21–1.22 on the (3+1)D tensor cases, which are
linear-only and so leave less interpreter work to fuse.

### New GPU coverage

Under the JIT, complex `log`, `sin`, `cos` and `pow` with a non-literal
exponent run on the GPU. Through the interpreter they are CPU-only:
`vec_complex_log_inplace`, `vec_complex_sin_inplace`,
`vec_complex_cos_inplace` and `vec_complex_pow_elementwise_inplace` return
`VEC_STATUS_UNSUPPORTED` on a Vulkan backend. `exp`, `sqrt` and real-literal
powers already had Vulkan kernels.

### Two caveats

- **The first compile in a process costs time.** glslang adds 260–330 ms to
  the first `propagate` that compiles a kernel. Later calls hit the SPIR-V
  cache, and a warm init is then 19–43 ms *faster* than the interpreter's. A
  process that propagates once pays the compile and may finish slower overall
  than the interpreter; a process that propagates repeatedly pays it once.
- **GPU transcendentals are less accurate than the CPU's.** They come from
  `double_math.glslinc`, not libm. Against the CPU result, division lands
  around 1.5e-11, `log` around 5.6e-11, and `pow`, `sin` and `cos` up to about
  1.8e-7, where the CPU backend holds 1e-16. This is a property of the GPU
  helpers, not of the JIT: the Vulkan interpreter already evaluated
  real-literal powers through them. The JIT itself adds nothing to the gap: it
  agrees with the Vulkan interpreter to ≤ 1.9e-16. Against the CPU,
  `test_operator_jit_vulkan` sets the bound per expression by the helpers it
  uses — 1e-12 for plain algebra and `exp`, 1e-10 with division, 1e-9 with
  `sqrt` or `log`, and 1e-6 for `pow` and `sin`/`cos`.

## Fused RK Algebra

Each RK update that was a copy/scale/add chain is one kernel,
`dst = Σ cₖ·srcₖ` over two to four real-weighted vectors. Per step on Vulkan
that takes the RK4 algebra from 19 dispatches to 6, the ERK4(3) algebra from 24
to 8, and an interpolated record from 5 to 1. It is worth a further 8–13 % on
top of the JIT, for ×1.32–2.40 against the interpreter overall.

Buffer roles are unchanged, so k5 reuse and the step controller see the same
contract. The unfused sequence is still in the source and is used by the CPU
backend always, and on Vulkan with `NLO_DISABLE_VK_FUSED_STEP=1`.

## Compact Tensor Axis Symbols

Tensor-mode operator expressions may read the coordinate symbols `wt`, `kx`,
`ky`, `t`, `x` and `y`. Expanding each 1-D axis into a full-grid mesh at init
costs six buffers the size of the field. JIT kernels index the 1-D axes
directly, so when every program that reads them is a JIT kernel the meshes are
not allocated at all. That cuts peak process GPU memory by about 22 % on the
GRIN cases, at unchanged solve time.

With the interpreter, after a JIT failure, or on the CPU backend, the meshes
are allocated exactly as before.

## Environment Switches

| Variable | Set to | Effect |
|---|---|---|
| `NLO_OPERATOR_JIT` | exactly `0` | Keeps every operator program on the interpreter. |
| `NLO_DISABLE_VK_FUSED_STEP` | `1` | Restores the unfused RK algebra on Vulkan. |
| `NLO_GPU_TIMESTAMPS` | `1` | Creates the Vulkan timestamp query pool. |

The first two are debugging and comparison switches, not tuning knobs: the
defaults are the faster path on every case measured, and these exist to
reproduce the interpreter or unfused numbers and to tell a suspected JIT or
fusion bug from one in the surrounding code. The third is instrumentation.

Their value parsing is deliberately narrow, and worth knowing before relying on
a shell export:

- `NLO_OPERATOR_JIT` disables the JIT only when the value is exactly `0`. Unset
  or anything else, `00` and `false` included, leaves the JIT on wherever it
  applies.
- `NLO_DISABLE_VK_FUSED_STEP` and `NLO_GPU_TIMESTAMPS` are off when unset,
  empty, or starting with `0`, and on for any other value. Use `1`.
- `NLO_GPU_TIMESTAMPS` is read when the backend is created, so it must be set
  before the first simulation state is built. Changing it mid-process does
  nothing.

`operator_jit_programs` and `operator_jit_fallbacks` are how to confirm that
`NLO_OPERATOR_JIT=0` took effect: with it set, every active operator program is
counted as a fallback and no programs are counted.

## GPU Phase Timings

With `NLO_GPU_TIMESTAMPS=1` and profiling enabled through
`nlolib_perf_profile_set_enabled`, `nlolib_perf_profile_read_gpu_phases`
returns phase-tagged timestamp results: six phase totals in milliseconds
(other, FFT, linear, nonlinear, RK algebra, error norm), six interval counters,
and four host slots (init ms, solve ms, queue submits, dropped timestamps).

This is a C API call. The bindings expose the rest of the perf-profile surface
(`perf_profile_set_enabled`, `perf_profile_is_enabled`, `perf_profile_reset`,
`perf_profile_read`) but not the GPU phase read. Unlike the two JIT counters
above, the phase timings need profiling enabled as well as the switch. The query pool is allocated
only when the switch is set, so an unset run carries no query cost.

## Where the Numbers Come From

Every figure above was measured on Linux with a Quadro P4000 (NVIDIA 535.179),
Release, through `benchmarks/assessment/assess_bench.py`, as medians of
interleaved processes with warm-ups discarded. Expect different ratios on
different hardware; the ordering has held on everything measured so far.

`docs/gpu-jit-cuda-assessment.md` is the dated assessment that led to this
work, including the CUDA and NVRTC variants that were measured and archived
rather than ported. It is a record of that assessment, not a description of
current behaviour.

## Build Requirements

The JIT compiles GLSL at runtime, so glslang is linked into `nlolib` whenever
`ENABLE_VULKAN_BACKEND=ON`, independently of `ENABLE_VKFFT`. The GLSL include
files the generated source `#include`s are embedded in the library at build
time, so an installed wheel or toolbox does not need the source tree.
`NLOLIB_GLSLANG_PROVIDER` selects where glslang comes from; see
[Build and Install](build_and_install.md).
