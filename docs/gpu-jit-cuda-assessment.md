# GPU JIT / CUDA branch assessment

Status: **PAUSED 2026-11-14, marked for follow-up** — Phases 0 and 3 done, Phase 1/2 partial (1-D cases only), Phase 4 not started. Resume steps at the end of this file. Plan: `docs/tasks-branch-assessment-plan.md` on
`claude/worktree-audit-9d3049`. Subject: `optimisation/runtim-vulkan-compilation`
@ `ad0d8ac` plus the saved uncommitted NVRTC work, snapshotted as local branch
`assess/tasks-snapshot` (never pushed). Baseline: `main` @ `a23ccff`.

## Pre-registered decision thresholds

Fixed before any Phase 1/2 measurement. These are the plan's suggested
thresholds, made precise so they can be applied mechanically.

- **Metric.** End-to-end wall time of the propagate call (state init excluded,
  reported separately), median of ≥ 10 timed runs after 3 warm-ups, IQR
  reported. "x % faster" means `t_feature ≤ (1 − x/100) · t_baseline`.
- **Representative GPU cases.** The Phase 1 workload matrix (1-D Kerr,
  1-D Kerr + Raman, (2+1)-D GRIN, (3+1)-D tensor dispersion, runtime
  expression), each at its largest listed size that fits in 8 GB.
- **Correct.** Final field relative L2 intensity error ≤ 1e-10 against the
  reference build on fixed-step runs, plus matching energy drift and step-size
  history on adaptive runs. A feature that fails this gets no speed claim.
- **Port**: ≥ 20 % faster on at least two representative cases, correct, and
  no runtime dependency that breaks the wheel, MATLAB toolbox or Julia
  artifact; *or* it enables something `main` cannot do at all.
- **Port as optional** (off by default, not in release artifacts): meets the
  speed and correctness bar but needs heavy runtime dependencies.
- **Archive**: < 10 % faster on every case, or cannot be made correct.
- **10–20 %**: no automatic verdict; decided on port size, recorded with reasons.
- **Gate 0**: the snapshot cannot be built and pass `test_nlo_vector_backend`,
  `test_fft`, `test_core_state_alloc`, `test_core_perf_profile` within about a
  day → "not salvageable without a rewrite".
- **Gate 1**: *f_op* (operator evaluation + RK algebra share of GPU step time on
  `main`) < 15 % on every GPU case → A and D are low value by Amdahl's law.

## Phase 1/2: measurements so far (PARTIAL — paused 2026-11-14)

Harness: `benchmarks/assessment/assess_bench.py` (same Python script for both
trees; `--tree`/`--lib` pick the build), driven by `run_matrix.sh`, with
variants interleaved inside each case. Metric: library-internal `solve_rk4`
wall time (init timed separately), fixed step, 200 steps for the 1-D cases,
3 warm-ups + 10 timed runs, median (IQR). Raw rows:
`docs/assessment-data/fixed-partial-2026-11-14.jsonl`; correctness:
`docs/assessment-data/fixed-partial-correctness.csv`.

Variants: `main` (interpreter); `snap_interp` (JIT off, fused off);
`snap_fused` (B alone); `snap_jit` (A alone); `snap_jit_fused` (A+B, branch
default); `cuda` (C, JIT off); `nvrtc_jit` (C+D). `main_ts` is `main` with
timestamp queries on (phase breakdown only).

Covered: all 1-D cases on all variants. Not yet: GRIN (2+1)-D and (3+1)-D
tensor cases for the snapshot variants (only `main` rows exist), adaptive
runs (step-size history), memory headroom (largest grid per backend).

### Solve time, median ms (IQR) and speedup vs main

| Case | main | snap_interp | snap_fused (B) | snap_jit (A) | snap_jit_fused (A+B) | cuda (C) | nvrtc_jit (C+D) |
|---|---|---|---|---|---|---|---|
| kerr1d 1024 | 122 (19) | 120 x1.01 | 121 x1.00 | 90 x1.36 | 69 x1.76 | 137 x0.88 | 69 x1.76 |
| kerr1d 4096 | 156 (8) | 285 (131) x0.55 | 151 x1.03 | 225 (98) x0.69 | 109 x1.43 | 153 x1.02 | 105 x1.49 |
| kerr1d 16384 | 124 (12) | 120 x1.03 | 123 x1.01 | 114 (71) x1.08 | 104 (45) x1.19 | 155 x0.80 | 98 x1.27 |
| kerr1d 65536 | 273 (169) | 204 x1.34 | 207 x1.32 | 156 x1.75 | 125 x2.18 | 190 x1.43 | 132 x2.06 |
| raman1d 4096 | 259 (4) | 243 x1.07 | 235 x1.10 | 224 (57) x1.15 | 227 (67) x1.14 | 228 x1.13 | 207 x1.25 |
| raman1d 16384 | 184 (8) | 202 x0.91 | 179 x1.03 | 187 x0.98 | 173 x1.06 | 221 x0.83 | 196 x0.94 |
| expr1d 16384 | 215 (44) | 179 x1.20 | 182 x1.18 | 127 x1.69 | 96 x2.23 | 242 x0.89 | 104 x2.06 |

Caveats: several IQRs are large (laptop GPU clock/thermal noise), for example
`main` kerr1d 65536 and `snap_interp`/`snap_jit` kerr1d 4096; the
`snap_interp` ≈ `main` sanity check holds except in those noisy cells. Rerun
before quoting single cells.

### Init time (ms, cold first call / median of timed calls)

JIT adds ~250–400 ms per propagate call on Vulkan (`snap_jit*` ≈ 500–650 ms vs
`main` ≈ 200–230 ms): the JIT cache lives on the backend, so every
`propagate` recompiles. CUDA init is ~20 ms warm, NVRTC ~60 ms warm (driver
caches). First-in-process cold init is noisy (`main` 10.5 s and `nvrtc_jit`
6.3 s on the very first run of the session: driver shader-cache warm-up).

### Gate 1: main's GPU phase breakdown (Vulkan timestamps, % of solve time)

| Case | FFT | linear | nonlinear | RK algebra | GPU idle / host | f_op |
|---|---|---|---|---|---|---|
| kerr1d 1024 | 24 % | 12 % | 14 % | 8 % | 43 % | 33 % |
| kerr1d 4096 | 52 % | 9 % | 9 % | 6 % | 24 % | 24 % |
| kerr1d 16384 | 44 % | 16 % | 11 % | 9 % | 21 % | 36 % |
| kerr1d 65536 | 37 % | 18 % | 13 % | 11 % | 21 % | 42 % |
| raman1d 4096 | 63 % | 5 % | 9 % | 3 % | 20 % | 18 % |
| raman1d 16384 | 44 % | 7 % | 12 % | 5 % | 32 % | 24 % |
| expr1d 16384 | 23 % | 8 % | 28 % | 5 % | 36 % | 41 % |

**Gate 1 does not trigger:** f_op is 18–42 % on every 1-D case, above the 15 %
threshold. At these sizes per-step cost is dominated by per-dispatch latency
and host overhead (solve time barely grows from nt = 1024 to 65536), which is
exactly what JIT fusion attacks. "FFT" includes the copies and the inverse
scaling around each VkFFT call.

### Correctness so far (relative L2 of final field vs main)

| Case | snap_interp | snap_fused | snap_jit | snap_jit_fused | cuda / nvrtc_jit | main_cpu (FFTW) |
|---|---|---|---|---|---|---|
| raman1d 4096 / 16384 | ≤ 9e-15 | ≤ 1e-14 | ≤ 9e-15 | ≤ 1e-14 | 6e-13 / 7e-14 | 8e-14 / 3e-13 |
| expr1d 16384 | 0 | 0 | 1e-14 | 1e-14 | 3e-11 | 3e-11 |
| kerr1d (all sizes) | 0 | 0 | 0 | 1e-8 – 5e-8 | 1e-6 – 2e-6 | 9e-8 – 1.2e-6 |

Interpretation: on the well-resolved cases (Raman, runtime expression) every
variant agrees to ≤ 3e-11, inside the 1e-10 bar; JIT without fusion is
bit-identical to `main` on Kerr. The kerr1d case (γ = 80, 16 rad nonlinear
phase) is under-resolved at the benchmark step (5e-2 discretisation error vs a
16× refined run), so it amplifies round-off: `main`'s own CPU and Vulkan
paths differ by up to 1.2e-6 there, and the difference grows linearly with
step count. By the pre-registered bar, fused (A+B) and CUDA fail on kerr1d;
by "no worse than main's own CPU/GPU disagreement" fused passes and CUDA is
roughly at that level (1.1–1.9e-6 vs 0.1–1.2e-6). Decide which reading
applies before Phase 4; adaptive step-history comparison is still to do.

### Provisional reading (not a verdict)

- A (Vulkan JIT): 1.36–1.75× alone on 3 of 7 1-D cases, up to 2.2× with B, no
  new runtime dependency. On track for **port** if the 2-D/3-D cases agree.
- B (fused plan): ≤ 1.03× alone except where noise dominates; its value is in
  combination with A. The profiling half is independently useful.
- C (CUDA): 0.80–1.43×, mostly slower than or level with `main`'s Vulkan; heavy
  runtime deps. Leaning **archive** unless headroom shows bigger grids fit.
- D (NVRTC): ~equal to A+B on Vulkan; no gain over porting A. Leaning
  **archive** with C.

## Phase 0: snapshot build (Gate 0: passed)

Snapshot worktree `.claude/worktrees/assess-tasks-snapshot`, local branch
`assess/tasks-snapshot` (not pushed): `ad0d8ac` + saved
`tasks-uncommitted.patch` + `cuda_operator_jit.cu` (`683748c`), then two
test-only fixes (`19f6d80`) and assessment timers (`9fafec0`).

Toolchain: VS 2022 17.14 (MSVC 19.44), CUDA 13.0.88, Vulkan SDK 1.4.341.1,
Release, `-DNLO_INSTALL_GIT_HOOKS=OFF -DNLO_BUMP_PATCH_ON_BUILD=OFF`
(the branch predates main's option rename). Device: RTX 4070 Laptop GPU
(8 GB, driver 616.92), selected automatically as the discrete device for
Vulkan and as ordinal 0 for CUDA; AC power, Balanced plan.

| Build | Configures | Builds | Gate 0 tests |
|---|---|---|---|
| Vulkan only | yes | after fix 1 | 4/4 pass (Release and Debug) |
| Vulkan + CUDA + cuFFT | yes | yes, no source changes | 4/4 after fixes 2–3 (Debug), `test_core_state_alloc` crashes in Release |
| + NVRTC | yes | yes, no source changes | as above; NVRTC warm-up subtest passes |

Friction log (maintenance-cost evidence):

1. `test_core_perf_profile` did not link in a Vulkan-only build: the uncommitted
   patch made `vulkan_stubs.c` call CUDA JIT entry points but did not add
   `cuda_backend_stubs.c` to that test target.
2. `test_state_alloc.c` puts `nlo_init_simulation_state(...)` and similar calls
   *inside* `assert()`. Release builds define `NDEBUG`, so the calls vanish and
   the CUDA subtests dereference a NULL state (access violation). The test is
   only meaningful in Debug.
3. The branch changed `AUTO` to prefer CUDA when built, but its own Vulkan
   subtests still request `AUTO` and assert a Vulkan backend.
4. CUDA 12 → 13: no changes needed. CUDA 13.0 still ships `cufft64_12.dll`;
   NVRTC is `nvrtc64_130_0.dll` (+ `nvrtc-builtins64_130.dll`).
5. 5 of 10 Python tests fail on the snapshot for reasons internal to the branch
   (tests import helpers the branch renamed, a missing `grin` module, a
   `storage_enabled` meta assertion, a fixed-vs-adaptive accuracy assertion).
   Not caused by A–D, not fixed.
6. Every build directory writes `python/Release/nlolib.dll` in the source tree
   (also true on `main`), so parallel variant builds clobber each other. Each
   variant DLL was rebuilt in turn and copied to `dist-<variant>/`.

## Phase 3: cost of landing it

### Merge cost

- `git merge-tree main assess/tasks-snapshot`: **48 conflicting files**
  (44 for `ad0d8ac` alone, as the audit found).
- Root cause: `main`'s `2669906` (2026-04-15) "Removed NLO_ prefix" renamed
  767 identifiers across 150 files after the branch forked.
- Replaying that rename on both the merge base and the snapshot (scripted,
  derived from `2669906`) and re-merging leaves **21 files / 53 conflict
  hunks**. All core solver files (`rk4_kernel.c`, `init_state_builder.c`,
  `state.h`, `vector_backend.*`, `fft.c`, `vk_backend_dispatch.c`,
  `operator_program_*`) then merge cleanly. What remains is CMake (7 files),
  tests (5), bindings/examples (5), `README.md`, `nlolib.c`,
  `vk_backend_resources.c`, `perf_profile.*`, and one modify/delete
  (`nlo_vk_shader_paths.h.in`, renamed on `main`).

| Feature | Commit(s) | Files | +/− | Residual conflicts after rename replay | Depends on |
|---|---|---|---|---|---|
| A Vulkan JIT | `9c73bb0` | 20 | +2444/−204 | 7 files (CMake ×2, tests ×2, README, shader-paths, `vk_backend_resources.c`) | — |
| B fused plan + profiling | `92cfa0f` (shared with CUDA stubs) | 66 | +5895/−167 | 18 files | A for the temporal fused step; tensor/Raman fused steps do not need A |
| C CUDA backend | `ad0d8ac` | 16 | +2693/−21 | 7 files (all CMake/README) | B's backend-neutral step plan and `cuda` options struct |
| D NVRTC JIT | uncommitted | 32 | +999/−132 (+825 new `.cu`) | 11 files | A (lowering/DAG) and C |

The features are not independent commits: `92cfa0f` mixes B with the CUDA
option plumbing that C needs, and D needs both A's lowering and C.

### Runtime dependencies

- **A adds no new runtime dependency.** `main` already links glslang 12.3.1
  statically into `nlolib.dll` for VkFFT's runtime GLSL→SPIR-V compilation
  (`cmake/NLODependencies.cmake`: "runtime GLSL->SPIR-V for VkFFT, static").
  The JIT calls the same glslang C interface. The audit's statement that
  glslang is "build time only" on `main` is not accurate.
  DLL sizes: `main` 4.83 MB (imports `vulkan-1.dll` only); snapshot Vulkan
  build 7.35 MB (glslang from the Vulkan SDK static libs, plus an
  `libopenblas.dll` import that `main` has since removed), so the snapshot
  size delta is mostly the different glslang provider and older deps, not A.
- **C and D add heavy, hard-linked dependencies.** The CUDA build imports
  `cufft64_12.dll` (284 MB) and the NVRTC build also `nvrtc64_130_0.dll`
  (91 MB) + `nvrtc-builtins64_130.dll` (4.5 MB) in the DLL import table, so
  `nlolib.dll` fails to load on any machine without them. The release wheel
  (`tools/release/build_wheel_windows.ps1` on the PyPI release branch) builds
  Vulkan + VkFFT and bundles DLLs with delvewheel; bundling cuFFT alone is
  ~3× PyPI's default 100 MB file limit. C/D could only ship as a separate
  variant (for example a `-cu13` wheel depending on NVIDIA's PyPI runtime
  wheels) or with delay-loaded CUDA DLLs. Same for the MATLAB toolbox and
  the Julia artifact.

### Maintenance surface

- A touches no binding (internal, transparent). B/C add a `cuda` sub-struct to
  the execution options, which changes the struct layout in Python
  (`_binding.py`), Julia (`types.jl`, `NLOLib.jl`) and MATLAB
  (`nlolib_matlab.h`): +169/−45 lines over 8 binding files; D adds CUDA DLL
  discovery to `_binding.py`.
- A second GPU backend doubles the backend test matrix; CI has no NVIDIA
  runner, so C/D would be untested in CI. Multi-GPU/NCCL: unverified (one
  NVIDIA device).
- Direction: `TODO.md` still lists "chain together operations into single GPU
  kernels" (Low Priority), which A/B implement. A's lowering to a DAG plus
  GLSL emission is also the natural host for a future finite-difference
  kernel; mode coupling needs slice-wise kernel application, which the fused
  plan does not provide.

## Resume checklist (follow-up)

1. Rebuild nothing: `dist-main/` (this worktree) and `dist-vk|cuda|nvrtc/` (snapshot worktree) hold the variant DLLs. Put CUDA 13.0 `bin/x64` on `PATH`; use `C:/Users/Wenzel/miniconda3/envs/nlo_dev/python.exe`.
2. Finish the fixed-step matrix for the 2-D/3-D cases (all variants): `bash benchmarks/assessment/run_matrix.sh benchmarks/results/assessment/fixed2/rows.jsonl benchmarks/results/assessment/fixed2/fields 3 10 fixed "tensor3d:64 grin2p1:64 tensor3d:128 grin2p1:128" main main_ts main_cpu snap_interp snap_fused snap_jit snap_jit_fused cuda nvrtc_jit` (grin2p1 128 is ~7 s per run on main plus Python list conversion; budget ~1 h).
3. Adaptive correctness (step history): same script, mode `adaptive`, warm-up 0, runs 1, cases `kerr1d:4096 raman1d:4096 expr1d:16384 tensor3d:64 grin2p1:64`; then `compare_fields.py <fields> main`.
4. Memory headroom for C: `tensor3d:192 grin2p1:192 tensor3d:256 grin2p1:256` with `main snap_jit_fused cuda nvrtc_jit` (1 warm-up, 3 runs); record what fails to allocate.
5. Rerun the noisy 1-D cells (IQR > 20 % of median) before quoting them.
6. Decide the correctness reading for kerr1d (strict 1e-10 vs "within main's CPU/GPU disagreement"), then write Phase 4: decision table plus ordered porting plan (expected: B profiling → A → B fused plan; C/D archived or optional).
7. Cleanup when done: snapshot worktree `.claude/worktrees/assess-tasks-snapshot` (branch `assess/tasks-snapshot`, never push). Leave `nlolib.worktrees/tasks` alone until the decision is final.
