# GPU JIT / CUDA branch assessment

Status: **complete 2026-11-15.** The assessment was paused on 2026-11-14 and
resumed when the user chose to take forward A (Vulkan JIT) and B (fused step
plan).

Verdict:

| Feature | Verdict |
|---|---|
| A, Vulkan JIT | **port** |
| B, fused RK algebra | **port second** (judgment call) |
| B's profiling half | do not port |
| C, CUDA backend | **archive** |
| D, NVRTC JIT | **archive** |

The ordered porting plan follows Phase 4.

Plan: `docs/tasks-branch-assessment-plan.md` on `claude/worktree-audit-9d3049`.

Subject: `optimisation/runtim-vulkan-compilation` @ `ad0d8ac` plus the saved
uncommitted NVRTC work. It is snapshotted as the local branch
`assess/tasks-snapshot`, which was never pushed.

Baseline: `main` @ `a23ccff`.

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

## Phase 1/2: measurements

Harness: `benchmarks/assessment/assess_bench.py`. The same Python script drives
both trees; `--tree` and `--lib` pick the build. `run_matrix.sh` runs it with
the variants interleaved inside each case, so thermal drift on the laptop GPU
does not favour one variant.

Metric: the library-internal `solve_rk4` wall time; init is timed separately.
Fixed-step runs take 200 steps for the 1-D cases. Each cell is 3 warm-ups plus
10 timed runs, reported as median (IQR).

Raw rows and correctness CSVs are in `docs/assessment-data/`:
`fixed-partial-2026-11-14.jsonl` (first 1-D pass, all variants),
`rerun1d-2026-11-15.jsonl`, `fixed2-2026-11-15.jsonl`,
`adaptive-2026-11-15.jsonl`, `headroom-2026-11-15.jsonl` and
`memprobe-2026-11-15.*`.

Variants:

| Variant | Build | JIT (A) | Fused plan (B) |
|---|---|---|---|
| `main` | main, interpreter | — | — |
| `snap_interp` | snapshot | off | off |
| `snap_fused` | snapshot | off | on (B alone) |
| `snap_jit` | snapshot | on (A alone) | off |
| `snap_jit_fused` | snapshot | on | on (A+B, the branch default) |
| `cuda` | snapshot | off (C) | — |
| `nvrtc_jit` | snapshot | on (C+D) | — |

`main_ts` is `main` with timestamp queries on and is used only for the phase
breakdown. `main_cpu` is `main` on the FFTW CPU backend.

After the 2026-11-15 user decision to take forward A and B, only those two were
measured on the 2-D/3-D, adaptive and headroom cases. C and D have only the
1-D first pass.

### Solve time, fixed step: median ms (IQR) and speedup vs main

1-D cases. The Vulkan columns come from the 2026-11-15 rerun, where every IQR
is ≤ 32 ms and most are ≤ 3 ms. The `cuda` and `nvrtc_jit` columns come from
the first pass. kerr1d 1024 was not rerun.

| Case | main | snap_interp | snap_fused (B) | snap_jit (A) | snap_jit_fused (A+B) | cuda (C) | nvrtc_jit (C+D) |
|---|---|---|---|---|---|---|---|
| kerr1d 1024 (1st pass) | 122 (19) | 120 ×1.01 | 121 ×1.00 | 90 ×1.36 | 69 ×1.76 | 137 ×0.88 | 69 ×1.76 |
| kerr1d 4096 | 142 (1) | 142 ×1.00 | 142 ×1.00 | 111 (32) ×1.28 | 102 ×1.39 | 153 ×1.02¹ | 105 ×1.49¹ |
| kerr1d 16384 | 112 (2) | 146 ×0.77 | 116 (19) ×0.96 | 101 (10) ×1.11 | 96 ×1.17 | 155 ×0.80¹ | 98 ×1.27¹ |
| kerr1d 65536 | 172 (1) | 173 ×0.99 | 173 ×0.99 | 130 ×1.32 | 116 (9) ×1.48 | 190 ×1.43¹ | 132 ×2.06¹ |
| raman1d 4096 | 239 (2) | 228 ×1.05 | 219 ×1.09 | 216 ×1.11 | 207 ×1.16 | 228 ×1.13¹ | 207 ×1.25¹ |
| raman1d 16384 | 172 (2) | 161 ×1.07 | 154 ×1.12 | 152 ×1.13 | 143 (6) ×1.21 | 221 ×0.83¹ | 196 ×0.94¹ |
| expr1d 16384 | 168 (1) | 168 ×1.00 | 167 ×1.00 | 101 (7) ×1.66 | 97 (5) ×1.73 | 242 ×0.89¹ | 104 ×2.06¹ |

¹ Speedup against the first-pass `main`, whose IQRs were up to 169 ms; read
these columns as indicative only.

2-D/3-D cases have nxy² × 512 points. tensor3d runs 10 steps and grin2p1
runs 20.

| Case | main | snap_interp | snap_fused (B) | snap_jit (A) | snap_jit_fused (A+B) |
|---|---|---|---|---|---|
| tensor3d 64 | 490 (12) | 517 ×0.95 | 503 ×0.97 | 424 ×1.16 | 381 ×1.29 |
| tensor3d 128 | 1980 (43) | 1974 ×1.00 | 1930 ×1.03 | 1598 ×1.24 | 1423 ×1.39 |
| grin2p1 64 | 1446 (45) | 1481 ×0.98 | 1477 ×0.98 | 911 ×1.59 | 851 ×1.70 |
| grin2p1 128 | 5860 (90) | 5929 ×0.99 | 5649 ×1.04 | 3400 ×1.72 | 3086 ×1.90 |

Sanity check: `snap_interp` matches `main` within noise on every case except
kerr1d 16384, where it is 0.77×. The snapshot's own baseline is therefore not
faster than main, and the gains above come from A and B.

What B adds on top of A (`snap_jit` → `snap_jit_fused`): 4–12 %. The largest
gains are on the biggest grids: tensor3d 128 ×1.12, kerr1d 65536 ×1.12, and
grin2p1 128 ×1.10.

### Init time

JIT adds about 230–330 ms to every `propagate` call. On the rerun, init is
≈ 430 ms with JIT and ≈ 205 ms without; 2-D/3-D init is 580–610 ms against
270–385 ms. The JIT entry cache lives on the backend, so each `propagate`
compiles again.

On small 1-D problems this outweighs the solve gain. kerr1d 4096
init+solve is 301 ms on `main` and 535 ms on `snap_jit_fused`; grin2p1 128 is
6188 ms against 3665 ms.

The pre-registered metric excludes init, but the port must remove this cost;
see the A conditions below.

Warm CUDA init is ~20 ms and warm NVRTC ~60 ms, because the driver caches
compiled code.

### GPU phase breakdown on main (Gate 1)

Vulkan timestamps, as % of solve wall time.

| Case | FFT | linear | nonlinear | RK algebra | GPU idle / host | f_op |
|---|---|---|---|---|---|---|
| kerr1d 1024 | 24 % | 12 % | 14 % | 8 % | 43 % | 33 % |
| kerr1d 4096 | 52 % | 9 % | 9 % | 6 % | 24 % | 24 % |
| kerr1d 16384 | 44 % | 16 % | 11 % | 9 % | 21 % | 36 % |
| kerr1d 65536 | 37 % | 18 % | 13 % | 11 % | 21 % | 42 % |
| raman1d 4096 | 63 % | 5 % | 9 % | 3 % | 20 % | 18 % |
| raman1d 16384 | 44 % | 7 % | 12 % | 5 % | 32 % | 24 % |
| expr1d 16384 | 23 % | 8 % | 28 % | 5 % | 36 % | 41 % |
| tensor3d 64 | 28 % | 19 % | 10 % | 14 % | 29 % | 43 % |
| tensor3d 128 | 28 % | 19 % | 10 % | 15 % | 27 % | 45 % |
| grin2p1 64 | 16 % | 12 % | 51 % | 10 % | 10 % | 74 % |
| grin2p1 128 | 18 % | 12 % | 51 % | 10 % | 9 % | 73 % |

**Gate 1 does not trigger.** f_op is between 18 % and 74 %, always above the
15 % threshold. It is highest on GRIN, where the interpreted saturable
nonlinearity alone takes half the solve time. That is also the case where A
gains most (×1.7). "FFT" includes the copies and the inverse scaling around
each VkFFT call.

### Correctness, fixed step

Relative L2 of the final field against `main`.

| Case | snap_interp / snap_fused | snap_jit | snap_jit_fused | cuda / nvrtc_jit | main_cpu (FFTW) |
|---|---|---|---|---|---|
| raman1d 4096 / 16384 | ≤ 1e-14 | ≤ 9e-15 | ≤ 1e-14 | 6e-13 / 7e-14 | 8e-14 / 3e-13 |
| expr1d 16384 | 0 | 1e-14 | 1e-14 | 3e-11 | 3e-11 |
| tensor3d 64 / 128 | 0 | 2.6e-15 | 2.6e-15 | — | 7e-15 |
| grin2p1 64 / 128 | 0 | 3.3e-15 | 3.4e-15 | — | 1.1e-14 |
| kerr1d (all sizes) | 0 | 0 | 1e-8 – 5e-8 | 1e-6 – 2e-6 | 9e-8 – 1.2e-6 |

On every well-resolved case, every A/B variant agrees with `main` to
≤ 1e-14, inside the 1e-10 bar. A alone is bit-identical to `main` on kerr1d.

kerr1d uses γ = 80, which gives 16 rad of nonlinear phase. At the benchmark
step the case is under-resolved: its discretisation error is 5e-2 against a 16×
refined run, so it amplifies round-off. B's fused affine combinations
reassociate floating-point sums, and that produces 1e-8 – 5e-8 on this case.
`main`'s own CPU and Vulkan paths differ by more, up to 1.2e-6.

**Correctness reading.** This is a deviation from the pre-registered wording,
and it is recorded here.

- The 1e-10 bar applies to the well-resolved cases, and A and B pass it there.
- On kerr1d the bar sits below the case's own round-off floor: `main` fails it
  against itself, CPU vs GPU. There, the bar is "no worse than `main`'s CPU vs
  GPU disagreement". A+B passes by an order of magnitude; C sits at about that
  level.

### Correctness, adaptive step (step-size history)

Tolerance is 1e-6, except raman1d at 2e-6. Each case has a single run. The table gives the number of accepted
steps and the relative L2 of the final field against `main`.

| Case | main | main_cpu | snap_interp | snap_fused | snap_jit | snap_jit_fused | snapshot on CPU |
|---|---|---|---|---|---|---|---|
| kerr1d 4096 | 539 | 539, 4e-7 | 610, 2e-3 | 610, 2e-3 | 604, 2e-3 | 604, 2e-3 | — |
| raman1d 4096 | 95 | 95, 4e-14 | 98, 1.6e-6 | 98, 1.6e-6 | 98, 1.6e-6 | 98, 1.6e-6 | 95, 4e-14 |
| expr1d 16384 | 87 | 87, 3e-11 | 108, 3.4e-6 | 108, 3.4e-6 | 108, 3.4e-6 | 108, 3.4e-6 | — |
| tensor3d 64 | 4 | 4, 3e-15 | 4, 0 | 4, 0 | 4, 1e-15 | 4, 1e-15 | — |
| grin2p1 64 | 7 | 7, 4e-15 | 7, 0 | 7, 0 | 7, 2e-15 | 7, 2e-15 | — |

**The snapshot's Vulkan path breaks adaptive stepping, and the cause is
not A or B's kernels.** The evidence:

- The step history first diverges at step 2: the snapshot doubles the step
  and then rejects several times.
- `snap_interp` (A and B off) shows the same divergence, while the snapshot on
  CPU matches `main` exactly.
- The step controller code is identical on both trees.

Root cause: in `92cfa0f` (B's commit, shared with the CUDA stubs),
`nlo_vk_simulation_phase_flush` became asynchronous. It submits, sets
`submit_pending`, and no longer waits on the fence.
`nlo_vk_dispatch_complex_weighted_rms_error` flushes and then immediately
`memcpy`s the error pair from the staging buffer. The step controller
therefore reads a stale error, which in practice is the previous attempt's.
`main`'s flush waits on the fence (`vk_backend_phase.c`).

Fixed-step runs never read a value back mid-solve, so they are unaffected.

Within the snapshot, which shares this bug across all its Vulkan variants:

- B is bit-identical to `snap_interp` in adaptive mode.
- A changes only kerr1d's history (604 vs 610 steps), by round-off on the
  under-resolved case.
- The 2-D/3-D cases take few steps and accept all of them, so they are unaffected.

Per-step cost against `snap_interp`, adaptive mode, single run:

| Variant | kerr1d 4096 | raman1d 4096 | expr1d 16384 | tensor3d 64 | grin2p1 64 |
|---|---|---|---|---|---|
| B | ×1.01 | ×1.04 | ×1.00 | ×1.01 | ×0.96 |
| A | ×1.29 | ×1.04 | ×1.78 | ×1.14 | ×1.46 |
| A+B | ×1.42 | ×1.09 | ×1.47 | ×1.22 | ×1.61 |

These per-step factors are consistent with the fixed-step factors.

### Memory headroom

Each cell is one run (0 warm-ups). Peak dedicated VRAM comes from
`nvidia-smi`, sampled every 200 ms; the figure is the peak minus the idle
reading taken before the run.

About 1.8 GiB of the 8 GiB was already held by other processes (desktop,
browser), so about 6.2 GiB was usable. The 192 and 256 rows are in
`headroom-2026-11-15.jsonl`; the probe rows, including 160, are in
`memprobe-2026-11-15.jsonl`.

| Case (nxy² × 512) | main: solve, VRAM | A+B: solve, VRAM | CUDA (C): solve, VRAM |
|---|---|---|---|
| tensor3d 192 | 4.3 s, 5.9 GiB | 3.1 s, 5.6 GiB | — |
| grin2p1 160 | 9.1 s, 5.6 GiB | 5.1 s, 4.5 GiB | crash |
| grin2p1 192 | **225 s**, at the VRAM cap | 7.1 s, 5.6 GiB | crash |
| tensor3d 256, grin2p1 256 | init fails (status 2) | init fails (status 2) | crash |

**A's compact axis symbols save 1.08 GiB on GRIN 160 (19 %).** That is
about six full-grid buffers of 200 MiB each, as expected from dropping the
wt/kx/ky/t/x/y meshes.

At GRIN 192, `main` needs more VRAM than is free. WDDM then pages its buffers
into shared system memory, and the solve runs 32× slower. A+B still fits and
scales normally: 7.1 s against 5.1 s at 160, for a 1.44× larger grid.

So on this 8 GB laptop, A turns the largest practical GRIN grid from
"pages, minutes" into "fits, seconds". This is the "enables something `main`
cannot do" clause of the port threshold, in addition to the speedup.

tensor3d has no potential grid and fewer buffers, so it fits on both at 192.

**C cannot run tensor mode at all.** The CUDA build dies inside
`nlolib_propagate` with Windows error 0xC0000006 on every tensor case
(tensor3d at 64 and 256, grin2p1 at 64, 160, 192 and 256).

This is not a memory limit: it happens at 64 as well. Before dying it allocates
up to 4.7 GiB at GRIN 192, which is still below main's demand at that size.

This was not debugged. The 1-D cases run, so the failure is in the branch's
CUDA tensor path.

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

## Phase 4: decision

Every verdict uses the pre-registered thresholds. "≥ 20 % faster" means
t ≤ 0.8 · t_main, which is a speedup of ×1.25 or more. For the kerr1d
correctness reading, see the correctness section above.

| Feature | Speed vs main (fixed step) | ≥ 20 % on | Correct | Runtime deps | Verdict |
|---|---|---|---|---|---|
| **A** Vulkan JIT | ×1.11–1.72 alone; ×1.16–1.90 with B | 5 cases alone (kerr1d 4096 and 65536, expr1d, grin2p1 64 and 128); 7 with B | yes, ≤ 1e-14 everywhere and bit-identical on kerr1d | none new (glslang is already static in `main` for VkFFT) | **Port**, under the conditions below |
| **B** fused step plan | ×0.96–1.12 alone; +4–12 % on top of A | 0 alone | yes on well-resolved cases; kerr1d 5e-8 is inside `main`'s 1.2e-6 CPU vs GPU spread | none | **Port, second, after A** (judgment call, 10–20 % band; reasons below) |
| B profiling half | — | — | — | none | **Do not port**; `main`'s phase-tagged timestamps and host timers (`b1be44c`) cover the need at a fraction of the size |
| **C** CUDA backend | ×0.80–1.43 (first pass); slower than A+B on Vulkan on every 1-D case (level on raman1d 4096) | 1 case (kerr1d 65536, noisy baseline) | 1-D: about `main`'s CPU vs GPU spread on kerr1d. **Tensor mode: crashes** (0xC0000006) at every size | cuFFT 284 MB, hard-linked | **Archive** |
| **D** NVRTC JIT | ≈ A+B on Vulkan (×0.94–2.06) | 5 cases (first pass), but no gain over A+B | as C | C's, plus NVRTC 95 MB | **Archive** with C |

Reasons:

- **A clears every bar**, and Gate 1 shows why it works. GRIN spends half its
  solve time in the interpreted nonlinearity, and there A gives ×1.7.

  Separately, its compact axis symbols cut tensor-mode VRAM by about 19 %. That
  keeps GRIN 192 resident on an 8 GB card, where `main` pages and is 32×
  slower: it enables a grid `main` cannot run usefully.

  Its one
  material regression is the 230–330 ms extra init per `propagate`. That
  makes small 1-D problems slower end to end, so removing it is a condition of
  the port, not an optional follow-up.
- **B is a judgment call that comes out port.** It adds 4–12 % over A,
  growing with grid size: ×1.10–1.12 on the largest 1-D, 2-D and 3-D grids.
  The fused RK kernels are small, self-contained and backend-neutral, and the
  CPU fallbacks already exist. Against that:
  - B's commit carries the async-flush race that breaks adaptive stepping (see
    above).
  - It is entangled with the CUDA options struct.
  - It changes round-off.

  Porting only the RK-algebra kernels and the step plan coefficients, not the
  whole commit, leaves the race and the CUDA struct behind. The round-off
  change stays, and it is accepted under the correctness reading above. If
  that port runs over its budget (see
  step 5), drop B: A alone delivers 70–95 % of the combined A+B time saving.
- **C is not faster than what A+B delivers on the Vulkan path already in
  release**, and it hard-links 375 MB of NVIDIA runtime. CI has no NVIDIA
  runner, and it doubles the backend test matrix.
- **C also fails correctness outright**: the CUDA build crashes on every
  tensor-mode case, from 64 up. So headroom cannot rescue it.
- **D only exists on top of C**, and equals A+B on Vulkan.
- Archiving C and D means keeping `assess/tasks-snapshot` (or a tag on it)
  and the saved patch as the record. `nlolib.worktrees/tasks` can then be
  retired; that is the owner's call, and this assessment did not touch it.

## Porting plan (A, then B)

Each step is a separate small PR against current `main` (`95894a8` at the time
of writing). Each passes targeted `ctest` cases and leaves `main` releasable.

Code moves by hand from the snapshot onto current `main`, with identifiers
renamed by `benchmarks/assessment/rename_map.py`. No merge of the branch: 48
files conflict.

None of the steps changes the C, Python, MATLAB or Julia public API.

1. **Land the assessment instrumentation** (`b1be44c`: phase-tagged Vulkan
   timestamps behind `NLO_GPU_TIMESTAMPS=1`, plus host init/solve timers) and
   the harness under `benchmarks/assessment/`.
   - Purpose: every later step is measured with the same tool and baseline.
   - Before landing: run `main`'s own `ctest` on the instrumented build. This
     was not done during the assessment.
   - Size: about 300 lines.
2. **Operator-program lowering to a value DAG**, CPU only: from
   `operator_program_compile.c` / `operator_program.h` in `9c73bb0`.
   - Content: interned values, symbol masks, `lowered_hash`, and a
     `jit_eligible` flag.
   - No GPU code and no behaviour change; the interpreter still executes.
   - Tests: a new core test that lowers every runtime-operator example
     expression and checks eligibility and hash stability.
3. **Vulkan JIT** (`vk_operator_jit.c`, `operator_program_jit.h`,
   `nlo_complex_math.glslinc`), with the interpreter as fallback.
   - Port changes against the branch:
     - Embed the GLSL include sources at build time, the way `main` embeds
       SPIR-V (`vk_shader_blobs.h`). The branch reads them at runtime from
       the build-time source path (`nlo_vk_shader_paths.h.in`, gone on
       `main`), so installed wheels and toolboxes would silently fall back to
       the interpreter.
     - Link `main`'s resolved glslang target (`cmake/ResolveGlslang.cmake`),
       not the Vulkan SDK static libraries.
     - Add a **process-wide SPIR-V / pipeline cache keyed by `lowered_hash`**,
       which removes the per-`propagate` recompile. Acceptance: warm init
       within 50 ms of `main`'s.
     - Keep an env/option switch to force the interpreter, for debugging.
   - Tests:
     - JIT vs interpreter on each example expression, fixed step,
       ≤ 1e-12 relative L2.
     - A run from a relocated install (wheel smoke test) that asserts the JIT
       path was taken.
     - Adaptive step-history equality with the interpreter on the
       well-resolved cases.
   - Acceptance: rerun this assessment's fixed2/rerun1d matrix; expect
     ×1.25–1.9 on the 2-D/3-D cases and expr1d.
4. **Compact axis symbols for tensor mode** (`state.h`
   `tensor_compact_axis_symbols`, plus accessor fallbacks).
   - When JIT is active, kernels read the 1-D wt/kx/ky/t/x/y axes, which
     drops six full-grid buffers.
   - Separate PR, because it changes memory layout assumptions.
   - Tests: `test_state_alloc` with the JIT on and off in tensor mode, and
     the headroom probe.
5. **B: fused RK-algebra kernels and the step plan**, ported without the rest
   of `92cfa0f`.
   - Port:
     - the `complex_affine_comb2/3/4_real`, `axpy_inplace_real`,
       `embedded_error_pair_real` and `lerp` shaders, with their CPU fallbacks
       in `vector_backend.c`;
     - the `gpu_step_plan` coefficients;
     - the fused RK4 and ERK4(3) paths in `rk4_kernel.c`.
   - Leave out:
     - the `cuda` options sub-struct and any CUDA or multi-GPU fields;
     - the asynchronous `simulation_phase_flush`. If it is wanted later, every
       host readback must wait on the fence first.
     - the branch's perf-profile expansion.
   - Tests:
     - fused vs unfused ≤ 1e-12 on the well-resolved cases;
     - kerr1d within `main`'s CPU vs GPU spread;
     - adaptive step-history equality with the unfused path on every case.
       This test would have caught the async-flush race.
   - Budget: if the port grows past about 800 lines, or needs public-API
     changes, stop and keep A alone.

Order rationale:

- Steps 2–3 deliver 70–95 % of the measured A+B saving, and each is independently
  revertible.
- Step 4 is the memory win and depends on 3.
- Step 5 depends on 3 for the temporal fused path. Its tensor and Raman fused
  paths do not need 3, but the gain there is small.
- Then update the `TODO.md` item "chain together operations into single GPU
  kernels" and the README if any env switch becomes public.

## Not covered

- C and D were not measured on the 2-D/3-D, adaptive or headroom cases after
  the scope narrowed to A and B. Their verdicts rest on the 1-D first pass and
  the dependency cost.
- Multi-GPU / NCCL was not tested: there is only one NVIDIA device.
- The Linux and macOS builds of A were not tried. glslang is cross-platform
  and already used by `main` on every platform.
- `main`'s own `ctest` suite was not run on the instrumented build. Run it in
  porting step 1.
- Adaptive runs are a single sample each. Their timings are indicative, and
  their step histories are deterministic.

## State on disk

- This branch: the doc, `benchmarks/assessment/` (harness), and
  `docs/assessment-data/` (raw rows and correctness CSVs).
  `benchmarks/results/assessment/` (fields, logs) is local only.
- Snapshot worktree `.claude/worktrees/assess-tasks-snapshot`, on branch
  `assess/tasks-snapshot` (local; never push), holding the
  `dist-vk|cuda|nvrtc/` variant DLLs. Keep it until porting step 3 has
  landed, as the source for the hand port. Then remove the worktree and keep
  the branch, or tag it, as the archive of C and D.
- `nlolib.worktrees/tasks` was not touched. Retiring it is the owner's
  decision, now that the saved patch and the snapshot branch hold its work.
