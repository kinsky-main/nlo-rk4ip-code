# GPU JIT / CUDA branch assessment

Status: in progress. Plan: `docs/tasks-branch-assessment-plan.md` on
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
