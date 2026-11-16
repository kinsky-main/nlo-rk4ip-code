#!/usr/bin/env bash
# Drive assess_bench.py over cases x variants, interleaving variants within each
# case so slow thermal drift on the laptop GPU does not bias one variant.
# Usage:
#   run_matrix.sh <out.jsonl> <field_dir> <warmup> <runs> <mode> "<case:size> ..." [variant ...]
# main_ts (timestamp phase breakdown) always uses 1 warm-up and 3 runs: it is a
# breakdown, not a decision metric, and timestamps perturb wall time.
set -u
OUT="$1"; FIELDS="$2"; WARM="$3"; RUNS="$4"; MODE="$5"; CASES="$6"; shift 6
VARIANTS="${*:-main main_ts snap_interp snap_fused snap_jit snap_jit_fused cuda nvrtc_jit}"

HERE="$(cd "$(dirname "$0")" && pwd)"
MAIN="$(cd "$HERE/../.." && pwd)"
SNAP="${SNAP:-$(cd "$MAIN/../assess-tasks-snapshot" && pwd)}"
PY="${PY:-/c/Users/Wenzel/miniconda3/envs/nlo_dev/python.exe}"

one() {
  local variant="$1" tree="$2" lib="$3" backend="$4" c="$5" s="$6" w="$7" n="$8"; shift 8
  env "$@" "$PY" "$HERE/assess_bench.py" --tree "$tree" --lib "$lib" --variant "$variant" \
    --backend "$backend" --case "$c" --size "$s" --mode "$MODE" --warmup "$w" --runs "$n" \
    --out "$OUT" --field-dir "$FIELDS" 2>>"$OUT.stderr" | grep '^{' || echo "FAILED $variant $c $s" | tee -a "$OUT.failed"
}

for cs in $CASES; do
  c="${cs%%:*}"; s="${cs##*:}"
  for v in $VARIANTS; do
    case "$v" in
      main)           one main "$MAIN" "$MAIN/dist-main/nlolib.dll" vulkan "$c" "$s" "$WARM" "$RUNS" NLO_X=1 ;;
      main_cpu)       one main_cpu "$MAIN" "$MAIN/dist-main/nlolib.dll" cpu "$c" "$s" 0 1 NLO_X=1 ;;
      main_ts)        one main_ts "$MAIN" "$MAIN/dist-main/nlolib.dll" vulkan "$c" "$s" 1 3 NLO_GPU_TIMESTAMPS=1 ;;
      snap_interp)    one snap_interp "$SNAP" "$SNAP/dist-vk/nlolib.dll" vulkan "$c" "$s" "$WARM" "$RUNS" NLO_OPERATOR_JIT=0 NLO_DISABLE_VK_FUSED_STEP=1 ;;
      snap_fused)     one snap_fused "$SNAP" "$SNAP/dist-vk/nlolib.dll" vulkan "$c" "$s" "$WARM" "$RUNS" NLO_OPERATOR_JIT=0 ;;
      snap_jit)       one snap_jit "$SNAP" "$SNAP/dist-vk/nlolib.dll" vulkan "$c" "$s" "$WARM" "$RUNS" NLO_OPERATOR_JIT=1 NLO_DISABLE_VK_FUSED_STEP=1 ;;
      snap_jit_fused) one snap_jit_fused "$SNAP" "$SNAP/dist-vk/nlolib.dll" vulkan "$c" "$s" "$WARM" "$RUNS" NLO_OPERATOR_JIT=1 ;;
      cuda)           one cuda "$SNAP" "$SNAP/dist-cuda/nlolib.dll" cuda "$c" "$s" "$WARM" "$RUNS" NLO_OPERATOR_JIT=0 ;;
      cuda_nofuse)    one cuda_nofuse "$SNAP" "$SNAP/dist-cuda/nlolib.dll" cuda "$c" "$s" "$WARM" "$RUNS" NLO_OPERATOR_JIT=0 NLO_DISABLE_CUDA_FUSED_STEP=1 ;;
      nvrtc_jit)      one nvrtc_jit "$SNAP" "$SNAP/dist-nvrtc/nlolib.dll" cuda "$c" "$s" "$WARM" "$RUNS" NLO_OPERATOR_JIT=1 ;;
      nvrtc_off)      one nvrtc_off "$SNAP" "$SNAP/dist-nvrtc/nlolib.dll" cuda "$c" "$s" "$WARM" "$RUNS" NLO_OPERATOR_JIT=0 ;;
      *) echo "unknown variant $v" >&2 ;;
    esac
  done
done
