#!/usr/bin/env bash
# Determinism noise-floor control: one radix-OFF DSA-native server, each measured
# prompt issued TWICE, check greedy-output identity. Establishes whether the
# fresh-vs-reuse divergence is a real reuse effect or just nondeterminism.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../../20260616_r25/radix_authorization/r25_env.sh"
cd "$REPO"
OUTDIR="$HERE"; mkdir -p "$OUTDIR/logs"
NUM="${1:-48}"
slog="$OUTDIR/logs/serve_determinism.log"
teardown
echo "=== boot DSA-native (determinism control, radix-OFF) eager $(date -u +%H:%M:%SZ) ==="
args=("${DSA_ARGS[@]}" --disable-cuda-graph --disable-radix-cache)
python -m sglang.launch_server "${args[@]}" > "$slog" 2>&1 &
rc=0; ready_wait "$slog" || rc=$?
if [[ "$rc" != "0" ]]; then echo "!! boot FAIL rc=$rc"; tail -n 40 "$slog"; teardown; gpu_idle_wait; exit 10; fi
echo ">>> ready. smoke=$(smoke)"; grep -aE "disable_radix_cache=(True|False)" "$slog" | head -1
python "$HERE/determinism_control.py" --length 4096 --num "$NUM" --max-new-tokens 24 \
  --seed-base 1000 --out "$OUTDIR/determinism_control.jsonl" 2>&1 | tee "$OUTDIR/logs/determinism.log"
teardown; gpu_idle_wait
echo "=== final nvidia-smi ==="; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo "=== determinism control done $(date -u +%H:%M:%SZ) ==="
