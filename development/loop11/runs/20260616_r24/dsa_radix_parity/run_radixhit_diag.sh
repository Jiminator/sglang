#!/usr/bin/env bash
# Diagnostic — does the DSA-native default radix cache EVER produce a cache hit?
# Single boot (radix ON, EAGER, enable-cache-report). radixhit_diag.py sends
# short/medium/the exact 6090-tok ExpB prompt, each TWICE, reports cached_tokens.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R24="$HERE/../tablefree_radix"
source "$R24/r24_env.sh"
REPO=/sgl-workspace/sglang; cd "$REPO"
LOG="$HERE/radixhit_diag.log"; exec > >(tee -a "$LOG") 2>&1
echo "=== radix-hit diag start $(date -u +%H:%M:%SZ) ==="
GLM=/cluster-storage/models/models--zai-org--GLM-5.1-FP8/snapshots/f396cf805182f4ca10fa675e1a99815b3ca384db
slog="$HERE/radixhit_serve.log"
teardown
args=(--model-path "$GLM" --tp-size 8 --kv-cache-dtype fp8_e4m3 --page-size 64
  --dsa-prefill-backend flashmla_kv --dsa-decode-backend flashmla_kv
  --disable-overlap-schedule --disable-piecewise-cuda-graph --disable-cuda-graph
  --random-seed 20260607 --trust-remote-code --host 127.0.0.1 --port "$PORT"
  --mem-fraction-static 0.8 --max-running-requests 64 --cuda-graph-max-bs 64
  --enable-cache-report)
python -m sglang.launch_server "${args[@]}" > "$slog" 2>&1 &
rc=0; ready_wait "$slog" || rc=$?
if [[ "$rc" != "0" ]]; then echo "boot FAIL rc=$rc"; tail -40 "$slog"; teardown; gpu_idle_wait; exit 1; fi
echo ">>> ready. $(grep -aoE 'disable_radix_cache=(True|False)' "$slog" | head -1)"
python "$HERE/radixhit_diag.py"
echo "--- scheduler cached-token lines (real prefills) ---"
grep -aE "Prefill batch, #new-seq: 1, #new-token: [0-9]+, #cached-token" "$slog" | tail -10
teardown; gpu_idle_wait
echo "=== radix-hit diag done $(date -u +%H:%M:%SZ) ==="
