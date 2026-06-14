#!/usr/bin/env bash
# Loop 11 M2 — A7 AC-7 DSA-default regression. The shipped DSA-native default
# (DS OFF) must be un-regressed by the table_free (DS-only) changes.
#  (a) DS-off smoke           : DSA-native @0.8 (dsa08) coherent output.
#  (b) frozen Case-2 re-valid : DSA-native @0.7 (dsa07) — COMPARE max_total_num_tokens to frozen 142208.
#  (c) radix-ON DSA smoke     : DSA-native @0.8 with radix cache ON, coherent.
# One TP=8 server at a time; foreground per phase; teardown before each boot.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /sgl-workspace/sglang/development/profiling/runs/20260612/_env.sh
cd /sgl-workspace/sglang
LOG="$HERE/a7_stage.log"; exec > >(tee "$LOG") 2>&1
echo "=== A7 DSA-default regression start $(date -u +%Y-%m-%dT%H:%M:%SZ) HEAD=$(git rev-parse --short HEAD) ==="
FROZEN_CASE2=142208

ready_wait() {
  local slog="$1"
  for i in $(seq 1 80); do
    if grep -aqE "SIGQUIT received|Scheduler hit an exception|CUDA out of memory|Capture cuda graph failed|Not enough host memory|ValueError|Traceback" "$slog" 2>/dev/null; then return 2; fi
    if grep -aq "The server is fired up" "$slog" 2>/dev/null; then
      curl -sf --max-time 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && { echo "ready ~$((i*10))s"; return 0; }
    fi
    sleep 10
  done
  return 1
}
smoke() {
  curl -s --max-time 120 -X POST "http://127.0.0.1:${PORT}/generate" -H 'Content-Type: application/json' \
    -d '{"text": "The capital of France is", "sampling_params": {"max_new_tokens": 24, "temperature": 0}}' 2>/dev/null \
  | python3 -c "import json,sys
try:
 d=json.load(sys.stdin); t=(d.get('text') or '').strip().replace(chr(10),' ')
 print('OK:'+t[:60] if t else 'FAIL:empty')
except Exception as e: print('FAIL:'+str(e)[:60])"
}

# --- (a) DS-off smoke: DSA-native default @0.8 (dsa08) ---
teardown
echo "=== (a) DS-off smoke: DSA-native default dsa08 @0.8 $(date -u +%H:%M:%SZ) ==="
build_server_args dsa08
slog="$HERE/a7_dsa08_default_serve.log"
python -m sglang.launch_server "${SERVER_ARGS[@]}" > "$slog" 2>&1 &
if ready_wait "$slog"; then
  CAP_A=$(grep -aoE "max_total_num_tokens=[0-9]+" "$slog" | head -1 | grep -oE "[0-9]+"); SMK_A=$(smoke); ST_A=OK
else ST_A=BOOT_FAIL; CAP_A=NA; SMK_A=-; fi
echo ">>> (a) dsa08 status=$ST_A cap=$CAP_A smoke=$SMK_A"
teardown

# --- (b) frozen Case-2 re-validation: DSA-native @0.7 (dsa07) ---
echo "=== (b) frozen Case-2 re-validation: dsa07 @0.7 (frozen=$FROZEN_CASE2) $(date -u +%H:%M:%SZ) ==="
build_server_args dsa07
slog="$HERE/a7_case2_dsa07_serve.log"
python -m sglang.launch_server "${SERVER_ARGS[@]}" > "$slog" 2>&1 &
if ready_wait "$slog"; then
  CAP_B=$(grep -aoE "max_total_num_tokens=[0-9]+" "$slog" | head -1 | grep -oE "[0-9]+"); SMK_B=$(smoke); ST_B=OK
else ST_B=BOOT_FAIL; CAP_B=NA; SMK_B=-; fi
MATCH_B=$([[ "$CAP_B" == "$FROZEN_CASE2" ]] && echo "MATCH" || echo "MISMATCH(got $CAP_B vs frozen $FROZEN_CASE2)")
echo ">>> (b) case2 dsa07 status=$ST_B cap=$CAP_B vs frozen=$FROZEN_CASE2 -> $MATCH_B smoke=$SMK_B"
teardown

# --- (c) radix-ON DSA serving smoke: dsa08 with radix cache ON ---
echo "=== (c) radix-ON DSA smoke: dsa08 @0.8 radix-ON $(date -u +%H:%M:%SZ) ==="
build_server_args dsa08
ARGS=(); for a in "${SERVER_ARGS[@]}"; do [[ "$a" == "--disable-radix-cache" ]] && continue; ARGS+=("$a"); done
slog="$HERE/a7_dsa08_radixon_serve.log"
python -m sglang.launch_server "${ARGS[@]}" > "$slog" 2>&1 &
if ready_wait "$slog"; then
  CAP_C=$(grep -aoE "max_total_num_tokens=[0-9]+" "$slog" | head -1 | grep -oE "[0-9]+"); SMK_C=$(smoke)
  RADIX_C=$(curl -s --max-time 8 "http://127.0.0.1:${PORT}/get_server_info" | python3 -c "import json,sys;print(json.load(sys.stdin).get('disable_radix_cache','?'))")
  ST_C=OK
else ST_C=BOOT_FAIL; CAP_C=NA; SMK_C=-; RADIX_C=?; fi
echo ">>> (c) dsa08 radix-on status=$ST_C cap=$CAP_C disable_radix_cache=$RADIX_C (expect False) smoke=$SMK_C"
teardown

{
  echo "=== A7 AC-7 DSA-default regression ==="
  echo "(a) DS-off DSA default dsa08 @0.8: status=$ST_A max_total_num_tokens=$CAP_A smoke=$SMK_A"
  echo "(b) frozen Case-2 dsa07 @0.7    : status=$ST_B max_total_num_tokens=$CAP_B vs frozen=$FROZEN_CASE2 -> $MATCH_B smoke=$SMK_B"
  echo "(c) radix-ON DSA dsa08 @0.8     : status=$ST_C max_total_num_tokens=$CAP_C disable_radix_cache=$RADIX_C(expect False) smoke=$SMK_C"
} > "$HERE/a7_dsa_default_evidence.txt"
cat "$HERE/a7_dsa_default_evidence.txt"
echo "=== A7 done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
