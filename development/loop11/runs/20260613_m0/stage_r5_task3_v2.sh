#!/usr/bin/env bash
# Loop 11 R5 task3 validation (v2): robust readiness (poll the serve.log "fired up" marker +
# a bounded-curl /health), so a flaky wait_ready can't mis-report a healthy boot. Validates the
# DS indexer-cache gate end-to-end (committed feature; no probe hooks):
#   AC-1.1  DS @0.7 -> max_total_num_tokens 174848 (was 142208 ungated) = +23% tokens; DS @0.8 -> 504640.
#   AC-7    DSA @0.8 (DS off) -> 410560 unchanged; radix-ON DSA serving smoke coherent.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /sgl-workspace/sglang/development/profiling/runs/20260612/_env.sh
cd /sgl-workspace/sglang
LOG="$HERE/stage_r5_task3_v2.log"; exec > >(tee "$LOG") 2>&1
unset SGLANG_DS_PROBE_TABLE_TOKENS SGLANG_DS_PROBE_SKIP_INDEXER || true
echo "=== R5 task3 v2 validation start $(date -u +%Y-%m-%dT%H:%M:%SZ) HEAD=$(git rev-parse --short HEAD) ==="
git diff --quiet -- python/ && echo "production tree = committed (no uncommitted python)" || echo "NOTE: uncommitted python diff present"

ready_wait() {  # $1 = serve.log ; success when 'fired up' appears AND /health 200; fail fast on crash
  local slog="$1"
  for i in $(seq 1 80); do  # ~13 min budget
    if grep -aq "SIGQUIT received\|Scheduler hit an exception\|CUDA out of memory\|Capture cuda graph failed" "$slog" 2>/dev/null; then
      return 1
    fi
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
probe() {  # name  case  extra-args...
  local name="$1"; shift; local case="$1"; shift
  echo "=== $name ($case $*) $(date -u +%H:%M:%SZ) ==="
  teardown
  local args
  if [[ "$case" == "dsa08_radixon" ]]; then
    build_server_args dsa08
    args=(); for a in "${SERVER_ARGS[@]}"; do [[ "$a" == "--disable-radix-cache" ]] && continue; args+=("$a"); done
  else
    build_server_args "$case"; args=("${SERVER_ARGS[@]}" "$@")
  fi
  local slog="$HERE/r5v2_${name}_serve.log"
  python -m sglang.launch_server "${args[@]}" > "$slog" 2>&1 &
  local cap=0 fired smk note
  if ready_wait "$slog"; then
    cap=$(grep -aoE "max_total_num_tokens=[0-9]+" "$slog" | head -1 | grep -oE "[0-9]+")
    smk=$(smoke)
    note="OK"
  else
    note="BOOT_FAIL:$(grep -aiE 'SIGQUIT|exception|out of memory|capture cuda graph failed' "$slog" | head -1 | cut -c1-50)"
    cap=$(grep -aoE "max_total_num_tokens=[0-9]+" "$slog" | head -1 | grep -oE "[0-9]+")
    smk="-"
  fi
  {
    echo "probe=$name case=$case extra=$* status=$note"
    echo "max_total_num_tokens=${cap:-NA} bs_cap=$(( ${cap:-0} / 4608 ))"
    echo "smoke=$smk"
    grep -aE "TP0\] (KV Cache is allocated|Memory pool end|Capture cuda graph end|max_total_num_tokens)" "$slog" | head -4
    grep -a "disable_radix_cache=" "$slog" | head -1 || true
  } > "$HERE/r5v2_${name}_evidence.txt"
  echo ">>> $name status=$note cap=${cap:-NA} bs=$(( ${cap:-0} / 4608 )) smoke=$smk"
  teardown
}

probe ac1_ds_gate_070 ds07                              # expect 174848 (was 142208 ungated)
probe ac1_ds_gate_080 ds07 --mem-fraction-static 0.8    # expect 504640 / bs109
probe ac7_dsa_080     dsa08                             # expect 410560 (DSA unchanged)
probe ac7_dsa_080_radixon dsa08_radixon                 # radix-ON DSA serving smoke
echo "=== R5 task3 v2 validation done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
cat "$HERE"/r5v2_*_evidence.txt