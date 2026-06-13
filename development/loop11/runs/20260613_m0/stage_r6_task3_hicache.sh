#!/usr/bin/env bash
# Loop 11 R6 task3 repair: hierarchical-cache (HiCache) DSA-indexer-sidecar state-path closure.
# AC-7 + AC-1.1 evidence for the two on-target shared surfaces touched this round:
#   phase reject : DS + --enable-hierarchical-cache  -> clean validator ValueError (fail-closed),
#                  NO NoneType crash in DSAIndexerPoolHost.init_kv_buffer (the R5 gap).
#   phase dsa    : DSA-native + --enable-hierarchical-cache -> the DSA indexer host sidecar still
#                  builds (the gate guard is skipped for a non-gated pool) + coherent serve smoke.
# The served DSA default (no HiCache) is provably untouched (DSAIndexerPoolHost is only built under
# HiCache; validate_double_sparsity early-returns for non-DS) and was 410560 @0.8 in R5.
# One TP=8 server at a time; foreground (no background task -> does not pin the loop review hook).
set -uo pipefail
PHASE="${1:?usage: stage_r6_task3_hicache.sh <reject|dsa>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /sgl-workspace/sglang/development/profiling/runs/20260612/_env.sh
cd /sgl-workspace/sglang
LOG="$HERE/stage_r6_task3_hicache_${PHASE}.log"; exec > >(tee "$LOG") 2>&1
echo "=== R6 task3 HiCache phase=$PHASE start $(date -u +%Y-%m-%dT%H:%M:%SZ) HEAD=$(git rev-parse --short HEAD) ==="
git diff --quiet -- python/ && echo "production tree = committed (no uncommitted python)" || echo "NOTE: uncommitted python diff present"

ready_wait() {  # $1 = serve.log ; ready when 'fired up' + /health 200; fail fast on crash markers
  local slog="$1"
  for i in $(seq 1 80); do  # ~13 min budget
    if grep -aqE "SIGQUIT received|Scheduler hit an exception|CUDA out of memory|Capture cuda graph failed|Not enough host memory" "$slog" 2>/dev/null; then
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

# HiCache requires the radix cache, so drop --disable-radix-cache and add --enable-hierarchical-cache.
hicache_args() {  # $1 = base case (ds07|dsa08)
  build_server_args "$1"
  HC_ARGS=(); for a in "${SERVER_ARGS[@]}"; do [[ "$a" == "--disable-radix-cache" ]] && continue; HC_ARGS+=("$a"); done
  HC_ARGS+=(--enable-hierarchical-cache)
}

teardown

if [[ "$PHASE" == "reject" ]]; then
  echo "=== ds07 + --enable-hierarchical-cache : expect fail-closed validator error $(date -u +%H:%M:%SZ) ==="
  hicache_args ds07
  slog="$HERE/r6_ds_hicache_reject_serve.log"
  timeout 240 python -m sglang.launch_server "${HC_ARGS[@]}" > "$slog" 2>&1
  rc=$?
  echo "ds+hicache launch exit=$rc"
  has_err=$(grep -aqiE "does not support --enable-hierarchical-cache" "$slog" && echo yes || echo no)
  crash=$(grep -aqE "object is not iterable|init_kv_buffer" "$slog" && echo CRASH_FOUND || echo none)
  {
    echo "probe=ds_hicache_reject exit=$rc"
    echo "fail_closed_validator_error_present=$has_err"
    echo "nonetype_crash_in_init_kv_buffer=$crash"
    grep -aiE "ValueError: Standalone Double Sparsity does not support|hierarchical cache builds|query-signature selection" "$slog" | head -3
  } > "$HERE/r6_ds_hicache_reject_evidence.txt"
  echo ">>> reject: validator_error=$has_err crash=$crash exit=$rc"
  teardown

elif [[ "$PHASE" == "dsa" ]]; then
  echo "=== dsa08 + --enable-hierarchical-cache : expect sidecar build + smoke $(date -u +%H:%M:%SZ) ==="
  hicache_args dsa08
  slog="$HERE/r6_dsa_hicache_serve.log"
  python -m sglang.launch_server "${HC_ARGS[@]}" > "$slog" 2>&1 &
  if ready_wait "$slog"; then
    cap=$(grep -aoE "max_total_num_tokens=[0-9]+" "$slog" | head -1 | grep -oE "[0-9]+")
    smk=$(smoke); note="OK"
  else
    note="BOOT_FAIL:$(grep -aiE 'SIGQUIT|exception|out of memory|capture cuda graph failed|Not enough host memory' "$slog" | head -1 | cut -c1-70)"
    cap=$(grep -aoE "max_total_num_tokens=[0-9]+" "$slog" | head -1 | grep -oE "[0-9]+"); smk="-"
  fi
  {
    echo "probe=dsa_hicache status=$note"
    echo "max_total_num_tokens=${cap:-NA}"
    echo "smoke=$smk"
    echo "indexer_host_sidecar_built=$(grep -aqiE 'host memory for DSA indexer' "$slog" && echo yes || echo no)"
    grep -aiE "Allocating .* host memory for DSA indexer" "$slog" | head -2
    grep -aE "TP0\] (KV Cache is allocated|Capture cuda graph end|max_total_num_tokens)" "$slog" | head -3
  } > "$HERE/r6_dsa_hicache_evidence.txt"
  echo ">>> dsa_hicache status=$note cap=${cap:-NA} smoke=$smk"
  teardown
else
  echo "unknown phase $PHASE"; exit 2
fi
echo "=== R6 task3 HiCache phase=$PHASE done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
cat "$HERE"/r6_*_evidence.txt 2>/dev/null || true
