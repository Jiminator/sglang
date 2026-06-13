#!/usr/bin/env bash
# Loop 11 R2 AC-7 regression: the bounded-selector-width feature touches the shared
# cuda_graph_runner.py, so prove (a) DSA-native is byte-unchanged and (b) the DS default
# path (full_fallback) still boots+captures+serves like the frozen p01 anchor. Run on the
# FEATURE-ONLY tree (probe_hacks reverted). No probe env hooks.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /sgl-workspace/sglang/development/profiling/runs/20260612/_env.sh
cd /sgl-workspace/sglang
LOG="$HERE/stage_r2_regression.log"; exec > >(tee "$LOG") 2>&1
unset SGLANG_DS_PROBE_TABLE_TOKENS SGLANG_DS_PROBE_SKIP_INDEXER || true
echo "=== R2 AC-7 regression start $(date -u +%Y-%m-%dT%H:%M:%SZ) HEAD=$(git rev-parse --short HEAD) ==="
echo "=== production diff (feature only) ==="; git diff --stat -- python/

smoke() {
  curl -s --max-time 120 -X POST "http://127.0.0.1:${PORT}/generate" \
    -H 'Content-Type: application/json' \
    -d '{"text": "The capital of France is", "sampling_params": {"max_new_tokens": 24, "temperature": 0}}' 2>/dev/null \
  | python3 -c "import json,sys;
try:
 d=json.load(sys.stdin); t=(d.get('text') or '').strip().replace(chr(10),' ')
 print('OK:'+t[:50] if t else 'FAIL:empty')
except Exception as e: print('FAIL:'+str(e)[:50])"
}

# (a) DSA-native @0.8 (DS OFF): must boot + capture + serve unchanged.
echo "=== AC-7a: DSA-native @0.8 (DS off) $(date -u +%H:%M:%SZ) ==="
teardown
build_server_args dsa08
DSALOG="$HERE/r2_dsa_off_080_serve.log"
python -m sglang.launch_server "${SERVER_ARGS[@]}" > "$DSALOG" 2>&1 &
if wait_ready; then
  CAP=$(max_batch_from_server)
  echo ">>> DSA-off cap=$CAP capture=$(grep -ac 'Capture cuda graph end' "$DSALOG") smoke=$(smoke)"
  grep -aE "TP0\] (KV Cache is allocated|Capture cuda graph end|max_total_num_tokens)" "$DSALOG" | head -3 > "$HERE/r2_dsa_off_080_evidence.txt"
else echo "!! DSA-off boot FAILED"; tail -20 "$DSALOG"; fi
teardown

# (b) DS-on DEFAULT (full_fallback, default config) @0.7: must reproduce the frozen p01 anchor
#     (142208 tokens / bs30 / table 5.29 GB / coherent smoke) — runtime byte-compat proof.
echo "=== AC-7b: DS-on default (full_fallback) @0.7 $(date -u +%H:%M:%SZ) ==="
teardown
build_server_args ds07
DSLOG="$HERE/r2_ds_default_070_serve.log"
python -m sglang.launch_server "${SERVER_ARGS[@]}" > "$DSLOG" 2>&1 &
if wait_ready; then
  CAP=$(max_batch_from_server)
  echo ">>> DS-default cap=$CAP (expect 142208) capture=$(grep -ac 'Capture cuda graph end' "$DSLOG") smoke=$(smoke)"
  grep -aE "DS selector-width graph variants|TP0\] (KV Cache is allocated|max_total_num_tokens)|token_label_table:" "$DSLOG" | head -4 | tee "$HERE/r2_ds_default_070_evidence.txt"
else echo "!! DS-default boot FAILED"; tail -20 "$DSLOG"; fi
teardown
echo "=== R2 AC-7 regression done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
