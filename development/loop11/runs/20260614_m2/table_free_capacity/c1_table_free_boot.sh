#!/usr/bin/env bash
# Loop 11 M2 — C1 table_free capacity boot (PRIMARY GOAL).
# Boots the DURABLE launcher serve_double_sparsity.sh with TABLE_FREE=1 on the
# SAME GLM-5.1-FP8 int8/0.8/rs envelope as the task4 durable_int8 baseline
# (max_total_num_tokens=344064, bs_cap=74, token_label_table=6.82 GB/rank).
# Records boot fields + CUDA-graph capture (all selector-width buckets) + smoke.
# One TP=8 server; foreground; never expandable_segments (launcher strips it).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /sgl-workspace/sglang/development/profiling/runs/20260612/_env.sh
cd /sgl-workspace/sglang
LOG="$HERE/c1_stage.log"; exec > >(tee "$LOG") 2>&1
unset SGLANG_DS_PROBE_TABLE_TOKENS SGLANG_DS_PROBE_SKIP_INDEXER || true
echo "=== C1 table_free boot start $(date -u +%Y-%m-%dT%H:%M:%SZ) HEAD=$(git rev-parse --short HEAD) ==="
git diff --quiet -- python/ && echo "python_tree=committed" || echo "NOTE: uncommitted python/ diff present"

ready_wait() {
  local slog="$1"
  for i in $(seq 1 80); do
    if grep -aqE "SIGQUIT received|Scheduler hit an exception|CUDA out of memory|Capture cuda graph failed|Not enough host memory|ValueError|AssertionError|RuntimeError|Traceback" "$slog" 2>/dev/null; then return 2; fi
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

teardown
echo "=== boot DURABLE launcher serve_double_sparsity.sh TABLE_FREE=1 (GLM int8/0.8/rs) $(date -u +%H:%M:%SZ) ==="
slog="$HERE/c1_table_free_serve.log"
TABLE_FREE=1 MODEL_PATH="$GLM" CHANNEL_MASK_PATH="$MASK" PORT="$PORT" HOST=127.0.0.1 TP_SIZE=8 \
  LOG_DIR="$HERE/c1_launcher_logs" \
  bash /sgl-workspace/sglang/development/serve_double_sparsity.sh > "$slog" 2>&1 &
rc_wait=0
ready_wait "$slog" || rc_wait=$?
if [[ "$rc_wait" != "0" ]]; then
  echo "!! C1 table_free boot FAIL (rc_wait=$rc_wait) — traceback tail:"
  tail -n 60 "$slog"
  { echo "probe=c1_table_free status=BOOT_FAIL rc_wait=$rc_wait";
    echo "--- error/traceback tail ---";
    grep -anE "Error|error|ValueError|AssertionError|RuntimeError|OOM|out of memory|Exception|Traceback|table_free|capture" "$slog" | tail -40; } > "$HERE/c1_table_free_evidence.txt"
  cat "$HERE/c1_table_free_evidence.txt"
  teardown
  exit 1
fi

CAP=$(curl -s --max-time 8 "http://127.0.0.1:${PORT}/get_server_info" | python3 -c "import json,sys;d=json.load(sys.stdin);print(d.get('max_total_num_tokens') or 0)")
SMK=$(smoke)
BS=$(( CAP / 4608 ))
BASELINE_CAP=344064; BASELINE_BS=$(( BASELINE_CAP / 4608 ))
DELTA=$(( CAP - BASELINE_CAP ))
echo ">>> table_free cap=$CAP bs=$BS  (baseline table-config cap=$BASELINE_CAP bs=$BASELINE_BS delta=$DELTA) smoke=$SMK"
{
  echo "probe=c1_table_free (serve_double_sparsity.sh TABLE_FREE=1, GLM int8/0.8/rs) status=OK"
  echo "table_free  max_total_num_tokens=$CAP  bs_cap=$BS"
  echo "baseline    max_total_num_tokens=$BASELINE_CAP  bs_cap=$BASELINE_BS  (task4 durable_int8, table-config)"
  echo "delta_tokens=$DELTA  bs_cap_ge_64=$([[ $BS -ge 64 ]] && echo YES || echo NO)"
  echo "smoke=$SMK"
  echo "effective launcher args:"; grep -aE "^    (signature_dtype|mem_fraction|max_running_reqs|cuda_graph_max_bs)|effective double_sparsity_config" "$slog" | grep -v "server_args=" | head -5
  echo "--- boot fields ---"
  grep -aE "TP0\] (KV Cache is allocated|Capture cuda graph end|max_total_num_tokens)" "$slog" | head -3
  echo "--- table allocation (expect ABSENT for table_free) ---"
  grep -aiE "token_label_table" "$slog" | head -2 || echo "(no token_label_table line — table NOT allocated, as expected)"
  echo "--- CUDA-graph capture / selector-width buckets ---"
  grep -aiE "capture|bucket|selector.*width|width.*bucket" "$slog" | tail -20
  echo "--- any capture failure ---"
  grep -aiE "capture.*fail|fail.*capture|capture cuda graph failed" "$slog" | head -5 || echo "(no capture failure)"
} > "$HERE/c1_table_free_evidence.txt"
cat "$HERE/c1_table_free_evidence.txt"
echo "=== C1 boot done; LEAVING SERVER UP for C2 (do not teardown here) $(date -u +%H:%M:%SZ) ==="
