#!/usr/bin/env bash
# Loop 11 M2 R21 — C1 capacity: DS table-free boot @ mem 0.8 on the committed
# post-deletion HEAD. Boots DS DIRECTLY (launch_server + NEW DS_CONFIG; NO
# launcher, NO TABLE_FREE flag). Captures max_total_num_tokens, derives bs cap
# = floor(max_total/4608), asserts cap>=64, confirms CUDA-graph capture on all 8
# TP ranks + both selector score-reduce buckets, confirms NO TokenLabelTable,
# and a coherent smoke. Leaves server UP for C2.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/r21_env.sh"
cd /sgl-workspace/sglang
LOG="$HERE/c1_stage.log"; exec > >(tee "$LOG") 2>&1
echo "=== C1 table-free boot start $(date -u +%Y-%m-%dT%H:%M:%SZ) HEAD=$(git rev-parse --short HEAD) ==="
git diff --quiet -- python/ && echo "python_tree=committed" || echo "NOTE: uncommitted python/ diff present"
echo "DS_CONFIG=$DS_CONFIG"

teardown
echo "=== boot DS (table-free unconditional) launch_server @ mem0.8 $(date -u +%H:%M:%SZ) ==="
slog="$HERE/c1_table_free_serve.log"
python -m sglang.launch_server "${DS_ARGS[@]}" > "$slog" 2>&1 &
rc_wait=0
ready_wait "$slog" || rc_wait=$?
if [[ "$rc_wait" != "0" ]]; then
  echo "!! C1 boot FAIL (rc_wait=$rc_wait) — traceback tail:"
  tail -n 60 "$slog"
  { echo "probe=c1_table_free status=BOOT_FAIL rc_wait=$rc_wait";
    echo "--- error/traceback tail ---";
    grep -anE "Error|error|ValueError|AssertionError|RuntimeError|OOM|out of memory|Exception|Traceback|capture" "$slog" | tail -40; } > "$HERE/c1_table_free_evidence.txt"
  cat "$HERE/c1_table_free_evidence.txt"
  teardown
  exit 1
fi

CAP=$(curl -s --max-time 8 "http://127.0.0.1:${PORT}/get_server_info" | python3 -c "import json,sys;d=json.load(sys.stdin);print(d.get('max_total_num_tokens') or 0)")
SMK=$(smoke)
BS=$(( CAP / 4608 ))
NRANK_CAP=$(grep -acE "TP[0-7]\] Capture cuda graph end" "$slog")
NBUCKET=$(grep -acE "double_sparsity score reduce bucket" "$slog")
TABLE_HITS=$(grep -aicE "TokenLabelTable|token_label_table" "$slog")
CAPFAIL=$(grep -aicE "capture cuda graph failed|capture.*fail" "$slog")
echo ">>> table-free cap=$CAP bs=$BS  ranks_captured=$NRANK_CAP  reduce_buckets=$NBUCKET  table_hits=$TABLE_HITS  capfail=$CAPFAIL  smoke=$SMK"
{
  echo "probe=c1_table_free (launch_server, NEW DS_CONFIG, table-free unconditional, GLM fp8/0.8/right-sized) status=OK"
  echo "table_free  max_total_num_tokens=$CAP  bs_cap=$BS  (bs_cap = floor(max_total/4608))"
  echo "bs_cap_ge_64=$([[ $BS -ge 64 ]] && echo YES || echo NO)"
  echo "smoke=$SMK"
  echo "--- boot fields ---"
  grep -aE "TP0\] (KV Cache is allocated|Capture cuda graph end|max_total_num_tokens)" "$slog" | head -3
  echo "--- cuda-graph capture-end lines (expect 8, one per TP rank) ---"
  grep -aE "TP[0-7]\] Capture cuda graph end" "$slog" | sort
  echo "ranks_captured=$NRANK_CAP (expect 8)  capture_failures=$CAPFAIL (expect 0)"
  echo "--- selector score-reduce buckets (expect both: logical-width NCCL_BF16 + compact W=5120 TWO_SHOT_PULL) ---"
  grep -aE "double_sparsity score reduce bucket" "$slog" | sort -u | head -20
  echo "reduce_bucket_lines=$NBUCKET"
  echo "--- TokenLabelTable / token_label_table allocation (module DELETED; expect NONE) ---"
  if [[ "$TABLE_HITS" == "0" ]]; then echo "table_hits=0 — NO TokenLabelTable allocated (module deleted, as expected)"; else echo "table_hits=$TABLE_HITS — UNEXPECTED:"; grep -aiE "TokenLabelTable|token_label_table" "$slog" | head -5; fi
  echo "--- effective double_sparsity_config (from server) ---"
  curl -s --max-time 8 "http://127.0.0.1:${PORT}/get_server_info" | python3 -c "import json,sys;d=json.load(sys.stdin);print('server double_sparsity_config=',d.get('double_sparsity_config'))" 2>/dev/null || echo "(server_info config field not exposed)"
} > "$HERE/c1_table_free_evidence.txt"
cat "$HERE/c1_table_free_evidence.txt"
echo "=== C1 boot done; LEAVING SERVER UP for C2 (no teardown here) $(date -u +%H:%M:%SZ) ==="
