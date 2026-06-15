#!/usr/bin/env bash
# AC-7 — DSA default boot regression. NO --enable-double-sparsity, mem 0.8,
# shipped default (radix cache ON by default). max_total_num_tokens MUST == 410560.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/r24_env.sh"
cd /sgl-workspace/sglang
LOG="$HERE/a7_stage.log"; exec > >(tee "$LOG") 2>&1
echo "=== A7 DSA-default start $(date -u +%Y-%m-%dT%H:%M:%SZ) HEAD=$(git rev-parse --short HEAD) ==="
EXPECTED=410560

teardown
slog="$HERE/a7_serve.log"
echo "=== boot DSA-default (no DS) @ mem0.8 $(date -u +%H:%M:%SZ) ==="
python -m sglang.launch_server "${DSA_ARGS[@]}" > "$slog" 2>&1 &
rc_wait=0
ready_wait "$slog" || rc_wait=$?
if [[ "$rc_wait" != "0" ]]; then
  echo "!! A7 boot FAIL (rc_wait=$rc_wait) — tail:"
  tail -n 60 "$slog"
  { echo "probe=a7_dsa_default status=BOOT_FAIL rc_wait=$rc_wait";
    grep -anE "Error|ValueError|RuntimeError|OOM|out of memory|Traceback" "$slog" | tail -40; } > "$HERE/a7_evidence.txt"
  teardown; gpu_idle_wait
  exit 1
fi

CAP=$(curl -s --max-time 8 "http://127.0.0.1:${PORT}/get_server_info" | python3 -c "import json,sys;d=json.load(sys.stdin);print(d.get('max_total_num_tokens') or 0)")
SMK=$(smoke)
MATCH=$([[ "$CAP" == "$EXPECTED" ]] && echo MATCH || echo MISMATCH)
NRANK_KV=$(grep -acE "TP[0-7]\] KV Cache is allocated.*#tokens: ${EXPECTED}" "$slog")
echo ">>> A7 max_total_num_tokens=$CAP expected=$EXPECTED -> $MATCH  kv_ranks=$NRANK_KV  smoke=$SMK"
{
  echo "probe=a7_dsa_default (NO --enable-double-sparsity, mem 0.8, shipped radix-default) status=OK"
  echo "max_total_num_tokens=$CAP  expected=$EXPECTED  result=$MATCH"
  echo "kv_ranks_logging_${EXPECTED}=$NRANK_KV (expect 8)"
  echo "smoke=$SMK"
  echo "--- boot fields ---"
  grep -aE "TP0\] (KV Cache is allocated|max_total_num_tokens)" "$slog" | head -3
} > "$HERE/a7_evidence.txt"
cat "$HERE/a7_evidence.txt"

echo "=== A7 done; tearing down $(date -u +%H:%M:%SZ) ==="
teardown
gpu_idle_wait
[[ "$MATCH" == "MATCH" ]] && exit 0 || exit 1
