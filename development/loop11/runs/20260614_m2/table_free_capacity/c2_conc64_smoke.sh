#!/usr/bin/env bash
# Loop 11 M2 — C2 conc-64 admission smoke against the LIVE table_free server.
# Reuses the task4 recipe of record (1 trial / 60s warmup / 180s window / 64
# prompts). Records the decode #running-req peak (R9 owner-ratified gate >= 61).
# Server must already be up (booted by C1). Does NOT teardown — A7 phase tears
# the DS server down before booting DSA.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /sgl-workspace/sglang/development/profiling/runs/20260612/_env.sh
cd /sgl-workspace/sglang
LOG="$HERE/c2_stage.log"; exec > >(tee "$LOG") 2>&1
echo "=== C2 conc-64 admission smoke start $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
curl -sf --max-time 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 || { echo "!! no server on :$PORT — C1 must be up"; exit 1; }
slog="$HERE/c1_table_free_serve.log"

set +e
PORT=30000 HOST=127.0.0.1 MODE=ds_table_free RESULTS_DIR="$HERE/serving_c2" \
  CONCURRENCIES="64" TRIALS=1 WARMUP_SECONDS=60 MEASUREMENT_WINDOW_S=180 NUM_PROMPTS=64 \
  bash development/benchmark_baseline.sh
echo ">>> conc64 rc=$?"
set -e

f=$(ls "$HERE/serving_c2/"*"c64_t1.jsonl" 2>/dev/null | head -1)
RUN=$(grep -aoE "Decode batch. #running-req: [0-9]+" "$slog" | grep -oE "[0-9]+$" | sort -n | tail -1)
GATE=$([[ "${RUN:-0}" -ge 61 ]] && echo "PASS(decode #running-req=$RUN >= 61)" || echo "FAIL(decode #running-req=${RUN:-0} < 61)")
{
  echo "probe=c2_table_free_conc64 (live table_free server) decode_running_req_peak=${RUN:-0}"
  echo "admission_gate=$GATE"
  [[ -n "$f" ]] && python3 -c "import json;r=json.loads(open('$f').readline());print('achieved_conc=%.2f decTPS_p50=%.2f agg_tok/s=%.1f ttft_p99=%.2fs completed=%d'%(r['concurrency'],r['median_decode_throughput_tps'],r['output_throughput'],r['p99_ttft_ms']/1000,r['completed']))"
} > "$HERE/c2_conc64_evidence.txt"
cat "$HERE/c2_conc64_evidence.txt"
echo "=== C2 done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
