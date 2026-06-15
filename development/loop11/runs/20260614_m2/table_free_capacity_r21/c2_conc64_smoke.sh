#!/usr/bin/env bash
# Loop 11 M2 R21 — C2 conc-64 admission smoke against the LIVE table-free DS
# server (booted by C1). gen-shared-prefix isl~4096 / osl~512 / concurrency 64,
# 1 trial / 60s warmup / 180s window / 64 prompts (R20 c2 recipe). Records the
# decode #running-req peak; gate >= 61 (R9 admission gate). Does NOT teardown.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/r21_env.sh"
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
  echo "probe=c2_table_free_conc64 (live table-free DS server) decode_running_req_peak=${RUN:-0}"
  echo "admission_gate=$GATE"
  echo "bench_jsonl=$f"
  [[ -n "$f" ]] && python3 -c "import json;r=json.loads(open('$f').readline());print('achieved_conc=%.2f decTPS_p50=%.2f agg_tok/s=%.1f ttft_p99=%.2fs completed=%d'%(r['concurrency'],r['median_decode_throughput_tps'],r['output_throughput'],r['p99_ttft_ms']/1000,r['completed']))"
} > "$HERE/c2_conc64_evidence.txt"
cat "$HERE/c2_conc64_evidence.txt"
echo "=== C2 done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
