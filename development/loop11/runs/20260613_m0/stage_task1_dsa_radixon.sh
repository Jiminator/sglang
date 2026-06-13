#!/usr/bin/env bash
# Loop 11 task1: freeze the radix-ON DSA @0.8 directional baseline ladder.
# Recipe of record = runs/20260612 stage2 (same COMMON_ARGS, seeds, gsp shape,
# 1 trial / 60s warmup / 180s window / NUM_PROMPTS=64) with EXACTLY ONE change:
# radix cache ON (drop --disable-radix-cache). This is the loop's AC-2/AC-3
# comparison column; the 20260612 radix-off ladder stays the radix-off reference.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /sgl-workspace/sglang/development/profiling/runs/20260612/_env.sh
cd /sgl-workspace/sglang
LOG="$HERE/stage_task1.log"; exec > >(tee "$LOG") 2>&1
echo "=== TASK1 radix-ON DSA@0.8 ladder start $(date -u +%Y-%m-%dT%H:%M:%SZ) HEAD=$(git rev-parse --short HEAD) ==="

# Probe env hygiene: these must NEVER be set for baseline/served runs.
unset SGLANG_DS_PROBE_TABLE_TOKENS SGLANG_DS_PROBE_SKIP_INDEXER || true

teardown
build_server_args dsa08
# Single recipe change: radix ON.
ARGS=()
for a in "${SERVER_ARGS[@]}"; do
  [[ "$a" == "--disable-radix-cache" ]] && continue
  ARGS+=("$a")
done

SERVE_LOG="$HERE/dsa08_radixon_serve.log"
echo ">>> booting DSA@0.8 radix-ON ..."
python -m sglang.launch_server "${ARGS[@]}" > "$SERVE_LOG" 2>&1 &
if ! wait_ready; then echo "!! FAIL: not ready"; tail -50 "$SERVE_LOG"; teardown; exit 1; fi

# Verify radix is actually ON in the served config.
RADIX_OFF=$(curl -s --max-time 8 "http://127.0.0.1:${PORT}/get_server_info" | python3 -c "
import json,sys
d=json.load(sys.stdin)
print(d.get('disable_radix_cache', 'MISSING'))")
echo ">>> server disable_radix_cache=${RADIX_OFF} (expect False)"
if [[ "$RADIX_OFF" != "False" ]]; then echo "!! FAIL: radix not on"; teardown; exit 1; fi

CAP=$(max_batch_from_server)
echo ">>> max_total_num_tokens=${CAP} => bs_cap=floor(cap/4608)=$(( CAP / 4608 ))"

# Boot fields for the task0 accounting table (KV alloc, graph mem, stage avail-mem).
grep -aE "KV Cache is allocated|max_total_num_tokens|token_label_table|Capture cuda graph|avail mem|mem usage|Memory pool end" "$SERVE_LOG" | head -40 > "$HERE/dsa08_radixon_boot_fields.txt" || true

echo "=== ladder (conc 16/32/64, 1 trial, 60s warmup / 180s window, NUM_PROMPTS=64) ==="
set +e
PORT=30000 HOST=127.0.0.1 MODE=native_nsa \
  RESULTS_DIR="$HERE/serving" \
  CONCURRENCIES="16 32 64" TRIALS=1 \
  WARMUP_SECONDS=60 MEASUREMENT_WINDOW_S=180 NUM_PROMPTS=64 \
  bash development/benchmark_baseline.sh
SWEEP_RC=$?
set -e
echo ">>> ladder rc=$SWEEP_RC"
teardown

# Summarize (fields match the 20260612 SUMMARY.txt convention).
python3 - "$HERE/serving" <<'PYEOF' > "$HERE/serving/SUMMARY.txt"
import glob, json, os, sys
d = sys.argv[1]
print("# FROZEN loop11 task1 baseline — radix-ON DSA @ mem 0.8 (1 trial, 60s warmup/180s window,")
print("# NUM_PROMPTS=64, gsp 4096-ISL/512-OSL ~55% prefix, seeds 16:213/32:431/64:31234).")
print("# Single recipe change vs runs/20260612 stage2: radix cache ON. NEVER RE-RUN.")
print(f"{'config':<22} {'conc':>4} {'decTPS_p50':>10} {'agg_tok/s':>9} {'ach_conc':>8} {'ttft_mean':>9} {'ttft_med':>8} {'ttft_p99':>8} {'tpot_p99':>8} {'done':>5}")
for c in (16, 32, 64):
    fs = sorted(glob.glob(os.path.join(d, f"native_nsa_gsp_isl4096_osl512_c{c}_t1.jsonl")))
    for f in fs:
        with open(f) as fh:
            r = json.loads(fh.readline())
        print(f"{'DSA@0.8 radix-ON':<22} {c:>4} {r['median_decode_throughput_tps']:>10.2f} "
              f"{r['output_throughput']:>9.1f} {r['concurrency']:>8.2f} "
              f"{r['mean_ttft_ms']/1000:>9.2f} {r['median_ttft_ms']/1000:>8.2f} "
              f"{r['p99_ttft_ms']/1000:>8.2f} {r['p99_tpot_ms']:>8.2f} {r['completed']:>5}")
PYEOF
cat "$HERE/serving/SUMMARY.txt"
echo "=== TASK1 done $(date -u +%Y-%m-%dT%H:%M:%SZ) sweep_rc=$SWEEP_RC ==="
