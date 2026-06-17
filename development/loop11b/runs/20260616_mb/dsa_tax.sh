#!/usr/bin/env bash
# loop11b R1 — DSA distinct-prefix tax probe to MATCH the DS dedicated tax probe (ds_only.sh) for a
# clean, like-for-like AC-4 per-step decode-window ratio. Same fixed-conc (bs64,bs30), same distinct-prefix
# GSP shape (gsp-prompts-per-group=1 -> no reuse), DSA at mem 0.8 (matches DS mem 0.8). GRAPH mode.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /sgl-workspace/sglang/development/loop11b/runs/20260616_ma/mint/env.sh
cd "$REPO"
V2="$HERE/results_v2"; mkdir -p "$V2"/tax
export HOST=127.0.0.1 PORT=30000
LOG="$V2/dsa_tax.log"; exec > >(tee "$LOG") 2>&1
echo "=== loop11b R1 DSA tax probe start $(date -u +%FT%TZ) HEAD=$(git rev-parse --short HEAD) ==="

tax_probe() {  # $1=conc — distinct-prefix fixed-conc decode-window probe (matches ds_only.sh)
  local c="$1"; local out="$V2/tax/dsa_c${c}.jsonl"
  echo "--- DSA tax probe (distinct-prefix) conc=$c $(date -u +%H:%M:%SZ) ---"
  python3 -m sglang.bench_serving --backend sglang --host "$HOST" --port "$PORT" --seed 7 \
    --dataset-name generated-shared-prefix --gsp-num-groups $((c*4)) --gsp-prompts-per-group 1 \
    --gsp-system-prompt-len 2253 --gsp-question-len 1843 --gsp-output-len 64 --gsp-range-ratio 1.0 \
    --num-prompts $((c*4)) --max-concurrency "$c" --warmup-seconds 20 --measurement-window-seconds 90 \
    --output-file "$out" --output-details > "$V2/tax/log_dsa_c${c}.txt" 2>&1
  grep -aiE 'Median ITL|Mean ITL|Median E2E|Successful' "$V2/tax/log_dsa_c${c}.txt" | head -6
}

teardown 2>/dev/null || true
SLOG="$V2/serve_dsa080_tax.log"
echo "=== boot dsa080 (mem 0.8) $(date -u +%H:%M:%SZ) ==="
MODEL_PATH="$GLM" MEM_FRACTION_STATIC=0.8 bash "$REPO/development/serve_native_nsa.sh" > "$SLOG" 2>&1 &
ready_wait "$SLOG" || { echo "!! dsa080 boot FAIL"; tail -60 "$SLOG"; teardown; exit 10; }
echo ">>> dsa080 ready. smoke=$(smoke)"
tax_probe 64; tax_probe 30
teardown; gpu_idle_wait
echo "=== dsa_tax done $(date -u +%H:%M:%SZ) ==="
