#!/usr/bin/env bash
# loop11b R2 — full verdict re-run at the AC-5-fixed HEAD so every DS SLO trial carries the per-request
# no-op fields (trial_evidence.py PASS) and the comparators re-accept at one matched commit_sha.
# Design (unchanged from the validated R1 corrected runners): SWEEP-first (verdict protected), distinct-
# prefix tax probe (no reuse pathology). DS @0.8 radix-on; DSA @0.8 (same-memory) + @0.85 (prod-envelope).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /sgl-workspace/sglang/development/loop11b/runs/20260616_ma/mint/env.sh
cd "$REPO"
V2="$HERE/results_r3"; mkdir -p "$V2"/{ds080,dsa080,dsa085,tax}
export HOST=127.0.0.1 PORT=30000 TRIALS="${TRIALS:-2}" CONCURRENCIES="${CONCURRENCIES:-16 32 64}"
LOG="$V2/mb_r3.log"; exec >> "$LOG" 2>&1
echo "=== loop11b R3 full re-run start $(date -u +%FT%TZ) HEAD=$(git rev-parse --short HEAD) ==="

peak_running() { grep -aoE '#running-req: [0-9]+' "$1" 2>/dev/null | grep -oE '[0-9]+' | sort -n | tail -1; }

boot() {  # $1=label $2=serve-cmd -> sets SLOG
  teardown 2>/dev/null || true; SLOG="$V2/serve_${1}.log"
  echo "=== boot $1 $(date -u +%H:%M:%SZ) ==="
  eval "$2" > "$SLOG" 2>&1 &
  ready_wait "$SLOG" || { echo "!! $1 boot FAIL"; tail -60 "$SLOG"; teardown; gpu_idle_wait; return 10; }
  echo ">>> $1 ready. smoke=$(smoke)"
  curl -sf --max-time 20 "http://${HOST}:${PORT}/server_info" -o "$V2/server_info_${1}.json" || true
}

tax_probe() {  # $1=side $2=conc — distinct-prefix fixed-conc decode-window probe (no reuse pathology)
  local side="$1" c="$2"; local out="$V2/tax/${side}_c${c}.jsonl"
  echo "--- tax probe $side (distinct-prefix) conc=$c $(date -u +%H:%M:%SZ) ---"
  python3 -m sglang.bench_serving --backend sglang --host "$HOST" --port "$PORT" --seed 7 \
    --dataset-name generated-shared-prefix --gsp-num-groups $((c*4)) --gsp-prompts-per-group 1 \
    --gsp-system-prompt-len 2253 --gsp-question-len 1843 --gsp-output-len 64 --gsp-range-ratio 1.0 \
    --num-prompts $((c*4)) --max-concurrency "$c" --warmup-seconds 20 --measurement-window-seconds 90 \
    --output-file "$out" --output-details > "$V2/tax/log_${side}_c${c}.txt" 2>&1
  grep -aiE 'Median ITL|Successful' "$V2/tax/log_${side}_c${c}.txt" | head -4
}

# ---------- DS @ 0.8 (radix-on) — sweep first (verdict), then tax ----------
boot ds080 "MODEL_PATH=$GLM CHANNEL_MASK_PATH=$MASK bash $REPO/development/serve_double_sparsity.sh" || exit 10
echo "=== DS sweep (2 trials x conc [$CONCURRENCIES] x 600s) $(date -u +%H:%M:%SZ) ==="
RESULTS_DIR="$V2/ds080" MODE=double_sparsity bash "$REPO/development/benchmark.sh"; echo ">>> DS sweep rc=$?"
echo ">>> DS running-req peak: $(peak_running "$SLOG") | selector_errors=$(grep -ac selector_runtime_error "$SLOG") crashed=$(grep -ac 'SIGQUIT received' "$SLOG")"
tax_probe ds 64; tax_probe ds 30
teardown; gpu_idle_wait

# ---------- DSA @ 0.8 (same-memory) ----------
boot dsa080 "MODEL_PATH=$GLM MEM_FRACTION_STATIC=0.8 bash $REPO/development/serve_native_nsa.sh" || exit 11
echo "=== DSA@0.8 sweep $(date -u +%H:%M:%SZ) ==="
RESULTS_DIR="$V2/dsa080" MODE=native_nsa bash "$REPO/development/benchmark.sh"; echo ">>> DSA@0.8 sweep rc=$?"
echo ">>> DSA@0.8 peak: $(peak_running "$SLOG")"
tax_probe dsa 64; tax_probe dsa 30
teardown; gpu_idle_wait

# ---------- DSA @ 0.85 (production-envelope) ----------
boot dsa085 "MODEL_PATH=$GLM bash $REPO/development/serve_native_nsa.sh" || exit 12
echo "=== DSA@0.85 sweep $(date -u +%H:%M:%SZ) ==="
RESULTS_DIR="$V2/dsa085" MODE=native_nsa bash "$REPO/development/benchmark.sh"; echo ">>> DSA@0.85 sweep rc=$?"
echo ">>> DSA@0.85 peak: $(peak_running "$SLOG")"
teardown; gpu_idle_wait

# ---------- comparators (same HEAD both sides) ----------
cmp() {  # $1=tree-label $2=dsa-dir
  echo "=== compare $1 (DS@0.8 vs $2) $(date -u +%H:%M:%SZ) ==="
  python3 "$REPO/development/benchmark_compare.py" --ac11 \
    --ac11-baseline-results "$V2/$2"/native_nsa_gsp_isl4096_osl512_c*_t*.jsonl \
    --ac11-ds-results "$V2/ds080"/double_sparsity_gsp_isl4096_osl512_c*_t*.jsonl \
    --output "$V2/ac11_${1}.md" --json-output "$V2/ac11_${1}.json"
  echo "=== compare $1 rc=$? (0=PASS,3=absolute SLO fail,2=refusal) ==="
}
cmp production_envelope dsa085
cmp same_memory dsa080

# ---------- AC-5 fail-closed per-trial evidence (must PASS now) ----------
echo "=== AC-5 trial evidence (DS trials) $(date -u +%H:%M:%SZ) ==="
for j in "$V2/ds080"/double_sparsity_gsp_isl4096_osl512_c*_t*.jsonl; do
  python3 "$HERE/trial_evidence.py" "$j" >/dev/null 2>&1 && echo "  PASS $(basename "$j")" || echo "  !! REFUSE $(basename "$j")"
done
echo "=== mb_r3 done $(date -u +%H:%M:%SZ) ==="
