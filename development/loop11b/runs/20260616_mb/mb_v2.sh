#!/usr/bin/env bash
# loop11b R1 — clean M-B re-run from ONE HEAD addressing the Codex review:
#  - AC-4 controlled tax probe (B4): a DEDICATED fixed-concurrency bench at bs64 and bs30, GRAPH mode, mem 0.8,
#    radix declared. The steady-state median ITL = the one-batch decode window (each decode batch emits one
#    token per running request, so ITL == decode-batch time). DS/DSA bs64 ratio <= 1.10; bs30 10-step <= 380k us.
#  - AC-9 (B2): BOTH op-points from one HEAD — production_envelope (DS0.8/DSA0.85) + same_memory (DS0.8/DSA0.8),
#    radix-ON both, 2 trials/conc, 600s, separate artifact trees + comparator invocations.
#  - AC-2/3 (B3): capture the per-side running-request PEAK (conc-64 >= 61 means the workload reached nominal).
#  - AC-5 (B1 evidence): bench_serving now emits cached_tokens + DS no-op counters; trial_evidence.py gates them.
# Block-scheduled by side (one TP=8 server fits) -> LABELED unpaired; the HARD DS absolute verdict is DS-only
# (DEC-6), the DS/DSA ratio is reported with the drift caveat (per-boot thermal/clock logged).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /sgl-workspace/sglang/development/loop11b/runs/20260616_ma/mint/env.sh
cd "$REPO"
V2="$HERE/results_v2"; mkdir -p "$V2"/{ds080,dsa080,dsa085,tax}
export HOST=127.0.0.1 PORT=30000 TRIALS="${TRIALS:-2}" CONCURRENCIES="${CONCURRENCIES:-16 32 64}"
LOG="$V2/mb_v2.log"; exec > >(tee "$LOG") 2>&1
echo "=== loop11b R1 M-B clean re-run start $(date -u +%FT%TZ) HEAD=$(git rev-parse --short HEAD) ==="

boot() {  # $1=label $2=serve-cmd  -> sets SLOG
  teardown; SLOG="$V2/serve_${1}.log"
  echo "=== boot $1 $(date -u +%H:%M:%SZ) ==="
  eval "$2" > "$SLOG" 2>&1 &
  ready_wait "$SLOG" || { echo "!! $1 boot FAIL"; tail -80 "$SLOG"; teardown; gpu_idle_wait; return 10; }
  echo ">>> $1 ready. smoke=$(smoke)"
  curl -sf --max-time 20 "http://${HOST}:${PORT}/server_info" -o "$V2/server_info_${1}.json" || true
  echo ">>> $1 thermal/clock@ready: $(nvidia-smi --query-gpu=index,temperature.gpu,clocks.sm,power.draw --format=csv,noheader | tr '\n' '|')"
}

tax_probe() {  # $1=side  $2=conc  -> dedicated fixed-conc decode-window probe
  local side="$1" c="$2"; local out="$V2/tax/${side}_c${c}.jsonl"
  echo "--- tax probe $side conc=$c (warmup 20s, window 90s) $(date -u +%H:%M:%SZ) ---"
  python3 -m sglang.bench_serving --backend sglang --host "$HOST" --port "$PORT" --seed 7 \
    --dataset-name generated-shared-prefix --gsp-num-groups 1 --gsp-prompts-per-group $((c*4)) \
    --gsp-system-prompt-len 2253 --gsp-question-len 1843 --gsp-output-len 64 --gsp-range-ratio 1.0 \
    --num-prompts $((c*4)) --max-concurrency "$c" --warmup-seconds 20 --measurement-window-seconds 90 \
    --output-file "$out" --output-details > "$V2/tax/log_${side}_c${c}.txt" 2>&1
  grep -aiE 'Median ITL|Mean ITL|decode tok/s|Median decode' "$V2/tax/log_${side}_c${c}.txt" | head -6
}

peak_running() { grep -aoE '#running-req: [0-9]+' "$1" 2>/dev/null | grep -oE '[0-9]+' | sort -n | tail -1; }

# ---------- DS @ mem 0.8 (radix-on fixture) — shared by both comparisons ----------
boot ds080 "MODEL_PATH=$GLM CHANNEL_MASK_PATH=$MASK bash $REPO/development/serve_double_sparsity.sh" || exit 10
tax_probe ds 64; tax_probe ds 30
echo "=== DS sweep (2 trials x conc [$CONCURRENCIES] x 600s) $(date -u +%H:%M:%SZ) ==="
RESULTS_DIR="$V2/ds080" MODE=double_sparsity bash "$REPO/development/benchmark.sh"; echo ">>> DS sweep rc=$?"
echo ">>> DS running-req peak (all trials): $(peak_running "$SLOG")"
teardown; gpu_idle_wait

# ---------- DSA @ mem 0.8 (same-memory) ----------
boot dsa080 "MODEL_PATH=$GLM MEM_FRACTION_STATIC=0.8 bash $REPO/development/serve_native_nsa.sh" || exit 11
tax_probe dsa 64; tax_probe dsa 30
echo "=== DSA@0.8 sweep $(date -u +%H:%M:%SZ) ==="
RESULTS_DIR="$V2/dsa080" MODE=native_nsa bash "$REPO/development/benchmark.sh"; echo ">>> DSA@0.8 sweep rc=$?"
echo ">>> DSA@0.8 running-req peak: $(peak_running "$SLOG")"
teardown; gpu_idle_wait

# ---------- DSA @ mem 0.85 (production-envelope) ----------
boot dsa085 "MODEL_PATH=$GLM bash $REPO/development/serve_native_nsa.sh" || exit 12
echo "=== DSA@0.85 sweep $(date -u +%H:%M:%SZ) ==="
RESULTS_DIR="$V2/dsa085" MODE=native_nsa bash "$REPO/development/benchmark.sh"; echo ">>> DSA@0.85 sweep rc=$?"
echo ">>> DSA@0.85 running-req peak: $(peak_running "$SLOG")"
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

# ---------- AC-5 fail-closed per-trial evidence (DS trials) ----------
echo "=== AC-5 trial evidence (DS trials) $(date -u +%H:%M:%SZ) ==="
for j in "$V2/ds080"/double_sparsity_gsp_isl4096_osl512_c*_t*.jsonl; do
  python3 "$HERE/trial_evidence.py" "$j" && echo "  evidence PASS: $(basename "$j")" || echo "  !! evidence REFUSED: $(basename "$j")"
done
echo "=== mb_v2 done $(date -u +%H:%M:%SZ) ==="
