#!/usr/bin/env bash
# loop11b R1 — DS-only re-run, CORRECTED experimental design after the mb_v2 finding.
#
# Why this exists: mb_v2's first launch ran a conc-64 tax probe with 100% IDENTICAL prefixes
# (gsp-range-ratio 1.0, 1 group) BEFORE the sweep. That maximises the DS radix prefix-reuse path,
# which trips a selector shape bug ("36 vs 6790") pervasively (77500 errors in ~10s); the per-request
# abort path then leaks KV pages -> "pool memory leak detected" -> the TP=8 server crashed before the
# sweep produced any data. The check_finished->update_finish_state crash-fix (99ac584ac) removed the
# AttributeError crash but exposed that second (leak) layer under the pathological burst.
#
# Round-0 evidence: the SWEEP at the PRODUCTION ~55% reuse level is clean (2 selector errors over a full
# 6-trial sweep). So the bug scales with full-prefix reuse; the 100%-identical tax burst is unrepresentative.
#
# Corrected design:
#   1. SWEEP FIRST (the verdict) at radix-on, production ~55% reuse  -> ds080/  (protected: captured before
#      any teardown-time leak). Capture running-req peak + the per-phase selector-error count (HONEST).
#   2. DISTINCT-PREFIX tax probe LAST (gsp-prompts-per-group=1 -> every request its own prefix, NO reuse ->
#      no reuse-path selector bug) at fixed bs64 + bs30, GRAPH mode, mem 0.8 -> tax/. The per-step decode
#      window is reuse-independent, so a no-reuse probe is a valid AC-4 controlled measurement.
# DSA side is reused from the concurrent mb_v2 run (results_v2/dsa080,dsa085) at the SAME frozen HEAD.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /sgl-workspace/sglang/development/loop11b/runs/20260616_ma/mint/env.sh
cd "$REPO"
V2="$HERE/results_v2"; mkdir -p "$V2"/{ds080,tax}
export HOST=127.0.0.1 PORT=30000 TRIALS="${TRIALS:-2}" CONCURRENCIES="${CONCURRENCIES:-16 32 64}"
LOG="$V2/ds_only.log"; exec > >(tee "$LOG") 2>&1
echo "=== loop11b R1 DS-only re-run start $(date -u +%FT%TZ) HEAD=$(git rev-parse --short HEAD) ==="

peak_running() { grep -aoE '#running-req: [0-9]+' "$1" 2>/dev/null | grep -oE '[0-9]+' | sort -n | tail -1; }

# distinct-prefix fixed-conc decode-window probe (no radix reuse -> avoids the selector reuse pathology)
tax_probe() {  # $1=conc
  local c="$1"; local out="$V2/tax/ds_c${c}.jsonl"
  echo "--- tax probe (distinct-prefix) conc=$c $(date -u +%H:%M:%SZ) ---"
  python3 -m sglang.bench_serving --backend sglang --host "$HOST" --port "$PORT" --seed 7 \
    --dataset-name generated-shared-prefix --gsp-num-groups $((c*4)) --gsp-prompts-per-group 1 \
    --gsp-system-prompt-len 2253 --gsp-question-len 1843 --gsp-output-len 64 --gsp-range-ratio 1.0 \
    --num-prompts $((c*4)) --max-concurrency "$c" --warmup-seconds 20 --measurement-window-seconds 90 \
    --output-file "$out" --output-details > "$V2/tax/log_ds_c${c}.txt" 2>&1
  grep -aiE 'Median ITL|Mean ITL|Median E2E|Successful' "$V2/tax/log_ds_c${c}.txt" | head -6
}

# ---------- boot DS @ mem 0.8 (radix-on fixture) ----------
teardown 2>/dev/null || true
SLOG="$V2/serve_ds080.log"
echo "=== boot ds080 $(date -u +%H:%M:%SZ) ==="
MODEL_PATH="$GLM" CHANNEL_MASK_PATH="$MASK" bash "$REPO/development/serve_double_sparsity.sh" > "$SLOG" 2>&1 &
ready_wait "$SLOG" || { echo "!! ds080 boot FAIL"; tail -80 "$SLOG"; teardown; exit 10; }
echo ">>> ds080 ready. smoke=$(smoke)"
curl -sf --max-time 20 "http://${HOST}:${PORT}/server_info" -o "$V2/server_info_ds080.json" || true

# ---------- 1) SWEEP FIRST (the verdict) ----------
echo "=== DS sweep (2 trials x conc [$CONCURRENCIES] x 600s) $(date -u +%H:%M:%SZ) ==="
RESULTS_DIR="$V2/ds080" MODE=double_sparsity bash "$REPO/development/benchmark.sh"; echo ">>> DS sweep rc=$?"
echo ">>> DS running-req peak (all trials): $(peak_running "$SLOG")"
echo ">>> DS sweep selector_runtime_errors=$(grep -ac selector_runtime_error "$SLOG") server_crashed=$(grep -ac 'SIGQUIT received' "$SLOG")"

# ---------- 2) DISTINCT-PREFIX tax probe LAST (verdict already captured) ----------
echo "=== AC-4 tax probe (distinct-prefix, no reuse) $(date -u +%H:%M:%SZ) ==="
tax_probe 64; tax_probe 30
echo ">>> after tax selector_runtime_errors=$(grep -ac selector_runtime_error "$SLOG") server_crashed=$(grep -ac 'SIGQUIT received' "$SLOG")"
teardown; gpu_idle_wait

# ---------- AC-5 fail-closed per-trial evidence (DS trials) ----------
echo "=== AC-5 trial evidence (DS trials) $(date -u +%H:%M:%SZ) ==="
for j in "$V2/ds080"/double_sparsity_gsp_isl4096_osl512_c*_t*.jsonl; do
  python3 "$HERE/trial_evidence.py" "$j" && echo "  evidence PASS: $(basename "$j")" || echo "  !! evidence REFUSED: $(basename "$j")"
done
echo "=== ds_only done $(date -u +%H:%M:%SZ) ==="
