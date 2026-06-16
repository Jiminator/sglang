#!/usr/bin/env bash
# loop11b M-B — locked DS-vs-DSA sweep, PRODUCTION-ENVELOPE op-point (DS mem 0.8 / DSA mem 0.85),
# radix-ON both, 2 trials/conc (DEC-4, repeated run-to-run-stability at the same per-conc seed),
# conc 16/32/64, 600s window. ONE TP=8 server at a time => BLOCK-SCHEDULED by side, LABELED
# unpaired: the DS ABSOLUTE SLO verdict (DEC-6: decode-TPS p50 >= 30, P99 TTFT < 22s, judged
# regardless of DSA) is self-contained; the DS/DSA ratio is REPORTED with a drift caveat
# (thermal/clock logged per boot). The same-memory op-point (both 0.8) is DEFERRED-and-recorded
# (plan lower bound) — run as a follow-up if the schedule allows.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /sgl-workspace/sglang/development/loop11b/runs/20260616_ma/mint/env.sh
cd "$REPO"
RESULTS="$HERE/results_prod_envelope"; mkdir -p "$RESULTS"
export HOST=127.0.0.1 PORT=30000 RESULTS_DIR="$RESULTS" TRIALS="${TRIALS:-2}" CONCURRENCIES="${CONCURRENCIES:-16 32 64}"
LOG="$RESULTS/sweep.log"; exec > >(tee "$LOG") 2>&1
echo "=== loop11b locked sweep (production-envelope) start $(date -u +%FT%TZ) HEAD=$(git rev-parse --short HEAD) TRIALS=$TRIALS conc=[$CONCURRENCIES] ==="

boot_wait() {  # $1=label  $2=slog
  ready_wait "$2" || { echo "!! $1 boot FAIL"; tail -80 "$2"; teardown; gpu_idle_wait; return 10; }
  echo ">>> $1 ready. smoke=$(smoke)"
  curl -sf --max-time 20 "http://${HOST}:${PORT}/server_info" -o "$RESULTS/server_info_${1}.json" || true
  echo ">>> $1 thermal/clock@ready: $(nvidia-smi --query-gpu=index,temperature.gpu,clocks.sm,power.draw --format=csv,noheader | tr '\n' '|')"
}

# ---------- DS block (mem 0.8, radix-on via the minted fixture) ----------
teardown
echo "=== boot DS (serve_double_sparsity.sh; mem 0.8, radix-on fixture) $(date -u +%H:%M:%SZ) ==="
MODEL_PATH="$GLM" CHANNEL_MASK_PATH="$MASK" bash "$REPO/development/serve_double_sparsity.sh" > "$RESULTS/serve_ds.log" 2>&1 &
boot_wait ds "$RESULTS/serve_ds.log" || exit 10
echo "=== DS bench ($TRIALS trials x conc [$CONCURRENCIES] x 600s) $(date -u +%H:%M:%SZ) ==="
MODE=double_sparsity bash "$REPO/development/benchmark.sh"; DS_BENCH_RC=$?
teardown; gpu_idle_wait
echo ">>> DS bench rc=$DS_BENCH_RC $(date -u +%H:%M:%SZ)"

# ---------- DSA block (mem 0.85, radix-on default) ----------
echo "=== boot DSA (serve_native_nsa.sh; mem 0.85, radix-on) $(date -u +%H:%M:%SZ) ==="
MODEL_PATH="$GLM" bash "$REPO/development/serve_native_nsa.sh" > "$RESULTS/serve_dsa.log" 2>&1 &
boot_wait dsa "$RESULTS/serve_dsa.log" || exit 11
echo "=== DSA bench ($TRIALS trials x conc [$CONCURRENCIES] x 600s) $(date -u +%H:%M:%SZ) ==="
MODE=native_nsa bash "$REPO/development/benchmark.sh"; DSA_BENCH_RC=$?
teardown; gpu_idle_wait
echo ">>> DSA bench rc=$DSA_BENCH_RC $(date -u +%H:%M:%SZ)"

# ---------- compare (DEC-6: absolute SLO gates the exit; ratio reported) ----------
echo "=== benchmark_compare.py --ac11 $(date -u +%H:%M:%SZ) ==="
python3 "$REPO/development/benchmark_compare.py" --ac11 \
  --ac11-baseline-results "$RESULTS"/native_nsa_gsp_isl4096_osl512_c*_t*.jsonl \
  --ac11-ds-results "$RESULTS"/double_sparsity_gsp_isl4096_osl512_c*_t*.jsonl \
  --output "$RESULTS/ac11_report.md" --json-output "$RESULTS/ac11_verdict.json"
CMP_RC=$?
echo "=== compare rc=$CMP_RC (0=DS absolute SLO PASS, 3=DS absolute SLO FAIL, 2=input refusal) ==="
echo "=== sweep done $(date -u +%H:%M:%SZ) (ds_bench=$DS_BENCH_RC dsa_bench=$DSA_BENCH_RC cmp=$CMP_RC) ==="
