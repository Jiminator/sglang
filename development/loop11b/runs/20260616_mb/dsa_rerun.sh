#!/usr/bin/env bash
# loop11b M-B — re-run the DSA-native baseline at the MATCHED locked op-point (mem 0.85,
# max_running_requests=64, cuda_graph_max_bs=64) so benchmark_compare.py --ac11 accepts the
# cross-side operating point (the first run left DSA uncapped at 512/None). DS JSONLs are kept;
# this overwrites only the native_nsa_*.jsonl, then re-runs the comparator (DEC-6 absolute gate).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /sgl-workspace/sglang/development/loop11b/runs/20260616_ma/mint/env.sh
cd "$REPO"
RESULTS="$HERE/results_prod_envelope"
export HOST=127.0.0.1 PORT=30000 RESULTS_DIR="$RESULTS" TRIALS="${TRIALS:-2}" CONCURRENCIES="${CONCURRENCIES:-16 32 64}"
LOG="$RESULTS/dsa_rerun.log"; exec > >(tee "$LOG") 2>&1
echo "=== DSA re-run at matched op-point (64/64) $(date -u +%FT%TZ) HEAD=$(git rev-parse --short HEAD) ==="
teardown
MODEL_PATH="$GLM" bash "$REPO/development/serve_native_nsa.sh" > "$RESULTS/serve_dsa_rerun.log" 2>&1 &
ready_wait "$RESULTS/serve_dsa_rerun.log" || { echo "!! DSA boot FAIL"; tail -80 "$RESULTS/serve_dsa_rerun.log"; teardown; gpu_idle_wait; exit 10; }
echo ">>> DSA ready. smoke=$(smoke)"
curl -sf --max-time 20 "http://${HOST}:${PORT}/server_info" -o "$RESULTS/server_info_dsa.json" || true
echo ">>> DSA thermal/clock@ready: $(nvidia-smi --query-gpu=index,temperature.gpu,clocks.sm,power.draw --format=csv,noheader | tr '\n' '|')"
echo "=== DSA bench ($TRIALS trials x conc [$CONCURRENCIES] x 600s) $(date -u +%H:%M:%SZ) ==="
MODE=native_nsa bash "$REPO/development/benchmark.sh"; RC=$?
teardown; gpu_idle_wait
echo ">>> DSA bench rc=$RC $(date -u +%H:%M:%SZ)"
echo "=== benchmark_compare.py --ac11 (matched op-point) $(date -u +%H:%M:%SZ) ==="
python3 "$REPO/development/benchmark_compare.py" --ac11 \
  --ac11-baseline-results "$RESULTS"/native_nsa_gsp_isl4096_osl512_c*_t*.jsonl \
  --ac11-ds-results "$RESULTS"/double_sparsity_gsp_isl4096_osl512_c*_t*.jsonl \
  --output "$RESULTS/ac11_report.md" --json-output "$RESULTS/ac11_verdict.json"
CMP_RC=$?
echo "=== compare rc=$CMP_RC (0=DS absolute SLO PASS, 3=DS absolute SLO FAIL, 2=input refusal) $(date -u +%H:%M:%SZ) ==="