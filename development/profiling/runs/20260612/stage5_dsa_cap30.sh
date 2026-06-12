#!/usr/bin/env bash
# Stage 5 (apples-to-apples): DSA native FORCED to the same bs30 cap DS is KV-limited to,
# via --max-running-requests 30, at mem 0.7 (= DS's mem). Only difference from the DS column
# is DS-enablement itself; both decode at batch <=30. Directional serving sweep only.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/_env.sh"
cd /sgl-workspace/sglang
LOG="$HERE/stage5_dsa_cap30.log"; exec > >(tee "$LOG") 2>&1
echo "=== STAGE5 DSA bs30-cap start $(date -u +%H:%M:%S)Z ==="

teardown
# dsa07 (mem 0.7, DS off) + hard batch cap at 30 to match DS's KV-derived bs30 ceiling
build_server_args dsa07
SERVER_ARGS+=(--max-running-requests 30)
SERVE_LOG="$HERE/cap30_dsa/serve.log"; mkdir -p "$(dirname "$SERVE_LOG")"

echo ">>> booting DSA server (mem 0.7, DS-off, --max-running-requests 30) ..."
python -m sglang.launch_server "${SERVER_ARGS[@]}" > "$SERVE_LOG" 2>&1 &
if ! wait_ready; then echo "!! FAIL: DSA cap30 not ready"; tail -50 "$SERVE_LOG"; teardown; exit 1; fi

# sanity: confirm the running-batch cap is in effect
MRR=$(curl -s --max-time 8 "http://127.0.0.1:${PORT}/get_server_info" 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('max_running_requests') or (d.get('server_args') or {}).get('max_running_requests'))" 2>/dev/null)
echo ">>> server max_running_requests = ${MRR} (want 30)"

echo "=== DSA(bs30-cap) directional serving sweep (conc 16/32/64) $(date -u +%H:%M:%S)Z ==="
set +e
PORT=30000 HOST=127.0.0.1 MODE=native_nsa_cap30 \
  RESULTS_DIR="$HERE/serving" \
  CONCURRENCIES="16 32 64" TRIALS=1 \
  WARMUP_SECONDS=60 MEASUREMENT_WINDOW_S=180 NUM_PROMPTS=64 \
  bash development/benchmark_baseline.sh
SWEEP_RC=$?; set -e
echo ">>> DSA cap30 serving sweep rc=$SWEEP_RC"
teardown
echo "=== STAGE5 DSA bs30-cap done $(date -u +%H:%M:%S)Z sweep_rc=$SWEEP_RC ==="
