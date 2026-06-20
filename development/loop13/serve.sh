#!/usr/bin/env bash
# Boot GLM-5.1-FP8 from the DEV clone (no PYTHONPATH override).
#   Usage:  serve.sh <dsa|ds>
#   - dsa : native DSA indexer (DS off)            — the accuracy target
#   - ds  : current table-free Double Sparsity     — radix disabled (dev-clone gate)
# Run this BACKGROUNDED (it polls readiness with sleep). Writes PID to $PIDFILE.
set -uo pipefail
HERE=$(dirname "$(readlink -f "$0")")
# shellcheck source=_env.sh
source "$HERE/_env.sh" || exit 1

MODE="${1:?usage: serve.sh <dsa|ds>}"
LOG="$EVID/serve_${MODE}.log"

COMMON=(
  --model-path "$MODEL" --host "$HOST" --port "$PORT"
  --tp-size 8 --kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.8
  --max-running-requests 64 --cuda-graph-max-bs 64 --page-size 64
  --dsa-prefill-backend flashmla_kv --dsa-decode-backend flashmla_kv
  --disable-overlap-schedule --disable-piecewise-cuda-graph
  --random-seed 42 --trust-remote-code
)

case "$MODE" in
  dsa) EXTRA=() ;;
  ds)
    [ -s "$MASK" ] || { echo "FATAL: mask $MASK missing"; exit 2; }
    DS_CONFIG=$(printf '{"top_k": 2048, "page_size": 64, "channel_mask_path": "%s", "device_buffer_size": 4096, "scorer_norm": "off", "head_agg": "max", "anchor_mode": "off", "anchor_budget": 0, "enable_lifted_budget_decode": false, "lifted_budget_top_k": 0}' "$MASK")
    # NOTE: dev clone gates DS+radix -> --disable-radix-cache (output-neutral at temp 0).
    EXTRA=( --disable-radix-cache --enable-double-sparsity --double-sparsity-config "$DS_CONFIG" )
    ;;
  *) echo "FATAL: mode must be 'dsa' or 'ds'"; exit 2 ;;
esac

# *** NO PYTHONPATH *** — default editable install = dev clone (the guard enforced this).
nohup python3 -m sglang.launch_server "${COMMON[@]}" "${EXTRA[@]}" > "$LOG" 2>&1 &
echo $! > "$PIDFILE"
echo "[$MODE] PID=$(cat "$PIDFILE")  log=$LOG"

for i in $(seq 1 180); do
  curl -sf "http://$HOST:$PORT/health" >/dev/null 2>&1 && { echo "READY (~$((i*5))s)"; exit 0; }
  kill -0 "$(cat "$PIDFILE")" 2>/dev/null || { echo "SERVER DIED"; tail -40 "$LOG"; exit 1; }
  sleep 5
done
echo "TIMEOUT"; tail -40 "$LOG"; exit 1
