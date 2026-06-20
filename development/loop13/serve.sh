#!/usr/bin/env bash
# Boot GLM-5.1-FP8 from the DEV clone (no PYTHONPATH override).
#   Usage:  serve.sh <dsa|dsa_noradix|ds|ds_capture|ref|ds_forced_all>
#   - dsa           : native DSA indexer (DS off)          — the accuracy target
#   - dsa_noradix   : DSA + --disable-radix-cache          — radix-cache-neutral control
#   - ds            : current table-free Double Sparsity   — radix disabled (dev-clone gate)
#   - ds_capture    : production DS, EAGER + score/selection capture — cheap-control data
#   - ref           : perf-naive fp32 raw-dot reference selector (EAGER) — accuracy ceiling
#   - ds_forced_all : dense forced-all [0..seq-1] control (EAGER) — H3 downstream-isolation
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
  dsa_noradix) EXTRA=( --disable-radix-cache ) ;;
  ds)
    [ -s "$MASK" ] || { echo "FATAL: mask $MASK missing"; exit 2; }
    DS_CONFIG=$(printf '{"top_k": 2048, "page_size": 64, "channel_mask_path": "%s", "device_buffer_size": 4096, "scorer_norm": "off", "head_agg": "max", "anchor_mode": "off", "anchor_budget": 0, "enable_lifted_budget_decode": false, "lifted_budget_top_k": 0}' "$MASK")
    # NOTE: dev clone gates DS+radix -> --disable-radix-cache (output-neutral at temp 0).
    EXTRA=( --disable-radix-cache --enable-double-sparsity --double-sparsity-config "$DS_CONFIG" )
    ;;
  ds_capture)
    # Same production DS selection, but EAGER (--disable-cuda-graph, required for
    # host-side score/oracle copies) with per-(rank,req,layer) score capture and
    # per-step selection capture enabled. Drive with SINGLE requests (bs=1) so a
    # score row maps unambiguously to selection row 0. Capture dirs default to
    # CWD/.sglang_ds_*; override via SGLANG_DS_*_CAPTURE_DIR.
    [ -s "$MASK" ] || { echo "FATAL: mask $MASK missing"; exit 2; }
    DS_CONFIG=$(printf '{"top_k": 2048, "page_size": 64, "channel_mask_path": "%s", "device_buffer_size": 4096, "scorer_norm": "off", "head_agg": "max", "anchor_mode": "off", "anchor_budget": 0, "enable_lifted_budget_decode": false, "lifted_budget_top_k": 0, "score_capture": true, "selection_capture": true}' "$MASK")
    EXTRA=( --disable-radix-cache --disable-cuda-graph --enable-double-sparsity --double-sparsity-config "$DS_CONFIG" )
    ;;
  ref)
    # Performance-naive fp32 raw-dot REFERENCE selector (the accuracy ceiling):
    # dequantize the resident latent to fp32, exact absorbed channel-dot, exact
    # full-width torch top-k. EAGER (--disable-cuda-graph; host dequant illegal
    # under graph). selector_impl="reference_rawdot".
    [ -s "$MASK" ] || { echo "FATAL: mask $MASK missing"; exit 2; }
    DS_CONFIG=$(printf '{"top_k": 2048, "page_size": 64, "channel_mask_path": "%s", "device_buffer_size": 4096, "scorer_norm": "off", "head_agg": "max", "anchor_mode": "off", "anchor_budget": 0, "enable_lifted_budget_decode": false, "lifted_budget_top_k": 0, "selector_impl": "reference_rawdot"}' "$MASK")
    EXTRA=( --disable-radix-cache --disable-cuda-graph --enable-double-sparsity --double-sparsity-config "$DS_CONFIG" )
    ;;
  ds_forced_all)
    # Dense forced-all downstream-isolation control: for seq <= top_k the selector
    # emits logical [0..seq_len-1] (selection is a no-op), so residual dense
    # degradation localizes downstream of selection. EAGER for host-side checks.
    [ -s "$MASK" ] || { echo "FATAL: mask $MASK missing"; exit 2; }
    DS_CONFIG=$(printf '{"top_k": 2048, "page_size": 64, "channel_mask_path": "%s", "device_buffer_size": 4096, "scorer_norm": "off", "head_agg": "max", "anchor_mode": "off", "anchor_budget": 0, "enable_lifted_budget_decode": false, "lifted_budget_top_k": 0, "forced_all_dense_control": true}' "$MASK")
    EXTRA=( --disable-radix-cache --disable-cuda-graph --enable-double-sparsity --double-sparsity-config "$DS_CONFIG" )
    ;;
  ds_anchor)
    # Sparse-regime H3 confirmation: production top-2048 selection PLUS a recency
    # anchor that force-includes the most-recent ANCHOR_BUDGET slots (incl. the
    # current decode slot) on top of top-k. If sparse recovers, the current/recent
    # slot exclusion is the sparse bug too. ANCHOR_BUDGET env (default 64).
    [ -s "$MASK" ] || { echo "FATAL: mask $MASK missing"; exit 2; }
    AB="${ANCHOR_BUDGET:-64}"
    DS_CONFIG=$(printf '{"top_k": 2048, "page_size": 64, "channel_mask_path": "%s", "device_buffer_size": 4096, "scorer_norm": "off", "head_agg": "max", "anchor_mode": "recency", "anchor_budget": %s, "enable_lifted_budget_decode": false, "lifted_budget_top_k": 0}' "$MASK" "$AB")
    EXTRA=( --disable-radix-cache --enable-double-sparsity --double-sparsity-config "$DS_CONFIG" )
    ;;
  *) echo "FATAL: mode must be 'dsa', 'dsa_noradix', 'ds', 'ds_capture', 'ref', 'ds_forced_all', or 'ds_anchor'"; exit 2 ;;
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
