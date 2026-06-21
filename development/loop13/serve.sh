#!/usr/bin/env bash
# Boot GLM-5.1-FP8 from the DEV clone (no PYTHONPATH override).
#   Usage:  serve.sh <dsa|dsa_noradix|ds|ds_capture|ref|ds_forced_all>
#   - dsa           : native DSA indexer (DS off)          — the accuracy target
#   - dsa_noradix   : DSA + --disable-radix-cache          — radix-cache-neutral control
#   - ds            : current table-free Double Sparsity   — radix disabled (dev-clone gate)
#   - ds_capture    : production DS, EAGER + score/selection capture — cheap-control data
#   - ref           : fp32 raw-dot reference (EAGER) — scorer-isolation (production slot-validity)
#   - ref_faithful  : fp32 raw-dot reference, TF32 off + current slot included (EAGER) — faithful ceiling
#   - ref_cosine    : fp32 COSINE reference, TF32 off + current slot included (EAGER) — faithful cosine ceiling
#   - ds_forced_all : dense forced-all [0..seq-1] control (EAGER) — downstream-isolation
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
  ds_reduce_fp32)
    # AC-6 single-variable bisection arm (leg 7): EXACTLY the production `ds` config with the ONE
    # variable flipped -> score_reduce_dtype="fp32" (cross-TP score SUM-reduce in fp32 vs the bf16
    # default). Same graph mode as `ds`/production_ds so reduce dtype is the ONLY difference (a clean
    # single-variable step; eager would also change the graph path). Config-only; no selection fix.
    # (Dense-regime captures for the current-slot corroboration come from the separate eager run.)
    [ -s "$MASK" ] || { echo "FATAL: mask $MASK missing"; exit 2; }
    DS_CONFIG=$(printf '{"top_k": 2048, "page_size": 64, "channel_mask_path": "%s", "device_buffer_size": 4096, "scorer_norm": "off", "head_agg": "max", "anchor_mode": "off", "anchor_budget": 0, "enable_lifted_budget_decode": false, "lifted_budget_top_k": 0, "score_reduce_dtype": "fp32"}' "$MASK")
    EXTRA=( --disable-radix-cache --enable-double-sparsity --double-sparsity-config "$DS_CONFIG" )
    ;;
  ref)
    # Raw-dot REFERENCE scorer-ISOLATION control: exact fp32 absorbed channel-dot
    # but UNDER the production slot-validity condition (current decode slot still
    # excluded via _slot_written). Useful to exonerate the scorer/perf-opts, NOT a
    # faithful ceiling. EAGER. selector_impl="reference_rawdot".
    [ -s "$MASK" ] || { echo "FATAL: mask $MASK missing"; exit 2; }
    DS_CONFIG=$(printf '{"top_k": 2048, "page_size": 64, "channel_mask_path": "%s", "device_buffer_size": 4096, "scorer_norm": "off", "head_agg": "max", "anchor_mode": "off", "anchor_budget": 0, "enable_lifted_budget_decode": false, "lifted_budget_top_k": 0, "selector_impl": "reference_rawdot"}' "$MASK")
    EXTRA=( --disable-radix-cache --disable-cuda-graph --enable-double-sparsity --double-sparsity-config "$DS_CONFIG" )
    ;;
  ref_faithful)
    # FAITHFUL, leak-free raw-dot accuracy ceiling: exact fp32 absorbed channel-dot,
    # TF32 disabled, AND the current decode slot force-included (reference_include_current),
    # so the run is slot-validity-clean (dense reports selected==seq_len). This is the decision-gate
    # input for raw-dot. EAGER.
    [ -s "$MASK" ] || { echo "FATAL: mask $MASK missing"; exit 2; }
    DS_CONFIG=$(printf '{"top_k": 2048, "page_size": 64, "channel_mask_path": "%s", "device_buffer_size": 4096, "scorer_norm": "off", "head_agg": "max", "anchor_mode": "off", "anchor_budget": 0, "enable_lifted_budget_decode": false, "lifted_budget_top_k": 0, "selector_impl": "reference_rawdot", "reference_include_current": true}' "$MASK")
    EXTRA=( --disable-radix-cache --disable-cuda-graph --enable-double-sparsity --double-sparsity-config "$DS_CONFIG" )
    ;;
  ref_cosine)
    # FAITHFUL, leak-free COSINE accuracy ceiling (Loop-7 lever): direction-normalized
    # absorbed score on a materialized per-head signature (normalize after mask-channel
    # gather), TF32 disabled, current decode slot force-included. The decision-gate input
    # for cosine. EAGER. selector_impl="reference_cosine".
    [ -s "$MASK" ] || { echo "FATAL: mask $MASK missing"; exit 2; }
    DS_CONFIG=$(printf '{"top_k": 2048, "page_size": 64, "channel_mask_path": "%s", "device_buffer_size": 4096, "scorer_norm": "off", "head_agg": "max", "anchor_mode": "off", "anchor_budget": 0, "enable_lifted_budget_decode": false, "lifted_budget_top_k": 0, "selector_impl": "reference_cosine", "reference_include_current": true}' "$MASK")
    EXTRA=( --disable-radix-cache --disable-cuda-graph --enable-double-sparsity --double-sparsity-config "$DS_CONFIG" )
    ;;
  ref_cosine_noinc)
    # Single-variable bisection arm: EXACTLY ref_cosine with the ONE production-vs-faithful
    # variable flipped -> reference_include_current=false (production excludes the current decode
    # slot). Same cosine scorer, same head_agg=max, same exact-fp32/TF32-off path. Isolates whether
    # the sparse cosine recovery survives the production current-slot exclusion (sparse expected to
    # stay high -> scorer-driven, current-slot-independent; dense expected to drop -> current-slot is
    # the dense variable). EAGER.
    [ -s "$MASK" ] || { echo "FATAL: mask $MASK missing"; exit 2; }
    DS_CONFIG=$(printf '{"top_k": 2048, "page_size": 64, "channel_mask_path": "%s", "device_buffer_size": 4096, "scorer_norm": "off", "head_agg": "max", "anchor_mode": "off", "anchor_budget": 0, "enable_lifted_budget_decode": false, "lifted_budget_top_k": 0, "selector_impl": "reference_cosine", "reference_include_current": false}' "$MASK")
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
  ds_forced_all_assert)
    # Same forced-all dense control, EAGER, with forced_all_assert -> dump the post-adapter physical
    # slots per (rank,req,layer) for the AC-2.1 downstream-isolation assertions (guarded diagnostic,
    # host-side copy only). Capture dir defaults to CWD/.sglang_ds_forcedall; override via
    # SGLANG_DS_FORCEDALL_ASSERT_DIR.
    [ -s "$MASK" ] || { echo "FATAL: mask $MASK missing"; exit 2; }
    DS_CONFIG=$(printf '{"top_k": 2048, "page_size": 64, "channel_mask_path": "%s", "device_buffer_size": 4096, "scorer_norm": "off", "head_agg": "max", "anchor_mode": "off", "anchor_budget": 0, "enable_lifted_budget_decode": false, "lifted_budget_top_k": 0, "forced_all_dense_control": true, "forced_all_assert": true}' "$MASK")
    EXTRA=( --disable-radix-cache --disable-cuda-graph --enable-double-sparsity --double-sparsity-config "$DS_CONFIG" )
    ;;
  ds_garbage)
    # Production SCORED DS (NO forced-all override), EAGER, with forced_all_assert -> the diagnostic
    # dumps the post-adapter physical slots + _ds_slot_written bits for the REAL scored top-k selection,
    # so an offline reducer can compute the AC-4 length-cap garbage-rate (duplicate / -1 / unwritten /
    # out-of-range / adapter-error) on the actual production selection. Capture dir via
    # SGLANG_DS_FORCEDALL_ASSERT_DIR.
    [ -s "$MASK" ] || { echo "FATAL: mask $MASK missing"; exit 2; }
    DS_CONFIG=$(printf '{"top_k": 2048, "page_size": 64, "channel_mask_path": "%s", "device_buffer_size": 4096, "scorer_norm": "off", "head_agg": "max", "anchor_mode": "off", "anchor_budget": 0, "enable_lifted_budget_decode": false, "lifted_budget_top_k": 0, "forced_all_assert": true}' "$MASK")
    EXTRA=( --disable-radix-cache --disable-cuda-graph --enable-double-sparsity --double-sparsity-config "$DS_CONFIG" )
    ;;
  ref_faithful_garbage)
    # AC-4 garbage counters for the FAITHFUL raw-dot reference: ref_faithful config (reference_rawdot,
    # current slot INCLUDED) + forced_all_assert -> the same hook (deepseek_v2.py:2722, gated only on
    # forced_all_assert) dumps the reference scored selection's post-adapter physical slots + _ds_slot_written
    # bits, so the offline reducer can compute the AC-4 length-cap garbage-rate for this arm. Because the
    # reference INCLUDES the current slot, current_slot_unwritten is expected NONZERO (vs production's 0).
    # EAGER. Capture dir via SGLANG_DS_FORCEDALL_ASSERT_DIR.
    [ -s "$MASK" ] || { echo "FATAL: mask $MASK missing"; exit 2; }
    DS_CONFIG=$(printf '{"top_k": 2048, "page_size": 64, "channel_mask_path": "%s", "device_buffer_size": 4096, "scorer_norm": "off", "head_agg": "max", "anchor_mode": "off", "anchor_budget": 0, "enable_lifted_budget_decode": false, "lifted_budget_top_k": 0, "selector_impl": "reference_rawdot", "reference_include_current": true, "forced_all_assert": true}' "$MASK")
    EXTRA=( --disable-radix-cache --disable-cuda-graph --enable-double-sparsity --double-sparsity-config "$DS_CONFIG" )
    ;;
  ref_cosine_garbage)
    # AC-4 garbage counters for the FAITHFUL COSINE reference: ref_cosine config (reference_cosine,
    # current slot INCLUDED) + forced_all_assert. Same hook/dump as ref_faithful_garbage; current_slot_unwritten
    # expected NONZERO. EAGER. Capture dir via SGLANG_DS_FORCEDALL_ASSERT_DIR.
    [ -s "$MASK" ] || { echo "FATAL: mask $MASK missing"; exit 2; }
    DS_CONFIG=$(printf '{"top_k": 2048, "page_size": 64, "channel_mask_path": "%s", "device_buffer_size": 4096, "scorer_norm": "off", "head_agg": "max", "anchor_mode": "off", "anchor_budget": 0, "enable_lifted_budget_decode": false, "lifted_budget_top_k": 0, "selector_impl": "reference_cosine", "reference_include_current": true, "forced_all_assert": true}' "$MASK")
    EXTRA=( --disable-radix-cache --disable-cuda-graph --enable-double-sparsity --double-sparsity-config "$DS_CONFIG" )
    ;;
  ds_anchor)
    # Sparse-regime current/recent-slot confirmation: production top-2048 selection PLUS a recency
    # anchor that force-includes the most-recent ANCHOR_BUDGET slots (incl. the
    # current decode slot) on top of top-k. If sparse recovers, the current/recent
    # slot exclusion is the sparse bug too. ANCHOR_BUDGET env (default 64).
    [ -s "$MASK" ] || { echo "FATAL: mask $MASK missing"; exit 2; }
    AB="${ANCHOR_BUDGET:-64}"
    DS_CONFIG=$(printf '{"top_k": 2048, "page_size": 64, "channel_mask_path": "%s", "device_buffer_size": 4096, "scorer_norm": "off", "head_agg": "max", "anchor_mode": "recency", "anchor_budget": %s, "enable_lifted_budget_decode": false, "lifted_budget_top_k": 0}' "$MASK" "$AB")
    EXTRA=( --disable-radix-cache --enable-double-sparsity --double-sparsity-config "$DS_CONFIG" )
    ;;
  *) echo "FATAL: mode must be 'dsa', 'dsa_noradix', 'ds', 'ds_capture', 'ref', 'ref_faithful', 'ref_cosine', 'ref_cosine_noinc', 'ds_forced_all', 'ds_forced_all_assert', 'ds_garbage', 'ref_faithful_garbage', 'ref_cosine_garbage', or 'ds_anchor'"; exit 2 ;;
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
