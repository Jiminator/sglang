#!/usr/bin/env bash
# Loop 11 M2 R21 — shared env for the SHIPPED table-free DS path re-proof on the
# committed post-deletion HEAD (TokenLabelTable deleted; table-free absorbed-latent
# selection is the ONE unconditional DS selection path). NEW config schema: the
# fields signature_dtype / scorer_norm_hybrid_threshold / table_free were REMOVED
# and the parser now REJECTS them. We boot DS DIRECTLY via launch_server with the
# new DS_CONFIG (no serve_double_sparsity.sh launcher, no TABLE_FREE flag).
set -uo pipefail

GLM=/cluster-storage/models/models--zai-org--GLM-5.1-FP8/snapshots/f396cf805182f4ca10fa675e1a99815b3ca384db
MASK=/models/glm51-fp8-channel-mask-s256.safetensors
PORT=30000

# critical: expandable_segments breaks custom-all-reduce-v2 IPC at GLM TP=8.
unset PYTORCH_CUDA_ALLOC_CONF || true

# NEW table-free DS_CONFIG — only currently-allowed fields. NO signature_dtype,
# NO scorer_norm_hybrid_threshold, NO table_free (table-free is unconditional now).
DS_CONFIG='{"top_k": 2048, "page_size": 64, "channel_mask_path": "/models/glm51-fp8-channel-mask-s256.safetensors", "device_buffer_size": 4096, "scorer_norm": "off", "head_agg": "max", "anchor_mode": "off", "anchor_budget": 0, "recall_oracle": false, "enable_lifted_budget_decode": false, "lifted_budget_top_k": 0}'

# Shared op-point. mem 0.8 / right-sized envelope (max-running 64, cuda-graph-max-bs 64).
COMMON_ARGS=(--model-path "$GLM" --tp-size 8 --kv-cache-dtype fp8_e4m3 --page-size 64
  --dsa-prefill-backend flashmla_kv --dsa-decode-backend flashmla_kv
  --disable-overlap-schedule --disable-piecewise-cuda-graph --disable-radix-cache
  --random-seed 20260607 --trust-remote-code --host 127.0.0.1 --port "$PORT"
  --mem-fraction-static 0.8 --max-running-requests 64 --cuda-graph-max-bs 64)

# DS server (table-free, unconditional) args.
DS_ARGS=("${COMMON_ARGS[@]}" --enable-double-sparsity --double-sparsity-config "$DS_CONFIG")
# DSA-default args (NO --enable-double-sparsity).
DSA_ARGS=("${COMMON_ARGS[@]}")

ready_wait() {
  local slog="$1"
  for i in $(seq 1 80); do
    if grep -aqE "SIGQUIT received|Scheduler hit an exception|CUDA out of memory|Capture cuda graph failed|Not enough host memory|ValueError|AssertionError|RuntimeError|Traceback" "$slog" 2>/dev/null; then return 2; fi
    if grep -aq "The server is fired up" "$slog" 2>/dev/null; then
      curl -sf --max-time 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && { echo "ready ~$((i*10))s"; return 0; }
    fi
    sleep 10
  done
  return 1
}

smoke() {
  curl -s --max-time 120 -X POST "http://127.0.0.1:${PORT}/generate" -H 'Content-Type: application/json' \
    -d '{"text": "The capital of France is", "sampling_params": {"max_new_tokens": 24, "temperature": 0}}' 2>/dev/null \
  | python3 -c "import json,sys
try:
 d=json.load(sys.stdin); t=(d.get('text') or '').strip().replace(chr(10),' ')
 print('OK:'+t[:60] if t else 'FAIL:empty')
except Exception as e: print('FAIL:'+str(e)[:60])"
}

teardown() {
  pkill -f "sglang.launch_server" 2>/dev/null || true
  pkill -f "sglang::scheduler"   2>/dev/null || true
  sleep 20
  rm -f /dev/shm/psm_* /dev/shm/sem.mp-* 2>/dev/null || true
}
