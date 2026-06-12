#!/usr/bin/env bash
# Shared env + helpers for the 20260612 DS-vs-DSA re-measurement on current HEAD
# (Loop-10 landed: width-bucketed selector graphs, compact W=5120, bf16-authoritative
# top-k, pinned two-shot score reduce). Mirrors runs/20260609/_env.sh op-point exactly
# so the new numbers are directly comparable to the frozen Loop-8 profiling baseline.
set -uo pipefail

GLM=/cluster-storage/models/models--zai-org--GLM-5.1-FP8/snapshots/f396cf805182f4ca10fa675e1a99815b3ca384db
MASK=/models/glm51-fp8-channel-mask-s256.safetensors
OUT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT=30000

# critical: expandable_segments breaks custom-all-reduce IPC at GLM TP=8 (BL-20260608)
unset PYTORCH_CUDA_ALLOC_CONF || true

DS_CONFIG='{"top_k": 2048, "page_size": 64, "channel_mask_path": "/models/glm51-fp8-channel-mask-s256.safetensors", "device_buffer_size": 4096, "signature_dtype": "fp16", "scorer_norm": "off", "scorer_norm_hybrid_threshold": 8192, "head_agg": "max", "anchor_mode": "off", "anchor_budget": 0, "recall_oracle": false, "enable_lifted_budget_decode": false, "lifted_budget_top_k": 0}'

COMMON_ARGS=(--model-path "$GLM" --tp-size 8 --kv-cache-dtype fp8_e4m3 --page-size 64
  --dsa-prefill-backend flashmla_kv --dsa-decode-backend flashmla_kv
  --disable-overlap-schedule --disable-piecewise-cuda-graph --disable-radix-cache
  --random-seed 20260607 --trust-remote-code --host 127.0.0.1 --port "$PORT")

# build_server_args <case> -> SERVER_ARGS
#   ds07   : Double Sparsity ON, mem 0.7   (Case 1; DS serving + DS profiling)
#   dsa07  : DSA native, mem 0.7           (Case 2; apples-to-apples same bs/mem)
#   dsa08  : DSA native, mem 0.8           (Case 3 / DSA serving best)
build_server_args() {
  case "$1" in
    ds07)  SERVER_ARGS=("${COMMON_ARGS[@]}" --mem-fraction-static 0.7 --enable-double-sparsity --double-sparsity-config "$DS_CONFIG") ;;
    dsa07) SERVER_ARGS=("${COMMON_ARGS[@]}" --mem-fraction-static 0.7) ;;
    dsa08) SERVER_ARGS=("${COMMON_ARGS[@]}" --mem-fraction-static 0.8) ;;
    *) echo "unknown case $1" >&2; return 1 ;;
  esac
}

wait_ready() {  # poll /health up to ~8 min (TP=8 GLM cold boot + graph capture)
  for i in $(seq 1 48); do
    curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && { echo "ready ~$((i*10))s"; return 0; }
    sleep 10
  done
  return 1
}

# max admittable decode batch at ISL 4096 / OSL 512 = floor(KV_capacity / (4096+512))
max_batch_from_server() {
  curl -s --max-time 8 "http://127.0.0.1:${PORT}/get_server_info" 2>/dev/null | python3 -c "
import json,sys
try:
    d=json.load(sys.stdin)
except Exception:
    print(0); sys.exit()
# /get_server_info nests this under top level or internal_states; check both
cap=d.get('max_total_num_tokens')
if cap is None:
    ist=d.get('internal_states')
    if isinstance(ist,list) and ist and isinstance(ist[0],dict):
        cap=ist[0].get('max_total_num_tokens')
    elif isinstance(ist,dict):
        cap=ist.get('max_total_num_tokens')
print(int(cap) if cap else 0)
"
}

teardown() {
  pkill -f "sglang.launch_server" 2>/dev/null || true
  pkill -f "sglang::scheduler"   2>/dev/null || true
  sleep 20
  rm -f /dev/shm/psm_* /dev/shm/sem.mp-* 2>/dev/null || true
}
