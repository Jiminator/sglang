#!/usr/bin/env bash
# Shared environment + helpers for the Loop-8 DS-vs-DSA one-batch profiling runs.
# Sourced by run_case.sh. See development/profiling/plan.md.
set -uo pipefail

GLM=/cluster-storage/models/models--zai-org--GLM-5.1-FP8/snapshots/f396cf805182f4ca10fa675e1a99815b3ca384db
MASK=/models/glm51-fp8-channel-mask-s256.safetensors
OUT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # absolute .../runs/20260609

# critical: keep custom all-reduce working (expandable_segments breaks the IPC handles)
unset PYTORCH_CUDA_ALLOC_CONF || true

# DS config (Case 1) — matches serve_double_sparsity.sh defaults
DS_CONFIG='{"top_k": 2048, "page_size": 64, "channel_mask_path": "/models/glm51-fp8-channel-mask-s256.safetensors", "device_buffer_size": 4096, "signature_dtype": "fp16", "scorer_norm": "off", "scorer_norm_hybrid_threshold": 8192, "head_agg": "max", "anchor_mode": "off", "anchor_budget": 0, "recall_oracle": false, "enable_lifted_budget_decode": false, "lifted_budget_top_k": 0}'

# COMMON: identical for all three cases
COMMON_ARGS=(--model-path "$GLM" --tp-size 8 --kv-cache-dtype fp8_e4m3 --page-size 64
  --dsa-prefill-backend flashmla_kv --dsa-decode-backend flashmla_kv
  --disable-overlap-schedule --disable-piecewise-cuda-graph --disable-radix-cache
  --random-seed 20260607 --trust-remote-code --host 127.0.0.1 --port 30000)

# Build SERVER_ARGS array for a given case name into the global SERVER_ARGS.
build_server_args() {
  case "$1" in
    case1) SERVER_ARGS=("${COMMON_ARGS[@]}" --mem-fraction-static 0.7 --enable-double-sparsity --double-sparsity-config "$DS_CONFIG") ;;
    case2) SERVER_ARGS=("${COMMON_ARGS[@]}" --mem-fraction-static 0.7) ;;
    case3) SERVER_ARGS=("${COMMON_ARGS[@]}" --mem-fraction-static 0.8) ;;
    *) echo "unknown case $1" >&2; return 1 ;;
  esac
}

wait_ready() {  # poll /health up to ~6 min
  for i in $(seq 1 36); do
    curl -sf http://127.0.0.1:30000/health >/dev/null 2>&1 && { echo "ready ~$((i*10))s"; return 0; }
    sleep 10
  done
  return 1
}

teardown() {
  pkill -f "sglang.launch_server" 2>/dev/null || true
  pkill -f "sglang::scheduler"   2>/dev/null || true
  sleep 20
  rm -f /dev/shm/psm_* /dev/shm/sem.mp-* 2>/dev/null || true
}
