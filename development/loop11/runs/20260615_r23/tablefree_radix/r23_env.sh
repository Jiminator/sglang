#!/usr/bin/env bash
# Loop 11 R23 — table-free radix SERVING validation on the SOUND fail-closed gate
# (HEAD 49a401a72). Boots DS radix-ON via the dev override SGLANG_DS_RADIX_OVERRIDE=1
# (drops --disable-radix-cache). NEW config schema; the removed fields
# signature_dtype / scorer_norm_hybrid_threshold / table_free are rejected by the
# parser, so they are NOT present here.
set -uo pipefail

GLM=/cluster-storage/models/models--zai-org--GLM-5.1-FP8/snapshots/f396cf805182f4ca10fa675e1a99815b3ca384db
MASK=/models/glm51-fp8-channel-mask-s256.safetensors
PORT=30000

# critical: expandable_segments breaks custom-all-reduce-v2 IPC at GLM TP=8.
unset PYTORCH_CUDA_ALLOC_CONF || true

# Base op-point COMMON args MINUS --disable-radix-cache (radix-ON for the probes).
COMMON_ARGS=(--model-path "$GLM" --tp-size 8 --kv-cache-dtype fp8_e4m3 --page-size 64
  --dsa-prefill-backend flashmla_kv --dsa-decode-backend flashmla_kv
  --disable-overlap-schedule --disable-piecewise-cuda-graph
  --random-seed 20260607 --trust-remote-code --host 127.0.0.1 --port "$PORT"
  --mem-fraction-static 0.8 --max-running-requests 64 --cuda-graph-max-bs 64)

# DSA-default args (NO --enable-double-sparsity) for AC-7 — uses the SHIPPED
# default which KEEPS radix cache ON by default (no --disable-radix-cache).
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

gpu_idle_wait() {  # wait until all GPUs ~0 MiB
  for i in $(seq 1 30); do
    local mx
    mx=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
    if [[ "${mx:-99999}" -lt 500 ]]; then echo "gpus idle (max=${mx}MiB)"; return 0; fi
    sleep 5
  done
  echo "gpus still busy (max=${mx}MiB)"; return 1
}
