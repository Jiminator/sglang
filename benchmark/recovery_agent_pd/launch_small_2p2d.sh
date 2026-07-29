#!/usr/bin/env bash
# 2 prefill + 2 decode single-GPU workers of a small pinned chat model, all on
# one 4-GPU host. Validates PD routing, per-session affinity, and KV transfer
# before any large-model run. Start the router with launch_router.sh.
#
#   MODEL_PATH=Qwen/Qwen2.5-0.5B-Instruct MODEL_REVISION=<commit-sha> \
#     bash launch_small_2p2d.sh
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}
# Pin the exact snapshot so the topology test is reproducible.
MODEL_REVISION=${MODEL_REVISION:-7ae557604adf67be50417f59c2c2f167def9a775}
HOST=${HOST:-0.0.0.0}
LOG_DIR=${LOG_DIR:-/tmp/recovery_agent_pd_small}

common_args=(
  --model-path "$MODEL_PATH"
  --revision "$MODEL_REVISION"
  --host "$HOST"
  --disaggregation-transfer-backend nixl # v0.5.16 defaults to mooncake
  --mem-fraction-static 0.7
)

launch_worker() {
  local label=$1 gpu=$2
  shift 2
  if [ "${DRY_RUN:-0}" = 1 ]; then
    printf '#COMMAND small:%s\n' "$label"
    printf '%s\n' "$@"
    return 0
  fi
  CUDA_VISIBLE_DEVICES=$gpu python3 -m sglang.launch_server "$@" \
    > "$LOG_DIR/$label.log" 2>&1 &
}

[ "${DRY_RUN:-0}" = 1 ] || mkdir -p "$LOG_DIR"

# Prefill pool: ports 30001/30002, bootstrap ports 8998/8999.
launch_worker prefill0 0 "${common_args[@]}" \
  --disaggregation-mode prefill --port 30001 --disaggregation-bootstrap-port 8998
launch_worker prefill1 1 "${common_args[@]}" \
  --disaggregation-mode prefill --port 30002 --disaggregation-bootstrap-port 8999

# Decode pool: ports 30011/30012.
launch_worker decode0 2 "${common_args[@]}" --disaggregation-mode decode --port 30011
launch_worker decode1 3 "${common_args[@]}" --disaggregation-mode decode --port 30012

if [ "${DRY_RUN:-0}" = 1 ]; then
  exit 0
fi

echo "workers launching; logs in $LOG_DIR"
wait
