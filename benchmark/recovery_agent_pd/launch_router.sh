#!/usr/bin/env bash
# PD router with consistent hashing on both pools: every session's
# X-SMG-Routing-Key (sent by the recovery-agent benchmark client) pins the
# session to one prefill and one decode worker.
#
#   PREFILL_URLS="http://host1:30001 http://host2:30001" \
#   PREFILL_BOOTSTRAP_PORTS="8998 8998" \
#   DECODE_URLS="http://host3:30011 http://host4:30011" \
#     bash launch_router.sh
set -euo pipefail

PREFILL_URLS=${PREFILL_URLS:-"http://127.0.0.1:30001 http://127.0.0.1:30002"}
PREFILL_BOOTSTRAP_PORTS=${PREFILL_BOOTSTRAP_PORTS:-"8998 8999"}
DECODE_URLS=${DECODE_URLS:-"http://127.0.0.1:30011 http://127.0.0.1:30012"}
ROUTER_PORT=${ROUTER_PORT:-30000}

prefill_args=()
read -r -a urls <<<"$PREFILL_URLS"
read -r -a ports <<<"$PREFILL_BOOTSTRAP_PORTS"
for i in "${!urls[@]}"; do
  prefill_args+=(--prefill "${urls[$i]}" "${ports[$i]}")
done
decode_args=()
for url in $DECODE_URLS; do
  decode_args+=(--decode "$url")
done

exec python3 -m sglang_router.launch_router \
  --pd-disaggregation \
  --policy consistent_hashing \
  --prefill-policy consistent_hashing \
  --decode-policy consistent_hashing \
  "${prefill_args[@]}" \
  "${decode_args[@]}" \
  --host 0.0.0.0 --port "$ROUTER_PORT"
