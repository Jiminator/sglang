#!/usr/bin/env bash
# Loop-12 M6: boot table-free Double Sparsity on GLM-5.1-FP8 from the v2 shipping
# tree and prove DS is genuinely active (meta_info["double_sparsity"]).
#
# Runs the v2 code via PYTHONPATH (the editable install points at the dev clone).
# Radix cache is ON with NO fixture artifact and NO SGLANG_DS_RADIX_OVERRIDE — the
# gate was removed. NEVER sets expandable_segments (breaks custom all-reduce at TP=8).
set -uo pipefail

V2=/sgl-workspace/double-sparisty-v2/sglang
MODEL=/cluster-storage/models/models--zai-org--GLM-5.1-FP8/snapshots/f396cf805182f4ca10fa675e1a99815b3ca384db
MASK="${MASK:-/cluster-storage/models/glm51-fp8-channel-mask-loop12.safetensors}"
HOST=127.0.0.1
PORT="${PORT:-30000}"
LOG=/sgl-workspace/sglang/development/loop12/serve_ds_boot.log
DS_CONFIG=$(printf '{"top_k": 2048, "page_size": 64, "channel_mask_path": "%s", "device_buffer_size": 4096, "scorer_norm": "off", "head_agg": "max", "anchor_mode": "off", "anchor_budget": 0, "enable_lifted_budget_decode": false, "lifted_budget_top_k": 0}' "$MASK")

[ -s "$MASK" ] || { echo "FATAL: mask $MASK missing"; exit 2; }
# Guard: expandable_segments must NOT be set for serving.
case "${PYTORCH_CUDA_ALLOC_CONF:-}" in *expandable_segments*) echo "FATAL: expandable_segments set"; exit 3;; esac

echo ">>> booting DS server (radix ON, no fixture/override); log=$LOG"
# Env note: the v2 branch is off LATEST main, which genuinely uses the
# sglang-kernel 0.4.4 flash-attention `only_qv` path; the box was upgraded
# 0.4.3 -> 0.4.4 to satisfy it (prebuilt wheel). No version-skip needed.
PYTHONPATH="$V2/python" nohup python3 -m sglang.launch_server \
  --model-path "$MODEL" --host "$HOST" --port "$PORT" \
  --tp-size 8 --kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.8 \
  --max-running-requests 64 --cuda-graph-max-bs 64 --page-size 64 \
  --dsa-prefill-backend flashmla_kv --dsa-decode-backend flashmla_kv \
  --disable-overlap-schedule --disable-piecewise-cuda-graph \
  --enable-double-sparsity --double-sparsity-config "$DS_CONFIG" \
  --random-seed 42 --trust-remote-code > "$LOG" 2>&1 &
SERVER_PID=$!
echo "server PID=$SERVER_PID"

# Wait for readiness (up to ~12 min).
ready=0
for i in $(seq 1 144); do
  if curl -sf "http://$HOST:$PORT/health" >/dev/null 2>&1 || curl -sf "http://$HOST:$PORT/get_model_info" >/dev/null 2>&1; then ready=1; break; fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo "SERVER DIED during boot"; tail -40 "$LOG"; exit 4; fi
  sleep 5
done
[ "$ready" = 1 ] || { echo "server not ready in time"; tail -40 "$LOG"; kill "$SERVER_PID" 2>/dev/null; exit 5; }
echo ">>> server READY"

# Decode request; capture meta_info["double_sparsity"].
RESP=$(curl -sf "http://$HOST:$PORT/generate" -H 'Content-Type: application/json' -d '{
  "text": "Explain double sparsity attention in two sentences.",
  "sampling_params": {"max_new_tokens": 48, "temperature": 0}
}')
echo ">>> generate response (meta_info excerpt):"
echo "$RESP" | python3 -c '
import sys, json
r = json.load(sys.stdin)
mi = r.get("meta_info", {})
ds = mi.get("double_sparsity")
print(json.dumps(ds, indent=2) if ds is not None else "<<< NO double_sparsity in meta_info >>>")
ok = (ds is not None
      and ds.get("selected_tokens", 0) > 0
      and ds.get("total_tokens", 0) > ds.get("selected_tokens", 0)
      and ds.get("dense_fallback", 1) == 0)
print("DS_ACTIVE_PASS" if ok else "DS_ACTIVE_FAIL")
sys.exit(0 if ok else 1)
'
VERDICT=$?

echo ">>> startup bind-log check:"
grep -iE "bind shape check passed|bind_runtime_data completed|double_sparsity" "$LOG" | head -6

echo ">>> tearing down server PID=$SERVER_PID"
kill "$SERVER_PID" 2>/dev/null
for i in $(seq 1 60); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 2; done
kill -9 "$SERVER_PID" 2>/dev/null
echo ">>> M6 verdict rc=$VERDICT (0=DS active)"
exit $VERDICT
