#!/usr/bin/env bash
# Loop-12 M6 + M8 in one boot: prove DS genuinely active on a LONG-context
# request (seq > top_k so selected < total), then run the conc-64 perf parity
# eval against the same live server. One model load.
set -uo pipefail

V2=/sgl-workspace/double-sparisty-v2/sglang
MODEL=/cluster-storage/models/models--zai-org--GLM-5.1-FP8/snapshots/f396cf805182f4ca10fa675e1a99815b3ca384db
MASK="${MASK:-/cluster-storage/models/glm51-fp8-channel-mask-loop12.safetensors}"
HOST=127.0.0.1; PORT="${PORT:-30000}"
LOG=/sgl-workspace/sglang/development/loop12/serve_ds_boot.log
EVID=/sgl-workspace/sglang/development/loop12/perf_evidence
DS_CONFIG=$(printf '{"top_k": 2048, "page_size": 64, "channel_mask_path": "%s", "device_buffer_size": 4096, "scorer_norm": "off", "head_agg": "max", "anchor_mode": "off", "anchor_budget": 0, "enable_lifted_budget_decode": false, "lifted_budget_top_k": 0}' "$MASK")

[ -s "$MASK" ] || { echo "FATAL: mask $MASK missing"; exit 2; }
case "${PYTORCH_CUDA_ALLOC_CONF:-}" in *expandable_segments*) echo "FATAL: expandable_segments set"; exit 3;; esac
mkdir -p "$EVID"

echo ">>> booting DS server (radix ON, graphs ON, no fixture/override); log=$LOG"
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

cleanup() { echo ">>> teardown PID=$SERVER_PID"; kill "$SERVER_PID" 2>/dev/null; for i in $(seq 1 60); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 2; done; kill -9 "$SERVER_PID" 2>/dev/null; }
trap cleanup EXIT

ready=0
for i in $(seq 1 144); do
  if curl -sf "http://$HOST:$PORT/health" >/dev/null 2>&1 || curl -sf "http://$HOST:$PORT/get_model_info" >/dev/null 2>&1; then ready=1; break; fi
  kill -0 "$SERVER_PID" 2>/dev/null || { echo "SERVER DIED during boot"; tail -40 "$LOG"; exit 4; }
  sleep 5
done
[ "$ready" = 1 ] || { echo "server not ready"; tail -40 "$LOG"; exit 5; }
echo ">>> server READY"

# ---- M6: DS genuinely active on a LONG context (>2048 tokens so selected<total) ----
# Build a ~3000-token prompt by repeating a sentence.
LONG=$(python3 -c 'print(("The quick brown fox jumps over the lazy dog near the riverbank at dawn. " * 350))')
RESP=$(curl -sf "http://$HOST:$PORT/generate" -H 'Content-Type: application/json' \
  -d "$(python3 -c 'import json,sys; print(json.dumps({"text": sys.argv[1], "sampling_params": {"max_new_tokens": 32, "temperature": 0}}))' "$LONG Summarize the above.")")
echo ">>> long-context meta_info[double_sparsity]:"
echo "$RESP" | python3 -c '
import sys, json
r = json.load(sys.stdin)
ds = r.get("meta_info", {}).get("double_sparsity")
print(json.dumps(ds, indent=2) if ds is not None else "<<< NO double_sparsity >>>")
ok = (ds is not None and ds.get("selected_tokens",0) > 0
      and ds.get("total_tokens",0) > ds.get("selected_tokens",0)
      and ds.get("dense_fallback",1) == 0)
sys.exit(0 if ok else 1)
'
M6=$?
echo ">>> bind-log check:"; grep -iE "bind shape check passed|bind_runtime_data completed" "$LOG" | head -2
echo ">>> M6 (DS active on long context) rc=$M6 (0=pass)"
[ "$M6" = 0 ] || { echo "M6 FAILED — not running perf"; exit 6; }

# ---- M8: conc-64 perf parity against the live server ----
echo ">>> M8: conc-64 perf eval"
PYTHONPATH="$V2/python" python3 "$V2/benchmarks/bench_double_sparsity.py" \
  --model "$MODEL" --host "$HOST" --port "$PORT" \
  --num-prompts 256 --seed 42 --evidence-dir "$EVID"
M8=$?
echo ">>> M8 (perf parity) rc=$M8 (0=within band)"
exit $M8
