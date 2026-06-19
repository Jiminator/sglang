#!/usr/bin/env bash
# Loop-12 diagnostic: native DSA (DS OFF) on the SAME v2 latest-main base at
# conc 64. Isolates the DS port from base-environment drift — if DSA lands near
# DS's number, the port preserved behaviour and the absolute drop vs loop-11b is
# the base (triton 3.6.0 MoE config fallback, kernel bumps), not the port.
set -uo pipefail
V2=/sgl-workspace/double-sparisty-v2/sglang
MODEL=/cluster-storage/models/models--zai-org--GLM-5.1-FP8/snapshots/f396cf805182f4ca10fa675e1a99815b3ca384db
HOST=127.0.0.1; PORT="${PORT:-30000}"
LOG=/sgl-workspace/sglang/development/loop12/serve_dsa_baseline.log
EVID=/sgl-workspace/sglang/development/loop12/dsa_evidence
mkdir -p "$EVID"

echo ">>> booting NATIVE DSA (no DS) — same base, same knobs"
PYTHONPATH="$V2/python" nohup python3 -m sglang.launch_server \
  --model-path "$MODEL" --host "$HOST" --port "$PORT" \
  --tp-size 8 --kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.8 \
  --max-running-requests 64 --cuda-graph-max-bs 64 --page-size 64 \
  --dsa-prefill-backend flashmla_kv --dsa-decode-backend flashmla_kv \
  --disable-overlap-schedule --disable-piecewise-cuda-graph \
  --random-seed 42 --trust-remote-code > "$LOG" 2>&1 &
SERVER_PID=$!
echo "server PID=$SERVER_PID"
trap 'kill "$SERVER_PID" 2>/dev/null; for i in $(seq 1 60); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 2; done; kill -9 "$SERVER_PID" 2>/dev/null' EXIT

for i in $(seq 1 144); do
  curl -sf "http://$HOST:$PORT/health" >/dev/null 2>&1 && break
  kill -0 "$SERVER_PID" 2>/dev/null || { echo "DSA SERVER DIED"; tail -30 "$LOG"; exit 4; }
  sleep 5
done
echo ">>> DSA server READY"
PYTHONPATH="$V2/python" python3 "$V2/benchmarks/bench_double_sparsity.py" \
  --model "$MODEL" --host "$HOST" --port "$PORT" \
  --num-prompts 256 --seed 42 --evidence-dir "$EVID"
echo ">>> DSA baseline done rc=$?"
