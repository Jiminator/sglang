# Running and evaluating Double Sparsity v2 — operator runbook

Step-by-step instructions to **serve** table-free Double Sparsity (DS) on
GLM-5.1-FP8, **prove DS is genuinely active**, and **run the concurrency-64
performance eval**. The shipping code is the `double-sparsity-v2` branch on
`Jiminator/sglang` (cut from `origin/main`
`<BASE>=105e095e005d02a178fb6c5a23bd22ba644c90e4`).

The measured result this runbook reproduces is **35.05 p50 decode TPS / 22.90 s
P99 TTFT** at concurrency 64 — inside the loop-11b parity band (≥24.2 TPS /
≤30.1 s). See `V2_PERFORMANCE.md` for the numbers and caveats; this doc is the
**how-to-run**.

---

## 0. Prerequisites (one-time)

Hardware/model used for the reference run: **8×H200, TP=8, GLM-5.1-FP8**.

In this environment the pieces are already in place at these paths — substitute
your own if reproducing elsewhere:

```bash
# Shipping code (the double-sparsity-v2 branch checked out here). The editable
# install points at the dev clone, so v2 code is run via PYTHONPATH.
export V2=/sgl-workspace/double-sparisty-v2/sglang

# GLM-5.1-FP8 weights (HF snapshot dir).
export MODEL=/cluster-storage/models/models--zai-org--GLM-5.1-FP8/snapshots/f396cf805182f4ca10fa675e1a99815b3ca384db

# Calibrated channel mask (provenance + SHA in benchmarks/DOUBLE_SPARSITY.md).
export MASK=/cluster-storage/models/glm51-fp8-channel-mask-loop12.safetensors

export HOST=127.0.0.1 PORT=30000
export EVID=/sgl-workspace/sglang/development/loop12/perf_evidence
export LOG=/sgl-workspace/sglang/development/loop12/serve_ds_boot.log
```

If reproducing from scratch, check out the branch instead of using `$V2`:

```bash
git clone -b double-sparsity-v2 https://github.com/Jiminator/sglang.git
```

**Kernel floor.** Latest `origin/main` uses the `sglang-kernel >= 0.4.4`
flash-attention `only_qv` path. Install it (prebuilt abi3 wheel) — this is a
base-code requirement, not a DS dependency:

```bash
pip install sglang-kernel==0.4.4
python -c "import importlib.metadata as m; print('sglang-kernel', m.version('sglang-kernel'))"
# expect: sglang-kernel 0.4.4
```

**Operational guards (non-negotiable).**
- **Never** set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments` for serving — it
  breaks custom all-reduce IPC at TP=8. (Calibration only, in a separate
  process; see §1.)
- Run **one** TP=8 server at a time. Tear down (§6) and wait for the GPUs to go
  idle before launching another.

Confirm a clean start (all 8 GPUs at ~0 MiB, no server bound to `$PORT`):

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
curl -sf "http://$HOST:$PORT/health" && echo "PORT BUSY — tear down first" || echo "port free"
```

---

## 1. (Optional) Calibrate the channel mask

The mask is model- and quant-specific. The reference mask is already calibrated
(`$MASK`); regenerate only if you change the model/quant. This is the **only**
step allowed to set `expandable_segments`, and it must be a **separate process**
from serving:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH="$V2/python" python -m sglang.srt.layers.attention.double_sparsity.calibrate \
  --model "$MODEL" \
  --dtype fp8_e4m3 --kv-cache-dtype fp8_e4m3 --tp 8 \
  --label-dim 32 --page-size 64 --num-samples 256 --block-size 512 --seed 42 \
  --dataset <calibration corpus, one document per line> \
  --output "$MASK"
```

Mask content SHA-256 of the reference mask:
`35155ac46ad79fa82e531138434ff35708e2d8c2932889323a21a455342a9b00`.

---

## 2. Launch the DS server

Radix cache **on** (no fixture artifact, no override), CUDA graphs **on**.
Launched in the background with a PID file so the later steps and teardown are
independent shells.

```bash
[ -s "$MASK" ] || { echo "FATAL: mask $MASK missing"; exit 2; }
case "${PYTORCH_CUDA_ALLOC_CONF:-}" in *expandable_segments*) echo "FATAL: expandable_segments set for serving"; exit 3;; esac

DS_CONFIG=$(printf '{"top_k": 2048, "page_size": 64, "channel_mask_path": "%s", "device_buffer_size": 4096, "scorer_norm": "off", "head_agg": "max", "anchor_mode": "off", "anchor_budget": 0, "enable_lifted_budget_decode": false, "lifted_budget_top_k": 0}' "$MASK")

PYTHONPATH="$V2/python" nohup python3 -m sglang.launch_server \
  --model-path "$MODEL" --host "$HOST" --port "$PORT" \
  --tp-size 8 --kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.8 \
  --max-running-requests 64 --cuda-graph-max-bs 64 --page-size 64 \
  --dsa-prefill-backend flashmla_kv --dsa-decode-backend flashmla_kv \
  --disable-overlap-schedule --disable-piecewise-cuda-graph \
  --enable-double-sparsity --double-sparsity-config "$DS_CONFIG" \
  --random-seed 42 --trust-remote-code > "$LOG" 2>&1 &
echo $! > /tmp/ds_server.pid
echo "server PID=$(cat /tmp/ds_server.pid)"
```

Wait for readiness (model load + per-layer mask bind + graph capture; up to
~12 min):

```bash
for i in $(seq 1 144); do
  curl -sf "http://$HOST:$PORT/health" >/dev/null 2>&1 && { echo READY; break; }
  kill -0 "$(cat /tmp/ds_server.pid)" 2>/dev/null || { echo "SERVER DIED"; tail -40 "$LOG"; break; }
  sleep 5
done
```

Confirm the mask bound at startup:

```bash
grep -iE "bind shape check passed|bind_runtime_data completed" "$LOG" | head -2
```

---

## 3. Verify DS is genuinely active (not a silent dense fallback)

Send a **long-context** request (sequence > `top_k`=2048 so `selected < total`)
and inspect `meta_info["double_sparsity"]`. DS is active iff
`selected_tokens > 0`, `total_tokens > selected_tokens`, and
`dense_fallback == 0`:

```bash
LONG=$(python3 -c 'print("The quick brown fox jumps over the lazy dog near the riverbank at dawn. " * 350)')
RESP=$(curl -sf "http://$HOST:$PORT/generate" -H 'Content-Type: application/json' \
  -d "$(python3 -c 'import json,sys; print(json.dumps({"text": sys.argv[1], "sampling_params": {"max_new_tokens": 32, "temperature": 0}}))' "$LONG Summarize the above.")")
echo "$RESP" | python3 -c '
import sys, json
ds = json.load(sys.stdin).get("meta_info", {}).get("double_sparsity")
print(json.dumps(ds, indent=2) if ds is not None else "<<< NO double_sparsity >>>")
ok = (ds and ds.get("selected_tokens",0) > 0
      and ds.get("total_tokens",0) > ds.get("selected_tokens",0)
      and ds.get("dense_fallback",1) == 0)
print("DS_ACTIVE_PASS" if ok else "DS_ACTIVE_FAIL")
sys.exit(0 if ok else 1)
'
```

---

## 4. Run the concurrency-64 perf eval

A thin wrapper over stock `bench_serving` pins the loop-11b workload shape and
derives the gated metrics. It pins **one** shared-prefix group
(`--gsp-num-groups 1 --gsp-prompts-per-group <num_prompts>`) so all prompts
share the one system prompt; the stock default would otherwise be 64 groups × 16
and ignore `--num-prompts`.

```bash
PYTHONPATH="$V2/python" python3 "$V2/benchmarks/bench_double_sparsity.py" \
  --model "$MODEL" --host "$HOST" --port "$PORT" \
  --num-prompts 256 --seed 42 --evidence-dir "$EVID"
```

Workload: `generated-shared-prefix`, gsp 2253 / 1843 (ISL ≈4096), OSL 512,
range-ratio 1.0, max-concurrency 64, one trial, seed 42.

---

## 5. Read the verdict

```bash
python3 -m json.tool "$EVID/verdict.json"
```

PASS criteria (the wrapper exits 0 iff all hold):
- `request_shape_ok == true` and `actual_completed == num_prompts` (256/256) —
  fails closed otherwise,
- `p50_decode_tps >= 24.21` (loop-11b 26.9 − 10%),
- `p99_ttft_s <= 30.12` (loop-11b 25.1 + 20%),
- `parity == true`.

Reference accepted run: `p50_decode_tps 35.053`, `p99_ttft_s 22.901`,
`actual_completed 256`, `parity true`.

---

## 6. Tear down (always, before any next launch)

```bash
PID=$(cat /tmp/ds_server.pid 2>/dev/null)
[ -n "$PID" ] && kill "$PID" 2>/dev/null
for i in $(seq 1 60); do kill -0 "$PID" 2>/dev/null || break; sleep 2; done
kill -9 "$PID" 2>/dev/null; rm -f /tmp/ds_server.pid
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader   # expect ~0 MiB on all 8
```

Do **not** blanket-kill GPU PIDs via `nvidia-smi`, and do **not** `pkill -f` a
pattern that could match your parent shell. Kill the tracked PID only.

---

## Appendix — native-DSA sanity (DS off)

To measure native DSA on the same base for context, drop
`--enable-double-sparsity` / `--double-sparsity-config` from §2 (keep everything
else) and re-run §4. On this base native DSA measured ~26.06 TPS / ~46.50 s P99
TTFT — **same-base context only**, not a corrected-shape pass/fail baseline (it
predates the GSP-grouping pin). The high TTFT is base drift (triton-3.6.0 MoE
tuning-config fallback), not DS-specific.
