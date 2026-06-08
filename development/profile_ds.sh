#!/usr/bin/env bash
# Capture a Double-Sparsity decode profile to satisfy the AC-11 comparator's
# profiling obligation for a failing concurrency row: it characterizes WHERE the
# DS decode time goes (the index/scoring overhead behind the throughput gap) so a
# DS-vs-DSA TPS miss is published with a captured profile, not just a number.
#
# Boots the DS server (GLM-5.1 by default) with SGLANG_TORCH_PROFILER_DIR set,
# runs a short profiled bench burst at a failing concurrency, then summarizes the
# torch-profiler trace into the top GPU kernels by total time.
#
#   PROFILE_CONC=32 bash development/profile_ds.sh
set -euo pipefail   # fail closed: a failed boot/bench/empty-trace must NOT yield a "DONE" with no profile
cd "$(dirname "$0")/.."

GLM="${MODEL_PATH:-/cluster-storage/models/models--zai-org--GLM-5.1-FP8/snapshots/f396cf805182f4ca10fa675e1a99815b3ca384db}"
MASK="${CHANNEL_MASK_PATH:-/models/glm51-fp8-channel-mask-s256.safetensors}"
PROFILE_CONC="${PROFILE_CONC:-32}"          # a failing AC-11 row (fails TPS + TTFT)
PROFILE_NUM_STEPS="${PROFILE_NUM_STEPS:-40}"
OUT="${OUT:-development/loop8/runs/20260608_ac4/profile_ds_c${PROFILE_CONC}}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SGLANG_TORCH_PROFILER_DIR="$(pwd)/${OUT}/trace"
mkdir -p "$SGLANG_TORCH_PROFILER_DIR"

pkill -f "sglang.launch_server" 2>/dev/null || true; pkill -f "sglang::scheduler" 2>/dev/null || true
sleep 15; rm -f /dev/shm/psm_* /dev/shm/sem.mp-* 2>/dev/null || true

echo ">>> boot DS (mask=$MASK, conc=$PROFILE_CONC, trace=$SGLANG_TORCH_PROFILER_DIR)"
MODEL_PATH="$GLM" CHANNEL_MASK_PATH="$MASK" MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.7}" TOP_K=2048 \
  DISABLE_CUSTOM_ALL_REDUCE=1 RANDOM_SEED=20260607 SGLANG_TORCH_PROFILER_DIR="$SGLANG_TORCH_PROFILER_DIR" \
  bash development/serve_double_sparsity.sh > "$OUT/serve.log" 2>&1 &
ready=0
for i in $(seq 1 150); do
  if curl -sf http://127.0.0.1:30000/health >/dev/null 2>&1; then ready=1; break; fi
  sleep 10
done
if [[ "$ready" != "1" ]]; then echo "!! DS boot FAILED"; tail -40 "$OUT/serve.log"; exit 1; fi

echo ">>> profiled bench burst (conc=$PROFILE_CONC, ${PROFILE_NUM_STEPS} steps)"
# Fail closed: a nonzero bench must abort, not fall through to summary generation.
if ! python3 -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port 30000 \
  --dataset-name generated-shared-prefix --gsp-system-prompt-len 2253 --gsp-question-len 1843 \
  --gsp-output-len 512 --gsp-prompts-per-group "$PROFILE_CONC" \
  --num-prompts "$((PROFILE_CONC * 2))" --max-concurrency "$PROFILE_CONC" \
  --profile --profile-num-steps "$PROFILE_NUM_STEPS" \
  > "$OUT/bench.log" 2>&1; then
  echo "!! profiled bench FAILED"; tail -40 "$OUT/bench.log"
  pkill -f "sglang.launch_server" 2>/dev/null || true; pkill -f "sglang::scheduler" 2>/dev/null || true
  exit 1
fi
pkill -f "sglang.launch_server" 2>/dev/null || true; pkill -f "sglang::scheduler" 2>/dev/null || true; sleep 15

echo ">>> summarize trace -> $OUT/profile_summary.txt"
python3 - "$SGLANG_TORCH_PROFILER_DIR" "$OUT/profile_summary.txt" <<'PY'
import sys, os, gzip, json, collections
trace_dir, out = sys.argv[1], sys.argv[2]
# The profiler writes into a timestamped subdir as <id>-TP-N.trace.json.gz — walk recursively.
files = []
for root, _, fs in os.walk(trace_dir):
    for fn in fs:
        if ".trace.json" in fn or fn.endswith(".json") or fn.endswith(".json.gz"):
            files.append(os.path.join(root, fn))
if not files:
    open(out, "w").write(f"NO TRACE FILES under {trace_dir}\n")
    print(f"FAIL: no trace files under {trace_dir}", file=sys.stderr); sys.exit(2)
f = sorted(files)[0]  # TP-0, representative rank
op = gzip.open(f, "rt") if f.endswith(".gz") else open(f)
data = json.load(op)
evs = data.get("traceEvents", data) if isinstance(data, dict) else data
kern = collections.defaultdict(lambda: [0.0, 0])
for e in evs:
    if not isinstance(e, dict): continue
    if e.get("cat") in ("kernel", "Kernel", "gpu_memcpy", "gpu_memset") and "dur" in e:
        kern[e.get("name", "?")][0] += float(e["dur"]); kern[e.get("name", "?")][1] += 1
if not kern:
    open(out, "w").write(f"NO GPU-KERNEL ROWS in {os.path.basename(f)} ({len(evs)} events)\n")
    print(f"FAIL: trace {f} has zero GPU-kernel events — unusable profile", file=sys.stderr); sys.exit(3)
rows = sorted(kern.items(), key=lambda kv: -kv[1][0])[:20]
tot = sum(v[0] for v in kern.values()) or 1.0
with open(out, "w") as fh:
    fh.write(f"trace: {os.path.basename(f)}  ({len(evs)} events, total GPU-kernel us={tot:.0f})\n\n")
    fh.write(f"{'tot_us':>14} {'%':>6} {'calls':>8}  kernel\n")
    for name, (d, n) in rows:
        fh.write(f"{d:14.0f} {100*d/tot:6.1f} {n:8d}  {name[:90]}\n")
print(f"wrote {out} from {os.path.basename(f)} ({len(kern)} distinct GPU kernels)")
PY
echo ">>> DONE"; head -30 "$OUT/profile_summary.txt"
