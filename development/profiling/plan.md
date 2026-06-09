# Loop 8 — DS-vs-DSA one-batch profiling plan (GLM-5.1-FP8, 8×H200)

## Read this first (context for a fresh session)
You are running on a single node with **8 local NVIDIA H200 GPUs**. The repo is SGLang. This task is a
**GPU profiling job**, self-contained in this file — but read the background below so the numbers make sense.

**Glossary**
- **GLM-5.1-FP8** — the model under test (`model_type=glm_moe_dsa`, 78 layers, TP=8). Weights at the `GLM`
  path in *Fixed parameters*.
- **DSA (native sparse attention)** — the model's *built-in* trained sparse-attention indexer. This is the
  **served default**.
- **DS (Double Sparsity)** — an **opt-in, default-OFF** alternative sparse-attention path enabled with
  `--enable-double-sparsity` + a calibrated channel-mask file. It runs the *same* decode path as DSA **plus**
  an extra index/scoring stack, so it is strictly more work per token.
- We are profiling **why DS-on decode is slower than DSA-native**, and why DS can only sustain a small decode
  batch.

**Background files to read before running (repo-relative paths):**
1. `development/loop8/task9_gate_results.md` — the gate record. Has the SLO numbers, the "DS-on ≤ DSA /
   conc-64 ceiling" finding, and the op-point. **This is the source of the "Observed facts" below.**
2. `development/loop8/runs/20260608_ac4/slo2_ac11_report.txt` — the locked SLO sweep (DSA vs DS decode-TPS +
   P99 TTFT at conc 16/32/64) the profiling explains.
3. `development/loop8/runs/20260608_ac4/profile_ds_c32/profile_summary.txt` — an earlier DS conc-32 GPU-kernel
   summary (top kernels: NCCL all-reduce, MoE, and the DS-specific stack). This profiling job produces the
   apples-to-apples version of that.
4. `development/serve_double_sparsity.sh` and `development/serve_native_nsa.sh` — the production DS / DSA-native
   launchers. The exact server flags + the `DS_CONFIG` JSON below are copied from these; cross-check if unsure.
5. `docs_new/docs/developer_guide/benchmark_and_profiling.mdx` — the official profiling guide this plan follows.
6. `CLAUDE.md` — project engineering doctrine (surgical changes, prove with numbers, no drive-by edits).

**Hard environment rules (a wrong setting silently wastes a whole run):**
- **Never export `PYTORCH_CUDA_ALLOC_CONF=expandable_segments...` for serving.** That allocator uses CUDA VMM,
  which breaks the custom-all-reduce IPC handles during CUDA-graph capture → GLM TP=8 fails to boot
  (`custom_all_reduce.cuh: CUDA error: invalid argument`). Keep custom all-reduce ON (do NOT pass
  `--disable-custom-all-reduce`). The commands below `unset` it; don't re-add it.
- Only **one** TP=8 server fits on the 8 GPUs at a time — run the cases **sequentially**, `teardown` between.
- These are characterization runs; they do **not** gate anything (the acceptance work is already closed).

## Goal
Profile a **single decode batch** with `sglang.bench_one_batch_server` to explain the DS-vs-DSA decode-TPS
gap and the conc-64 batch ceiling we observed in the SLO sweeps. Three server configurations, each captured
**twice** (once with an **nsys** system trace, once with the **torch profiler**), **one trial per run** → 6
profiling runs total.

Observed facts this plan investigates:
- DS-on steady-state decode batch is KV-pool-capped at **~29** at `mem_fraction_static=0.7`; DSA-native sustains the
  full **64** at `mem_fraction_static=0.8`.
- DS-on decode-TPS ≈ 0.55× DSA at every concurrency; the proper-op-point profile attributed the gap to a
  ~14% DS-specific index/scoring kernel stack on top of the shared all-reduce + MoE base.

## Cases
| Case | Mode | `mem_fraction_static` | batch size | Purpose |
|------|------|------------------------|-----------|---------|
| **1 — DS** | Double Sparsity ON | 0.7 | **29** | Best DS config we've seen (the DS KV-pool-capped steady-state batch). |
| **2 — DSA@same** | DSA-native (DS OFF) | 0.7 | **29** | Apples-to-apples: identical server args to Case 1 **except** DS enablement; isolates the DS index/scoring cost at the same batch + mem fraction. |
| **3 — DSA@best** | DSA-native (DS OFF) | 0.8 | **64** | Best DSA config we've seen (full batch at its serving mem-fraction). |

**Invariant:** Case 1 and Case 2 server launch commands are **byte-identical except** Case 1 adds
`--enable-double-sparsity --double-sparsity-config '<DS_CONFIG>'`. Same model, TP, page size, KV dtype, DSA
backends, graph flags, radix policy, seed, **mem-fraction (0.7)**, custom-all-reduce ON. Case 3 differs only
in `--mem-fraction-static 0.8` (and we drive it at batch size 64).

## Fixed parameters (all cases)
- Model: `GLM=/cluster-storage/models/models--zai-org--GLM-5.1-FP8/snapshots/f396cf805182f4ca10fa675e1a99815b3ca384db`
- Mask (Case 1 only): `/models/glm51-fp8-channel-mask-s256.safetensors`
- TP=8, page 64, `--kv-cache-dtype fp8_e4m3`, DSA backends `flashmla_kv`, `--disable-overlap-schedule`,
  `--disable-piecewise-cuda-graph`, `--disable-radix-cache`, `--random-seed 20260607`, `--trust-remote-code`.
- **Custom all-reduce ON** (do NOT pass `--disable-custom-all-reduce`) and **NEVER set
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments`** for serving — per BL-20260608, expandable_segments breaks
  custom-all-reduce-v2 IPC handles at GLM TP=8 graph capture. (Both paired launchers strip it; here we run
  `launch_server` directly, so just keep it unset.)
- Workload shape: `--input-len 4096 --output-len 512` (the SLO ISL/OSL), `--dataset-name random` (fresh tokens,
  no prefix reuse → clean decode-cost profile), greedy (`--temperature 0`).
- Port 30000; one server up at a time (GLM cannot co-host two TP=8 servers).
- **Keep CUDA graph ON** (do NOT pass `--disable-cuda-graph`): we profile the *real* decode op-point (graph
  replay). Per the profiling guide, CUDA graph ON means the torch trace can't map graph-replayed kernels back
  to Python source, and NVTX markers inside graphs aren't emitted — that's an accepted trade-off here (we want
  the true op-point, and nsys `--cuda-graph-trace=node` still resolves per-kernel detail inside the graph).

> **Caveat (from the benchmarking guide):** `bench_one_batch_server` sends a *single* batch, so the server is
> **never in steady state and its latency metrics are biased**. We use these runs only for **kernel-level
> attribution** (where decode time goes, DS-on vs DS-off), NOT as SLO numbers — the authoritative throughput
> numbers are the locked `bench_serving` sweep in `development/loop8/runs/20260608_ac4/slo2_ac11_report.txt`.

```bash
GLM=/cluster-storage/models/models--zai-org--GLM-5.1-FP8/snapshots/f396cf805182f4ca10fa675e1a99815b3ca384db
MASK=/models/glm51-fp8-channel-mask-s256.safetensors
OUT=development/profiling/runs/20260609
mkdir -p "$OUT"
unset PYTORCH_CUDA_ALLOC_CONF    # critical: keep custom all-reduce working
# DS config (Case 1) — matches serve_double_sparsity.sh defaults:
DS_CONFIG='{"top_k": 2048, "page_size": 64, "channel_mask_path": "/models/glm51-fp8-channel-mask-s256.safetensors", "device_buffer_size": 4096, "signature_dtype": "fp16", "scorer_norm": "off", "scorer_norm_hybrid_threshold": 8192, "head_agg": "max", "anchor_mode": "off", "anchor_budget": 0, "recall_oracle": false, "enable_lifted_budget_decode": false, "lifted_budget_top_k": 0}'
```

### Common server-arg blocks
```bash
# COMMON: identical for all three cases
COMMON_ARGS="--model-path $GLM --tp-size 8 --kv-cache-dtype fp8_e4m3 --page-size 64 \
  --dsa-prefill-backend flashmla_kv --dsa-decode-backend flashmla_kv \
  --disable-overlap-schedule --disable-piecewise-cuda-graph --disable-radix-cache \
  --random-seed 20260607 --trust-remote-code --host 127.0.0.1 --port 30000"
```
- Case 1 server args: `$COMMON_ARGS --mem-fraction-static 0.7 --enable-double-sparsity --double-sparsity-config "$DS_CONFIG"`
- Case 2 server args: `$COMMON_ARGS --mem-fraction-static 0.7`           # == Case 1 minus the two DS flags
- Case 3 server args: `$COMMON_ARGS --mem-fraction-static 0.8`

### Helpers
```bash
wait_ready() { for i in $(seq 1 180); do curl -sf http://127.0.0.1:30000/health >/dev/null 2>&1 && { echo "ready ${i}0s"; return 0; }; sleep 10; done; return 1; }
teardown()  { pkill -f "sglang.launch_server" 2>/dev/null||true; pkill -f "sglang::scheduler" 2>/dev/null||true; sleep 20; rm -f /dev/shm/psm_* /dev/shm/sem.mp-* 2>/dev/null||true; }
```

---

## Profiler mode A — Torch profiler (per-rank kernel trace, prefill/decode split)
Per the guide, the **server must be launched with `SGLANG_TORCH_PROFILER_DIR`** to enable the profiler; the
client then selects the output dir (`--profile-output-dir`) and the guide recommends **setting the env on both
server and client** to avoid confusion about where traces land. The bench drives one batch and triggers the
server-side torch profiler via `--profile`; `--profile-by-stage` separates prefill from decode; `--profile-steps`
sets how many decode steps to capture (keep modest so the per-rank trace stays browser-openable — the guide
notes large traces won't load in perfetto/chrome).

Template (run after `wait_ready`):
```bash
TR="$(pwd)/$OUT/<CASE>_torch/trace"; mkdir -p "$TR"
SGLANG_TORCH_PROFILER_DIR="$TR" \
  python -m sglang.launch_server <CASE server args> > "$OUT/<CASE>_torch/serve.log" 2>&1 &
wait_ready || { echo FAIL; tail -40 "$OUT/<CASE>_torch/serve.log"; teardown; }
SGLANG_TORCH_PROFILER_DIR="$TR" \
  python -m sglang.bench_one_batch_server --base-url http://127.0.0.1:30000 --model-path "$GLM" --trust-remote-code \
  --batch-size <BS> --input-len 4096 --output-len 512 --temperature 0 --show-report \
  --profile --profile-by-stage --profile-activities CPU GPU --profile-steps 10 \
  --profile-output-dir "$TR" \
  --result-filename "$OUT/<CASE>_torch/result.jsonl" > "$OUT/<CASE>_torch/bench.log" 2>&1
teardown
```
- Output: `$TR/<ts>/{profile_id}-TP-N.trace.json.gz` per rank (+ `server_args.json`). View at https://ui.perfetto.dev.
- If the run hits `RuntimeError: ... Python replay stack is empty` (a known PyTorch profiler bug), re-run with
  `SGLANG_PROFILE_WITH_STACK=False` exported on the **server** launch (per the guide's workaround).
- Raise `--profile-steps` for a longer decode window if traces stay small enough to open.

## Profiler mode B — nsys system trace (kernels inside CUDA graphs, all 8 ranks)
The GPU work lives in the server process tree, so — exactly as the guide's "Profile a server" recipe —
**nsys wraps the `launch_server` command** with `--trace-fork-before-exec=true --cuda-graph-trace=node`.
`--cuda-graph-trace=node` is REQUIRED: decode is CUDA-graph-replayed, so without it the whole replay collapses
to one opaque node. Use `--delay` to skip the long boot/graph-capture, set `--duration` to a **large** value,
and **stop the capture manually** with `nsys stop` right after the single bench batch finishes (the guide's
recommended pattern — more robust than guessing a duration).

Template:
```bash
mkdir -p "$OUT/<CASE>_nsys"
nsys profile --output "$OUT/<CASE>_nsys/trace" --force-overwrite true \
  --trace cuda,nvtx,cublas --cuda-graph-trace node --trace-fork-before-exec true \
  --delay 210 --duration 900 \
  python -m sglang.launch_server <CASE server args> > "$OUT/<CASE>_nsys/serve.log" 2>&1 &
wait_ready || { echo FAIL; tail -40 "$OUT/<CASE>_nsys/serve.log"; teardown; }
# drive ONE batch (no --profile here; nsys is capturing server-side):
python -m sglang.bench_one_batch_server --base-url http://127.0.0.1:30000 --model-path "$GLM" --trust-remote-code \
  --batch-size <BS> --input-len 4096 --output-len 512 --temperature 0 --show-report \
  --result-filename "$OUT/<CASE>_nsys/result.jsonl" > "$OUT/<CASE>_nsys/bench.log" 2>&1
# immediately finalize the trace (per the guide): get the session id, then stop it
SID=$(nsys sessions list 2>/dev/null | grep -oE "profile-[0-9A-Za-z]+" | head -1)
nsys stop --session="$SID"            # writes trace.nsys-rep instantly
teardown
```
Notes:
- `--delay 210` ≈ GLM-5.1 TP=8 boot/graph-capture time; tune to the server's observed "fired up and ready"
  log so the capture starts right as the bench runs. `--duration 900` is just a safety cap — the `nsys stop`
  ends it as soon as the batch is done.
- Open `trace.nsys-rep` in `nsys-ui`. (Optional, if per-layer detail is wanted: the guide's layerwise path uses
  `--enable-layerwise-nvtx-marker` **with `--disable-cuda-graph`** + `/start_profile` `CUDA_PROFILER` — but that
  changes the op-point, so it is NOT used here.)

---

## The 6 runs (substitute `<CASE>`, `<BS>`, and the case server args)
| Run | CASE dir | Mode | Server args | BS |
|-----|----------|------|-------------|----|
| 1a | `case1_ds`   | nsys  | Case 1 (mem 0.7 + DS) | 29 |
| 1b | `case1_ds`   | torch | Case 1 (mem 0.7 + DS) | 29 |
| 2a | `case2_dsa07`| nsys  | Case 2 (mem 0.7, DS off) | 29 |
| 2b | `case2_dsa07`| torch | Case 2 (mem 0.7, DS off) | 29 |
| 3a | `case3_dsa08`| nsys  | Case 3 (mem 0.8, DS off) | 64 |
| 3b | `case3_dsa08`| torch | Case 3 (mem 0.8, DS off) | 64 |

Run them sequentially (one server at a time). Always `teardown` between runs.

## Execution mechanics (important for a fresh session)
- **A GLM TP=8 boot takes ~3–4 min** (weight load + 71 s CUDA-graph capture across 8 ranks). This exceeds a
  single foreground command's timeout, so **launch each server in the background and poll `/health`** (the
  `wait_ready` helper) before sending the bench — do not block on the boot in one shell call. A robust pattern:
  write the full per-run sequence (boot → wait_ready → bench → finalize → teardown) into a small script and run
  it as a background process, then poll its log / wait for it to exit.
- For the **nsys** runs, the server is launched *under* `nsys` (it owns the process group); `wait_ready` still
  polls `/health` normally. Finalize with `nsys stop` (see Mode B), then `teardown`.
- Each full run (boot + one batch + finalize) is roughly **5–8 min**; all 6 ≈ **45–60 min** wall-clock.
- After each run, confirm the expected artifact exists (`*.nsys-rep` for nsys, `trace/<ts>/*-TP-*.trace.json.gz`
  for torch) and the bench log shows a completed batch (non-zero decode tokens) before moving on.
- If you only have limited time, the **most informative single comparison is Case 1 vs Case 2** (the clean
  DS-on-vs-DS-off overhead at the same batch + mem fraction) — run runs 1a/1b/2a/2b first; Case 3 is secondary.

## Output layout
```
development/profiling/runs/20260609/
  case1_ds/{nsys/{trace.nsys-rep,serve.log,bench.log,result.jsonl}, torch/{trace/<ts>/*-TP-N.trace.json.gz,serve.log,bench.log,result.jsonl}}
  case2_dsa07/{nsys/...,torch/...}
  case3_dsa08/{nsys/...,torch/...}
```

## Pre-flight checks
1. `nvidia-smi` → 8 GPUs idle; `pgrep -f sglang.launch_server` empty (else `teardown`).
2. `PYTORCH_CUDA_ALLOC_CONF` unset (custom all-reduce must stay ON).
3. Server boot log shows `disable_custom_all_reduce=False`, `enable_double_sparsity=True` (Case 1) / `False`
   (Cases 2,3), and the expected `mem_fraction_static`.
4. Sanity: Case 1 must boot at bs 29 (DS KV pool ≈ 124–142k tokens fits 29×~4.6k). If `bench_one_batch_server`
   reports the server couldn't admit the full batch, the KV pool is the limiter — record it, don't force.

## Analysis (what to extract)
- **Case 1 vs Case 2 (same bs 29, same mem 0.7, DS on vs off):** the *clean* DS overhead. Torch: per-stage
  (prefill vs decode) kernel-time breakdown; the DS-specific kernels (top-k selection, fp8 MQA logits,
  hadamard signature, logical-score, sparse-MLA decode) appear in Case 1 and not Case 2. nsys: where those
  kernels sit on the decode-step timeline and whether they serialize with or overlap the all-reduce/MoE.
- **Case 2 vs Case 3 (DSA at bs 29 vs bs 64):** how DSA decode scales with batch (the dense baseline's
  batch-efficiency), to separate "DS is slow" from "small batch is inefficient".
- **Per decode step:** total GPU time, all-reduce share, MoE share, DS-specific share. Confirms whether the DS
  gap is the index/scoring stack (compute) vs. the KV-pool-driven small-batch inefficiency (Case 1's bs 29 is
  forced by its KV pool, not chosen).
- Cross-check the bench's reported decode throughput against the SLO-sweep numbers (DS ~17–23, DSA ~26–42).

## Deliverable — what to produce
Write findings to **`development/profiling/results.md`** (create it). It must contain:
1. A table of **per-case decode-step GPU-kernel breakdown** (% of GPU-kernel time): all-reduce (NCCL),
   MoE, attention, and — for Case 1 — the **DS-specific kernels** (top-k / gatherTopK, top-k transform,
   fp8 MQA logits, hadamard/signature, logical-score, sparse-MLA decode). One column per case.
2. The headline answer: **Case 1 vs Case 2** (same bs 29, same mem 0.7) → the *clean* DS overhead =
   (Case 1 decode-step time − Case 2 decode-step time), and which DS kernels account for it.
3. **Case 2 vs Case 3** (DSA bs 29 vs 64) → DSA's batch-efficiency, to separate "DS is slow" from "small
   batch is inefficient" (Case 1's bs 29 is *forced* by the DS KV pool, not chosen).
4. The bench `--show-report` latency numbers per case, with the explicit reminder they are single-batch /
   not-steady-state (cross-reference, don't equate, the locked SLO sweep).
5. Paths to every committed artifact (`.nsys-rep`, torch `trace/`, `result.jsonl`, serve/bench logs).

**How to parse the traces:**
- **Torch** (`*-TP-N.trace.json.gz`, one per rank): aggregate chrome-trace events with `cat in {"kernel",
  "gpu_memcpy"}` by `name`, summing `dur`, top-N by total — pick rank TP-0 as representative. The summarizer
  in `development/profile_ds.sh` (its inline Python `os.walk` + kernel-aggregation block) does exactly this;
  reuse/adapt it. `--profile-by-stage` writes prefill and decode traces separately — report the **decode** one.
- **nsys** (`trace.nsys-rep`): `nsys stats --report cuda_gpu_kern_sum --format table trace.nsys-rep` gives the
  per-kernel GPU-time summary; `--report cuda_gpu_trace` gives the timeline. Use the decode region (after
  prefill) for the steady per-step breakdown.
- Commit `results.md` + the small artifacts (summaries, `result.jsonl`); the raw multi-GB nsys/torch traces
  can stay uncommitted (note their paths) unless asked otherwise.

## Notes / decisions
- These are **profiling/characterization** runs, not gate runs — they do not feed AC-4 (closed R12). No
  comparator/op-point parity contract is enforced here; the point is to attribute the known DS decode cost.
- One trial per run (per request). Re-run a case if the bench log shows admission failure or a degenerate
  (empty-stream) batch.
- nsys `--cuda-graph-trace=node` is the load-bearing flag for decode (graph replay); without it the trace is
  useless for per-kernel decode attribution.
- Torch `--profile-by-stage` is the load-bearing flag for separating the prefill cost from the decode cost.

## Alignment with the benchmarking & profiling guide
This plan follows `docs_new/docs/developer_guide/benchmark_and_profiling.mdx`:
- Tool choice: `bench_one_batch_server` is the guide's "single batch as one HTTP request to a running server"
  tool — it carries HTTP + scheduler overhead and is explicitly **not steady-state** (metrics biased), so we
  use it for kernel attribution, not for SLO numbers.
- Torch: server launched with `SGLANG_TORCH_PROFILER_DIR`; client `--profile` (+ env set on both sides per the
  guide); `SGLANG_PROFILE_WITH_STACK=False` is the documented fallback for the `with_stack` PyTorch profiler bug.
- nsys: the guide's "Profile a server" recipe — `nsys profile --trace-fork-before-exec=true
  --cuda-graph-trace=node ... python -m sglang.launch_server ...` with `--delay`, a large `--duration`, and
  manual `nsys stop --session=profile-XXXXX` (via `nsys sessions list`) to finalize the `.nsys-rep` instantly.
- CUDA graph kept ON for a realistic op-point (the guide notes the trade-offs: no Python-source mapping inside
  graph regions and no in-graph NVTX markers; both accepted here).
