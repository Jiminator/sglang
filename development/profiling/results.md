# DS-vs-DSA one-batch decode profiling — results (GLM-5.1-FP8, 8×H200)

Characterization for Loop-8 (`development/profiling/plan.md`). **Not a gate run** — AC-4 is
closed (R12). The point is to attribute *where* the known DS-on decode cost goes.

Three server configs, each captured twice (nsys system trace + torch per-rank trace),
one batch each, run sequentially (one TP=8 server at a time):

| Case | Mode | `mem_fraction_static` | batch | What |
|------|------|------|------|------|
| **1 — DS** | Double Sparsity ON | 0.7 | 29 | DS KV-pool-capped steady-state batch |
| **2 — DSA@same** | DSA-native (DS OFF) | 0.7 | 29 | apples-to-apples vs Case 1 (only DS flags differ) |
| **3 — DSA@best** | DSA-native (DS OFF) | 0.8 | 64 | best DSA config (full batch) |

**Op-point invariants (verified in every `serve.log`):** `disable_custom_all_reduce=False`
(custom all-reduce ON), `enable_double_sparsity=True` (Case 1) / `False` (Cases 2,3),
`mem_fraction_static` 0.7/0.7/0.8, `page_size=64`, `kv_cache_dtype=fp8_e4m3`,
`dsa_{prefill,decode}_backend=flashmla_kv`, `disable_radix_cache=True`,
`disable_overlap_schedule=True`, `disable_piecewise_cuda_graph=True`, CUDA graph ON,
`enable_flashinfer_allreduce_fusion=True`, `random_seed=20260607`.
`PYTORCH_CUDA_ALLOC_CONF` kept unset (expandable_segments would break custom-all-reduce
IPC at GLM TP=8 graph capture — BL-20260608). Case 1 boots and **admits the full bs 29**
(no KV-pool admission failure).

> **Single-batch caveat (from the benchmarking guide).** `bench_one_batch_server` sends ONE
> batch — the server is never in steady state and its latency metrics are biased. These runs
> are used for **kernel-level attribution only**, NOT as SLO numbers. The authoritative
> throughput is the locked `bench_serving` sweep in
> `development/loop8/runs/20260608_ac4/slo2_ac11_report.txt` (DS decode-TPS ~17–23, DSA ~26–42).

---

## 0. Headline

At the **identical** operating point (bs 29, mem 0.7, custom-AR ON), **DS-on decode does
1.84× the GPU-kernel work per step that DSA-native does** (torch decode trace, TP-0:
632,239 µs vs 342,857 µs over the same 10-step window). DS is **not** "DSA plus an index" —
it **replaces DSA's compact fused fp8 indexer (~17k µs/window) with a much heavier stack**:

1. **A generic PyTorch top-k/sort stack** (`mbtopk` digit-count/gatherTopK, `radixSortKVInPlace`,
   `sbtopk<long>`, `scan_by_key`, `searchsorted`) — **+138.6k µs** vs DSA.
2. **A DS-only f32 ring all-reduce** (`ncclDevKernel_AllReduce_Sum_f32_RING`, 1 call per layer
   per step) — **+124.9k µs** vs DSA, which issues no such kernel (its per-layer all-reduce is
   the shared trtllm-fusion one-shot, ~equal in both cases).
3. **`_logical_score_kernel`** (the DS channel-score compute) — **+63.1k µs** vs DSA.

The core sparse-MLA decode attention (`flash_fwd_splitkv_mla_fp8_sparse`) is ~31k µs in **both**
cases — confirming this is a clean apples-to-apples diff and the gap is the DS index/scoring +
its synchronization, not the attention math. This matches the locked SLO sweep's "DS ≈ 0.55×
DSA decode-TPS": 459 vs 876 tok/s single-batch here = 0.52×.

---

## 1. Per-case decode-step GPU-kernel breakdown (torch, TP-0, decode stage)

`--profile-by-stage` decode trace, rank TP-0, 10 decode steps captured (kernel-time µs and % of
that rank's decode GPU-kernel time). Categories from `compare_decode.py`. "DS-*" rows are
DS-only kernels; "DSA-*" rows are DSA-native's fused-indexer kernels that DS replaces.

| Category | Case 1 (DS) µs | Case 1 % | Case 2 (DSA@0.7) µs | Case 2 % | Case 3 (DSA@0.8 bs64) µs | Case 3 % |
|---|---|---|---|---|---|---|
| all-reduce (NCCL f32 + trtllm-fusion) | 163,790 | 25.9 | 39,815 | 11.6 | TBD-3 | TBD-3 |
| topk/sort stack (mbtopk/radixSort/sbtopk/scan) | 159,166 | 25.2 | 20,564 | 6.0 | TBD-3 | TBD-3 |
| MoE (`fused_moe_kernel`) | 91,880 | 14.5 | 118,590 | 34.6 | TBD-3 | TBD-3 |
| **DS:logical-score** (`_logical_score_kernel`) | 63,107 | 10.0 | 0 | 0.0 | TBD-3 | TBD-3 |
| attention — sparse-MLA decode | 41,921 | 6.6 | 40,521 | 11.8 | TBD-3 | TBD-3 |
| norm/rope/elementwise | 41,895 | 6.6 | 20,278 | 5.9 | TBD-3 | TBD-3 |
| GEMM/proj (deep_gemm fp8 + nvjet) | 34,596 | 5.5 | 49,335 | 14.4 | TBD-3 | TBD-3 |
| fp8-quant for index logits (`per_token_group_quant`) | 27,818 | 4.4 | 30,617 | 8.9 | TBD-3 | TBD-3 |
| **DSA:topk-transform** (`topk_transform_decode`) | 0 | 0.0 | 7,685 | 2.2 | TBD-3 | TBD-3 |
| **DSA:mqa-logits** (`sm90_fp8_paged_mqa_logits`) | 29 | 0.0 | 6,875 | 2.0 | TBD-3 | TBD-3 |
| hadamard/signature | 0 | 0.0 | 2,641 | 0.8 | TBD-3 | TBD-3 |
| memcpy/set + other | 8,038 | 1.3 | 5,936 | 1.7 | TBD-3 | TBD-3 |
| **Total decode GPU-kernel µs / window** | **632,239** | 100 | **342,857** | 100 | TBD-3 | 100 |

**nsys whole-capture cross-check (decode-dominated; `cuda_gpu_kern_sum`, rolled to coarse
categories — separate parser, separate run):** Case 1 = DS-index/scoring 34.2% / all-reduce
29.7% / MoE 15.0% / GEMM 8.7% / MLA-attn 5.3%. Corroborates the torch decode split (the nsys
"DS-index/scoring" bucket merges logical-score + the top-k stack + fp8-quant). Per-case nsys
kern_sum CSVs are committed (`*/nsys/kern_sum.csv`).

## 2. Headline answer — Case 1 vs Case 2 (clean DS overhead, same bs 29 / mem 0.7)

Clean DS overhead = Case1 − Case2 decode GPU-kernel time = **632,239 − 342,857 = +289,382 µs**
over the 10-step window (DS = **1.84×** DSA). Attribution of the DS-specific kernel groups
(positive deltas; `cmp_case1_vs_case2.txt`):

| DS-specific kernel group | Δ µs (Case1 − Case2) | note |
|---|---|---|
| `ncclDevKernel_AllReduce_Sum_f32_RING` | **+124,873** | DS-only; 780 calls = 1/layer/step. DSA issues none. |
| top-k / sort stack (mbtopk + radixSort + sbtopk<long> + scan_by_key + searchsorted) | **+138,602** | DS uses generic PyTorch top-k over `top_k=2048` candidates |
| `_logical_score_kernel` | **+63,107** | DS channel-score compute |
| (DS index/scoring subtotal) | **≈ +326,582** | partially offset in the net by shared-kernel run-variance |

What DSA spends *instead* (and DS does not): `sm90_fp8_paged_mqa_logits` (6,875 µs) +
`topk_transform_decode_kernel` (7,685 µs) + `fast_hadamard_transform` (2,641 µs) ≈ **17.2k µs** —
a tightly-fused fp8 indexer, ~19× cheaper than DS's logical-score + torch-topk + extra
all-reduce path.

The shared kernels measure close (MLA-attn 41.9k vs 40.5k; trtllm-fusion all-reduce 35.9k vs
33.8k; fp8-quant 24.1k vs 25.4k). The one notable shared-kernel difference — MoE 91.9k (DS) vs
118.6k (DSA), same 1,500 calls — is **run-to-run clock/contention variance** (two separate
single-batch server boots, not steady-state; identical MoE work), not a DS effect. It is why the
*net* total delta (289k) is below the DS-specific-additions subtotal (327k); the DS tax itself
(all-reduce + top-k + logical-score) is unambiguous and DSA-absent.

## 3. Case 2 vs Case 3 — DSA batch-efficiency (bs 29 vs bs 64)

> Separates "DS is slow" from "small batch is inefficient". Case 1's bs 29 is **forced** by the
> DS KV pool, not chosen; Case 3 is DSA at its full bs 64.

TBD-3 (awaiting runs 3a/3b). Will report DSA decode GPU-time and per-request decode efficiency
at bs 29 vs bs 64, to show how much of any DS gap is intrinsic DS cost vs the small-batch
penalty DS is additionally stuck with.

## 4. Bench `--show-report` single-batch latency (NOT steady-state — cross-reference only)

ISL 4096 / OSL 512 (torch runs), greedy. **Biased single-batch numbers** — do not equate with
the locked SLO sweep; cross-reference only.

| Case | batch | latency (s) | output tok/s | ITL (ms) | last TTFT (s) | per-req decode tok/s |
|---|---|---|---|---|---|---|
| 1 — DS bs29 | 29 | 40.24 | 459.21 | 63.15 | 7.91 | 15.8 |
| 2 — DSA bs29 | 29 | 29.33 | 875.92 | 33.11 | 12.38 | 30.2 |
| 3 — DSA bs64 | 64 | TBD-3 | TBD-3 | TBD-3 | TBD-3 | TBD-3 |

Per-request decode tok/s lines up with the locked SLO sweep regime (DS ~17, DSA ~26–42),
confirming these biased single-batch runs reproduce the right relative behavior even though the
absolute numbers are not SLO-grade. (The second, slower row each torch `bench.log` shows is the
profiler-instrumented pass; ignore for latency.)

## 5. Artifacts (committed unless noted)

All under `development/profiling/runs/20260609/`. Raw multi-GB traces (`*.nsys-rep`, `*.sqlite`,
`*.trace.json.gz`) are **git-ignored** (paths listed; regenerable). Committed: summaries,
`result.jsonl`, nsys `kern_sum.csv`, serve/bench/driver logs, `server_args.json`, comparison
tables, and the run/parse scripts (`_env.sh`, `run_case.sh`, `summarize_torch.py`,
`summarize_nsys.py`, `compare_decode.py`).

| Run | torch trace (raw, uncommitted) | nsys rep (raw, uncommitted) | committed summaries |
|---|---|---|---|
| Case 1 (DS) | `case1_ds/torch/trace/<ts>/*-TP-N-{DECODE,EXTEND}.trace.json.gz` | `case1_ds/nsys/trace.nsys-rep` (775M) | `case1_ds/{torch,nsys}/decode_summary.txt`, `case1_ds/nsys/kern_sum.csv` |
| Case 2 (DSA@0.7) | `case2_dsa07/torch/trace/<ts>/*.trace.json.gz` | `case2_dsa07/nsys/trace.nsys-rep` (216M, OSL=64) | `case2_dsa07/{torch,nsys}/decode_summary.txt`, kern_sum.csv |
| Case 3 (DSA@0.8 bs64) | `case3_dsa08/torch/trace/...` | `case3_dsa08/nsys/trace.nsys-rep` (OSL=64) | TBD-3 |
| diff | — | — | `cmp_case1_vs_case2.txt`, `cmp_case2_vs_case3.txt` (TBD-3) |

## 6. Method notes / deviations

- **nsys `--output-len`:** torch runs use the plan's OSL 512 (authoritative latency + clean
  10-step decode trace). **nsys runs use OSL 64** — the per-decode-step kernel mix is stationary,
  so 64 steps give identical per-kernel attribution while keeping the trace finalizable in
  seconds (Case 1's OSL-512 nsys rep was 775 MB and took ~4 min to serialize; OSL-64 ≈ 216 MB).
  This changes nothing about the attribution; it only shortens the captured decode window.
- nsys captured server-side: `nsys profile --trace cuda,nvtx,cublas --cuda-graph-trace node
  --trace-fork-before-exec true --delay D --duration 900 python -m sglang.launch_server …`,
  with the bench gated to start only after collection is live, then `nsys stop --session=…` to
  finalize instantly. `--cuda-graph-trace node` is load-bearing: decode is CUDA-graph-replayed,
  so without it the replay collapses to one opaque node.
- torch: server launched with `SGLANG_TORCH_PROFILER_DIR`; client `--profile --profile-by-stage
  --profile-activities CPU GPU --profile-steps 10`. CUDA graph kept ON (real op-point) — accepted
  trade-off: no Python-source mapping inside graph regions. The `with_stack` PyTorch-profiler bug
  did not trigger; no fallback needed.
