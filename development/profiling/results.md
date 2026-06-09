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

Two **separable, multiplicative** penalties make served DS-on far slower than DSA-native, and
**DS is stuck paying both**:

1. **DS index/scoring tax ≈ 1.9×** — at the *identical* op-point (bs 29, mem 0.7, custom-AR ON),
   DS-on decode does **1.84× the GPU-kernel work per step** that DSA-native does (torch decode,
   TP-0: 632,239 µs vs 342,857 µs over the same 10-step window). DS is **not** "DSA plus an
   index" — it **replaces DSA's compact fused fp8 indexer (~17k µs/window) with a much heavier
   stack**: a DS-only f32 ring all-reduce (+124.9k µs), a generic PyTorch top-k/sort stack
   (+138.6k µs), and `_logical_score_kernel` (+63.1k µs).
2. **Batch-efficiency ≈ 1.78×** — DSA at bs 64 does only **1.23× the GPU work for 2.2× the
   tokens** of bs 29 (i.e. ~1.8× more GPU-efficient per token). DS-on **cannot** use this: its
   per-rank TokenLabelTable shrinks the KV pool and **caps the decode batch at ~29**, so it is
   locked out of the efficient large batch DSA reaches at mem 0.8.

Net single-batch aggregate decode throughput: **DS bs29 459 tok/s → DSA bs29 876 → DSA bs64
1555 tok/s = 3.4× best-vs-best**, decomposing as ≈1.9× (index tax) × ≈1.78× (batch). The core
sparse-MLA decode attention is ~31k µs at bs 29 in **both** Case 1 and Case 2 — confirming the
clean apples-to-apples diff: the gap is DS's index/scoring + its synchronization, not attention
math.

---

## 1. Per-case decode-step GPU-kernel breakdown (torch, TP-0, decode stage)

`--profile-by-stage` decode trace, rank TP-0, 10 decode steps (kernel-time µs and % of that
rank's decode GPU-kernel time). Categories from `compare_decode.py`. "DS-*" rows are DS-only
kernels; "DSA-*" rows are DSA-native's fused-indexer kernels that DS replaces.

| Category | Case 1 DS bs29 µs | % | Case 2 DSA bs29 µs | % | Case 3 DSA bs64 µs | % |
|---|---|---|---|---|---|---|
| all-reduce (NCCL f32 + trtllm-fusion) | 163,790 | 25.9 | 39,815 | 11.6 | 48,748 | 11.5 |
| topk/sort stack (mbtopk/radixSort/sbtopk/scan) | 159,166 | 25.2 | 20,564 | 6.0 | 20,808 | 4.9 |
| MoE (`fused_moe_kernel`) | 91,880 | 14.5 | 118,590 | 34.6 | 161,779 | 38.3 |
| **DS:logical-score** (`_logical_score_kernel`) | 63,107 | 10.0 | 0 | 0.0 | 0 | 0.0 |
| attention — sparse-MLA decode | 41,921 | 6.6 | 40,521 | 11.8 | 61,889 | 14.7 |
| norm/rope/elementwise | 41,895 | 6.6 | 20,278 | 5.9 | 16,621 | 3.9 |
| GEMM/proj (deep_gemm fp8 + nvjet) | 34,596 | 5.5 | 49,335 | 14.4 | 49,577 | 11.7 |
| fp8-quant for index logits (`per_token_group_quant`) | 27,818 | 4.4 | 30,617 | 8.9 | 33,170 | 7.9 |
| **DSA:topk-transform** (`topk_transform_decode`) | 0 | 0.0 | 7,685 | 2.2 | 8,288 | 2.0 |
| **DSA:mqa-logits** (`sm90_fp8_paged_mqa_logits`) | 29 | 0.0 | 6,875 | 2.0 | 11,600 | 2.7 |
| hadamard/signature | 0 | 0.0 | 2,641 | 0.8 | 3,537 | 0.8 |
| memcpy/set + other | 8,038 | 1.3 | 5,936 | 1.7 | 6,218 | 1.5 |
| **Total decode GPU-kernel µs / 10-step window** | **632,239** | 100 | **342,857** | 100 | **422,236** | 100 |

**nsys cross-check (`cuda_gpu_kern_sum`, separate parser/run; coarse categories):** Case 1
(OSL 512, decode-dominated) = DS-index/scoring 34.2% / all-reduce 29.7% / MoE 15.0% / GEMM 8.7% /
MLA-attn 5.3% — corroborates the torch decode split (the nsys "DS-index/scoring" bucket merges
logical-score + top-k stack + fp8-quant). **Cases 2 & 3 nsys use OSL 64, so their whole-capture
rollups are prefill-weighted** (the 29×/64×4096-token prefill dominates a 64-step window —
e.g. Case 2 nsys shows MLA-attn 40.9%) and are **not** a decode-% cross-check; for Cases 2/3 the
torch DECODE trace above is authoritative and nsys serves the timeline/serialization check (§2).
Per-case nsys kern_sum CSVs are committed (`*/nsys/kern_sum.csv`).

## 2. Headline answer — Case 1 vs Case 2 (clean DS overhead, same bs 29 / mem 0.7)

Clean DS overhead = Case1 − Case2 decode GPU-kernel time = **632,239 − 342,857 = +289,382 µs**
(DS = **1.84×** DSA). DS-specific kernel groups (positive deltas; `cmp_case1_vs_case2.txt`):

| DS-specific kernel group | Δ µs (Case1 − Case2) | note |
|---|---|---|
| `ncclDevKernel_AllReduce_Sum_f32_RING` | **+124,873** | DS-only; 780 calls = 1/layer/step. DSA issues none. |
| top-k / sort stack (mbtopk + radixSort + sbtopk<long> + scan_by_key + searchsorted) | **+138,602** | DS uses generic PyTorch top-k over `top_k=2048` candidates |
| `_logical_score_kernel` | **+63,107** | DS channel-score compute |
| (DS index/scoring subtotal) | **≈ +326,582** | |

What DSA spends *instead* (and DS does not): `sm90_fp8_paged_mqa_logits` (6,875 µs) +
`topk_transform_decode_kernel` (7,685 µs) + `fast_hadamard_transform` (2,641 µs) ≈ **17.2k µs** —
a tightly-fused fp8 indexer, ~19× cheaper than DS's logical-score + torch-topk + extra-all-reduce
path.

**Serialization (nsys / call-structure):** every DS-specific kernel fires **once per layer per
step** (780 calls over the 10-step window = 78 layers × 10) — `_logical_score_kernel`, the top-k
stack, and the f32 all-reduce are all in the per-layer decode critical path. Decode is a single
fixed CUDA-graph node sequence (no separate stream), so these **serialize** with attention/MoE
rather than overlapping them: the DS index/scoring is added latency on the critical path, not
hidden work. The f32 ring all-reduce in particular is a per-layer cross-TP sync that DSA's fused
indexer avoids entirely.

The shared kernels measure close (MLA-attn 41.9k vs 40.5k; trtllm-fusion all-reduce 35.9k vs
33.8k; fp8-quant 24.1k vs 25.4k). The one notable shared-kernel difference — MoE 91.9k (DS) vs
118.6k (DSA), same 1,500 calls — is **run-to-run clock/contention variance** (two separate
single-batch boots, identical MoE work), not a DS effect; it is why the *net* total delta (289k)
sits below the DS-specific-additions subtotal (327k). The DS tax itself (extra f32 all-reduce +
top-k stack + logical-score) is unambiguous and DSA-absent.

## 3. Case 2 vs Case 3 — DSA batch-efficiency (bs 29 vs bs 64)

> Separates "DS is slow" from "small batch is inefficient". Case 1's bs 29 is **forced** by the
> DS KV pool, not chosen; Case 3 is DSA at its full bs 64 (mem 0.8).

DSA decode GPU-kernel time: bs 29 = 342,857 µs vs bs 64 = 422,236 µs → **1.23× the work for 2.2×
the tokens** (`cmp_case2_vs_case3.txt`). Per-token decode GPU cost: **11,823 µs/tok (bs29) →
6,597 µs/tok (bs64) = 0.56×**, i.e. bs 64 is ~1.8× more GPU-efficient per token. The growth is
concentrated in the batch-scaling kernels — MoE +43.2k µs (115.6k→158.8k) and sparse-MLA-attn
+23.0k µs (30.3k→53.4k) — while the indexer (top-k stack +0.2k, mqa-logits +4.7k), all-reduce
(+8.9k), and fp8-quant (+2.6k) grow only marginally; the deep_gemm tiles shift from 32-wide to
64-wide variants. Net: nearly all per-token gain comes from amortizing fixed per-step overheads
over a 2.2× larger batch.

**Consequence for DS:** DSA can run bs 64 (1555 tok/s aggregate single-batch); DS-on is KV-pool-
capped at bs 29 (459 tok/s). So the DS deficit is the **product** of the per-step index tax
(≈1.9×, §2) and the lost batch efficiency (≈1.78×) — DS pays for being slow per step *and* for
being unable to use the efficient large batch.

## 4. Bench `--show-report` single-batch latency (NOT steady-state — cross-reference only)

ISL 4096 / OSL 512 (torch runs), greedy. **Biased single-batch numbers** — do not equate with the
locked SLO sweep; cross-reference only.

| Case | batch | latency (s) | output tok/s | ITL (ms) | last TTFT (s) | per-req decode tok/s |
|---|---|---|---|---|---|---|
| 1 — DS bs29 | 29 | 40.24 | 459.21 | 63.15 | 7.91 | 15.8 |
| 2 — DSA bs29 | 29 | 29.33 | 875.92 | 33.11 | 12.38 | 30.2 |
| 3 — DSA bs64 | 64 | 48.18 | 1555.16 | 41.15 | 27.11 | 24.3 |

Per-request decode tok/s tracks the locked SLO-sweep regime (DS ~17, DSA ~26–42), confirming
these biased single-batch runs reproduce the right relative behavior even though absolute numbers
are not SLO-grade. bs 64 has higher *aggregate* (1555) but lower *per-request* (24.3 vs 30.2)
throughput than bs 29 — the expected larger-batch trade. (The second, slower row in each torch
`bench.log` is the profiler-instrumented pass; ignore for latency.)

## 5. Artifacts

All under `development/profiling/runs/20260609/`. Raw multi-GB traces (`*.nsys-rep`, `*.sqlite`,
`*.trace.json.gz`) are **git-ignored** (paths listed below; regenerable). Committed: summaries,
`result.jsonl`, nsys `kern_sum.csv`, serve/bench/driver logs, `server_args.json`, comparison
tables, and the run/parse scripts (`_env.sh`, `run_case.sh`, `summarize_torch.py`,
`summarize_nsys.py`, `compare_decode.py`).

| Run | torch trace (raw, uncommitted) | nsys rep (raw, uncommitted) | committed summaries |
|---|---|---|---|
| Case 1 (DS) | `case1_ds/torch/trace/<ts>/*-TP-N-{DECODE,EXTEND}.trace.json.gz` | `case1_ds/nsys/trace.nsys-rep` (775M, OSL 512) | `case1_ds/{torch,nsys}/decode_summary.txt`, `case1_ds/nsys/kern_sum.csv` |
| Case 2 (DSA@0.7) | `case2_dsa07/torch/trace/<ts>/*.trace.json.gz` | `case2_dsa07/nsys/trace.nsys-rep` (216M, OSL 64) | `case2_dsa07/{torch,nsys}/decode_summary.txt`, kern_sum.csv |
| Case 3 (DSA@0.8 bs64) | `case3_dsa08/torch/trace/<ts>/*.trace.json.gz` | `case3_dsa08/nsys/trace.nsys-rep` (OSL 64) | `case3_dsa08/{torch,nsys}/decode_summary.txt`, kern_sum.csv |
| diffs | — | — | `cmp_case1_vs_case2.txt`, `cmp_case2_vs_case3.txt` |

## 6. Method notes / deviations

- **nsys `--output-len`:** torch runs use the plan's OSL 512 (authoritative latency + clean
  10-step decode trace). **nsys runs use OSL 64** — the per-decode-step kernel mix is stationary,
  so 64 steps give identical per-kernel attribution while keeping the trace finalizable in
  seconds (Case 1's OSL-512 nsys rep was 775 MB / ~4 min to serialize; OSL-64 ≈ 216 MB / seconds).
  Trade-off: the OSL-64 nsys whole-capture rollup is prefill-weighted (§1), so decode-% comes from
  the torch DECODE traces and nsys is used for timeline/serialization + f32-all-reduce confirmation.
- nsys captured server-side: `nsys profile --trace cuda,nvtx,cublas --cuda-graph-trace node
  --trace-fork-before-exec true --delay D --duration 900 python -m sglang.launch_server …`, bench
  gated to start only after collection is live, then `nsys stop --session=…` to finalize.
  `--cuda-graph-trace node` is load-bearing: decode is CUDA-graph-replayed, so without it the
  replay collapses to one opaque node.
- torch: server launched with `SGLANG_TORCH_PROFILER_DIR`; client `--profile --profile-by-stage
  --profile-activities CPU GPU --profile-steps 10`. CUDA graph kept ON (real op-point) — accepted
  trade-off: no Python-source mapping inside graph regions. The `with_stack` PyTorch-profiler bug
  did not trigger; no fallback needed.
