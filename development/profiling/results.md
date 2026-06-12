# DS-vs-DSA decode profiling + serving ladder — results (GLM-5.1-FP8, 8×H200)

**Re-measured 2026-06-12 on HEAD `ce914e5b9`** (branch `dev/double-sparsity-standalone`,
after the Loop-10 kernel work landed: width-bucketed DS selector CUDA graphs, compact W=5120
selection-capture, bf16-authoritative radix top-k with copy-back removal, and the pinned bf16
two-shot score reduce). This supersedes the original Loop-8 characterization (frozen run
`runs/20260609/`, HEAD ~`10e642c2f`); both runs share the identical op-point so the numbers
are directly comparable. New run: `runs/20260612/`.

**Two things changed vs the Loop-8 baseline, and the doc now leads with both:**

1. **Serving** — DS-on per-request decode throughput jumped **~1.7×** and now **clears the
   30 tok/s decode-SLO floor at conc 16/32/64**, where Loop-8 DS failed it badly.
2. **Kernel attribution** — the DS index/scoring/transport tax that dominated Loop-8 decode
   **collapsed**: same-batch DS-vs-DSA decode GPU-kernel time went from **1.84× to 1.05×**.

> **Scope caveat.** The serving sweep here is a **directional** read (1 trial, 60s warmup,
> 180s window, conc 16/32/64), **not** the locked AC-11 publication sweep (3 trials × 600s).
> Single-trial TTFT and aggregate-throughput numbers carry real run-to-run variance; the
> per-request decode-TPS p50 is the stable metric. The single-batch profiling below is
> `bench_one_batch_server` (one batch, never steady-state) — kernel attribution only, not SLO.

Op-point (every `serve.log`, both runs): TP=8, `page_size=64`, `kv_cache_dtype=fp8_e4m3`,
`dsa_{prefill,decode}_backend=flashmla_kv`, `disable_radix_cache=True` (both columns radix-off,
apples-to-apples), `disable_overlap_schedule`, `disable_piecewise_cuda_graph`, CUDA graph ON,
custom all-reduce ON, `enable_flashinfer_allreduce_fusion=True`, `random_seed=20260607`,
`PYTORCH_CUDA_ALLOC_CONF` unset (expandable_segments breaks custom-all-reduce IPC at GLM TP=8 —
BL-20260608). DS config: `top_k=2048`, fp16 signatures, `scorer_norm=off`, `head_agg=max`.

---

## 0. Headline

**The DS index/scoring tax is largely paid off; the remaining DS deficit is the structural
KV-pool batch cap, not the selector.**

- **Single-batch per-step (same batch, bs30, mem 0.7):** DS-on decode does **1.05×** the
  GPU-kernel work of DSA-native (361,786 vs 343,820 µs / 10-step window, TP-0) — down from the
  Loop-8 **1.84×** (632,239 vs 342,857). The DS-specific index/score/transport kernels now total
  **~74.5k µs** and replace DSA's **18.4k µs** fused indexer; net same-batch overhead is
  **+17,966 µs**, and the Loop-8 dominators are gone: the DS-only **f32 ring all-reduce
  (+124.9k µs) and the generic torch top-k stack (+138.6k µs)** were replaced by a **bf16
  two-shot reduce (11.3k)** and a **fused radix top-k (29.0k DS-only)**.
- **Batch efficiency (DSA bs30 → bs64):** unchanged at **1.72×** — DS is still KV-pool-capped
  at **bs30** (max_total_num_tokens 142,208 / 4608) while DSA reaches **bs64** (410,560 / 4608),
  so DSA decode is ~1.72× more GPU-efficient per token at its full batch.
- **Best-vs-best per-token GPU cost:** DS bs30 (12,060 µs/tok) vs DSA bs64 (6,667 µs/tok) =
  **1.81×**, decomposing as **1.05× (index tax) × 1.72× (batch efficiency)** — down from the
  Loop-8 **3.4× = 1.9× × 1.78×**. The index factor collapsed; the batch factor did not.

The core sparse-MLA decode attention is ~42k µs at bs30 in **both** DS and DSA (42,653 vs
42,148) — the clean apples-to-apples confirmation that the gap is DS's index/scoring +
synchronization, not attention math.

---

## 1. Serving ladder — DS vs DSA (directional, current HEAD)

`bench_serving`, gsp 4096-ISL / 512-OSL / ~55% prefix (radix-off, so no reuse), 1 trial,
60s warmup, 180s window. Per-request decode throughput = `output_tokens / (e2e − ttft)`;
SLO floor = 30 tok/s p50. Three configs:

- **DS @0.7 (bs30)** — DS-on, KV-pool naturally caps the decode batch at 30.
- **DSA @0.7 (bs30-cap)** — DSA-native at the *same* mem 0.7, batch forced to 30 via
  `--max-running-requests 30`. **This is the apples-to-apples column**: only DS-enablement differs.
- **DSA @0.8 (bs64)** — DSA at its full batch (KV pool reaches bs64). DSA's real deployment best.

| conc | config | decode-TPS p50 | agg tok/s | achieved conc | TTFT med/p99 (s) |
|---|---|---:|---:|---:|---:|
| 16 | DS @0.7 (bs30) | **39.4** ✅ | 494 | 16.0 | 3.6 / 3.7 |
| 16 | DSA @0.7 (bs30-cap) | 40.9 | 418 | 16.0 | 7.1 / 7.1 |
| 16 | DSA @0.8 (bs64, best) | 38.7 | 404 | 16.0 | 7.1 / 7.2 |
| 32 | DS @0.7 (bs30) | **33.0** ✅ | 574 | 26.0 | 6.4 / 28.4 |
| 32 | DSA @0.7 (bs30-cap) | 31.9 | 464 | 27.2 | 12.8 / 41.6 |
| 32 | DSA @0.8 (bs64, best) | 31.5 | 541 | 32.0 | 14.2 / 14.2 |
| 64 | DS @0.7 (bs30) | **33.3** ✅ | 577 | 39.4 | 23.9 / 45.2 |
| 64 | DSA @0.7 (bs30-cap) | 32.0 | 465 | 41.4 | 34.8 / 60.2 |
| 64 | DSA @0.8 (bs64, best) | 25.1 | 676 | 64.0 | 28.1 / 28.1 |

**Before/after — DS decode-TPS p50 (same op-point):** Loop-8 DS was **23.1 / 17.2 / 17.2**
(conc 16/32/64), all **below** the 30 floor. Current HEAD DS is **39.4 / 33.0 / 33.3**, all
**above** it — a 1.7× lift that tracks the 1.75× single-batch kernel-time improvement
(632,239 → 361,786 µs).

**How to read it honestly:**
- **Matched batch (DS bs30 vs DSA bs30-cap) — the clean comparison.** At the same batch the two
  are **within ~4% on decode-TPS** (essentially tied; the sign flips by concurrency within
  single-trial noise). The cleanest point is **conc 16**, where both fully admit bs16 with no
  queueing: **DSA 40.9 vs DS 39.4 → DSA 3.7% faster**, matching the **1.05× per-step kernel ratio**
  (§3) to the decimal. So DS's per-request decode cost is now ~5% above DSA at matched batch — the
  index tax is fully visible and *small* (was ~2× pre-Loop-10). DSA's apparent conc-64 "loss" in
  the old framing (25.1) was **only** its larger batch: cap DSA to bs30 and it recovers to 32.0.
- **The TTFT tail is the batch cap, not DS.** When DSA is *also* bs30-capped, its **p99 TTFT
  blows up too** — in fact worse than DS (41.6 / 60.2 s vs DS 28.4 / 45.2 s at conc 32/64). Both
  are admission/queue-bound once the running batch is capped at 30; DSA @bs64 keeps a tight tail
  (median ≈ p99) only because it admits the full batch. So the tail belongs to the small KV-pool
  batch cap, not to the DS selector. (The DS-vs-DSA-bs30-cap TTFT/aggregate gaps also reflect
  *how* each cap is imposed — DS by natural KV admission, DSA by the cruder `max-running-requests`
  throttle — so decode-TPS, a steady-state per-step metric, is the cleanest matched-batch number;
  treat the matched-batch TTFT/aggregate as noisier.)
- **DSA's real advantage is the batch it can run.** DSA @0.8 reaches bs64 and wins **aggregate**
  at high concurrency (676 vs DS 577 tok/s at conc 64) with a tight TTFT tail — the batch-
  efficiency win (§4) that DS cannot access while KV-pool-capped at bs30. This is the structural
  deficit that remains after the index tax was paid down; closing it needs a smaller per-rank
  TokenLabelTable, not selector tuning. (DS-on's served default remains OFF; DSA is the SLO
  default — unchanged.)

---

## 2. Per-case decode-step GPU-kernel breakdown (torch, TP-0, decode)

`--profile-by-stage` decode trace, rank TP-0, 10 decode steps over the 780-call window
(78 layers × 10). Classifier corrected to attribute the DS radix stack (`_radix_hist/_radix_scan/
_emit/_block_count/_block_prefix`) and split the shared output sampler (`gatherTopK`+`bitonicSort`,
present in all cases) from the DS-only kernels. Full grounding: `runs/20260612/breakdown.md`.

| category | Case1 DS bs30 µs | % | Case2 DSA bs30 µs | % | Case3 DSA bs64 µs | % |
|---|---:|---:|---:|---:|---:|---:|
| MoE (`fused_moe_kernel` + align/sum) | 95,182 | 26.3 | 123,804 | 36.0 | 167,091 | 39.2 |
| attention — sparse-MLA decode | 42,653 | 11.8 | 42,148 | 12.3 | 64,031 | 15.0 |
| all-reduce (total) | 45,061 | 12.5 | 35,038 | 10.2 | 44,311 | 10.4 |
| &nbsp;&nbsp;↳ DS-only bf16 two-shot reduce | 11,341 | 3.1 | 1,016 | 0.3 | 1,803 | 0.4 |
| &nbsp;&nbsp;↳ trtllm-fusion oneshot lamport | 33,258 | 9.2 | 33,558 | 9.8 | 41,784 | 9.8 |
| **DS:logical-score** (`_logical_score_kernel`) | 23,537 | 6.5 | 0 | 0.0 | 0 | 0.0 |
| DS radix top-k (total) | 49,513 | 13.7 | 20,582 | 6.0 | 20,790 | 4.9 |
| &nbsp;&nbsp;↳ **DS-only radix stack** | 29,010 | 8.0 | 0 | 0.0 | 0 | 0.0 |
| &nbsp;&nbsp;↳ shared output sampler | 20,503 | 5.7 | 20,582 | 6.0 | 20,790 | 4.9 |
| **DS index plumbing** (logical→physical, gathers) | 11,742 | 3.2 | 0 | 0.0 | 0 | 0.0 |
| **DSA fused indexer** (mqa-logits+topk-transform+hadamard) | 29 | 0.0 | 18,430 | 5.4 | 24,661 | 5.8 |
| fp8-quant for index logits | 28,031 | 7.7 | 30,389 | 8.8 | 33,647 | 7.9 |
| GEMM/proj (deep_gemm fp8 + nvjet) | 32,422 | 9.0 | 47,621 | 13.9 | 49,551 | 11.6 |
| norm/rope/elementwise | 33,511 | 9.3 | 25,701 | 7.5 | 22,472 | 5.3 |
| memcpy/set + other | 105 | 0.0 | 106 | 0.0 | 104 | 0.0 |
| **Total decode GPU-kernel µs / 10-step window** | **361,786** | 100 | **343,820** | 100 | **426,658** | 100 |

(Indented `↳` rows are sub-totals of the row above, not additive categories. The Case-1 DSA
fused-indexer "29 µs" is a scheduler metadata stub; the real DSA indexer kernels are absent under DS.)

**nsys cross-check (`runs/20260612/case1_ds/nsys/kern_sum.csv`, OSL 64):** the whole-capture
rollup is **prefill-weighted** (bs30 × 4096-token prefill dominates a 64-step window), so its top
kernel is the **prefill** `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` (31%) — a prefill transport
signal, **not** a decode-% cross-check (use the torch DECODE trace above for that). What it *does*
confirm: there is **no `…_Sum_f32_RING`** kernel anywhere — the Loop-8 DS-decode f32 ring
all-reduce is gone, and DS decode transport is the custom **bf16** path (two-shot + trtllm fusion).

## 3. Clean DS overhead — Case1 vs Case2 (same bs30 / mem 0.7)

Net decode GPU-kernel delta = **361,786 − 343,820 = +17,966 µs → DS = 1.05× DSA** (Loop-8:
+289,382 µs / 1.84×). Reading it by what each side actually runs:

| DS-specific kernel group (DSA runs none) | Δ µs | note |
|---|---:|---|
| `_logical_score_kernel` | **+23,537** | DS channel-score compute (1×780; Loop-8: 63,107) |
| DS-only radix top-k stack | **+28,931** | `_radix_hist`(4×780)/`_radix_scan`(4×780)/`_emit`/`_block_count`/`_block_prefix` — replaces Loop-8's +138.6k torch top-k stack |
| DS index plumbing | **+11,742** | logical→physical + scatter/gather + index_fill/copy + bf16/f16 selection copies (all 1–2×780) |
| DS-pinned **bf16 two-shot** all-reduce | **+10,325** | 11,341 µs / 800 calls vs DSA's 1,016 / 20 — replaces Loop-8's +124.9k f32 ring |
| (DS-specific subtotal) | **≈ +74,535** | |

What DSA spends *instead* (and DS does not): the fused indexer
`sm90_fp8_paged_mqa_logits` (6,963) + `topk_transform_decode` (7,678) + `fast_hadamard_transform`
(2,635) + `fused_store_indexer_cache` (1,126) = **18,430 µs**.

So DS's selection machinery is **~4× the cost of DSA's fused indexer in isolation** (74.5k vs
18.4k), but that is now a small slice of the decode step (was the dominant cost). The **net**
collapses to +17,966 µs because this capture's Case-2 happened to carry more MoE (+28.6k) and
GEMM (+15.2k) — per-boot `fused_moe_kernel` timing variance plus DSA's extra projection GEMMs,
**not** a DS saving. Treat the same-batch DS tax as bounded by **[+18k net, +56k variance-robust]**
(the +56k = +74.5k DS-specific − 18.4k DSA indexer, holding shared kernels equal). Either bound is
a **5–16× reduction** of the Loop-8 +289k tax.

**Serialization (unchanged structure):** every DS-only kernel still fires once (or a small fixed
multiple) per layer per step — `_logical_score_kernel`, the radix stack, the plumbing, and the
bf16 two-shot reduce are all 780-aligned, on the per-layer decode critical path inside the single
CUDA-graph node sequence (no separate stream), so they serialize with attention/MoE rather than
overlapping. The work is smaller now, but it is still added latency, not hidden.

## 4. Case2 vs Case3 — DSA batch-efficiency (bs30 vs bs64)

> Separates "DS is slow" from "small batch is inefficient". Case1's bs30 is **forced** by the DS
> KV pool (142,208 tok / 4608), not chosen; Case3 is DSA at its full bs64 (410,560 / 4608).

DSA decode GPU-kernel time: bs30 = 343,820 µs vs bs64 = 426,658 µs → **1.24× the work for 2.13×
the tokens**. Per-token: **11,461 µs/tok (bs30) → 6,667 µs/tok (bs64) = 0.58×**, i.e. bs64 is
**1.72×** more GPU-efficient per token. Growth concentrates in the batch-scaling kernels — MoE
+43.3k (123.8k→167.1k) and sparse-MLA-attn +21.9k (42.1k→64.0k) — while the indexer, all-reduce,
and fp8-quant grow marginally; deep_gemm tiles shift 32-wide → 64-wide.

**Consequence for DS:** DS-on is KV-pool-capped at bs30 and cannot reach DSA's efficient bs64.
With the index tax now ~paid off, **this batch cap is the dominant remaining structural reason**
served DS aggregate throughput trails DSA at high concurrency (§1). Lifting it needs a smaller
per-rank TokenLabelTable footprint (or a larger KV pool), not more selector tuning.

## 5. Bench `--show-report` single-batch latency (NOT steady-state — cross-reference only)

ISL 4096 / OSL 512, greedy, biased single-batch (do not equate with the §1 SLO sweep).

| Case | batch | latency (s) | output tok/s | ITL (ms) |
|---|---:|---:|---:|---:|
| 1 — DS bs30 | 30 | 22.82 | 919.0 | 32.6 |
| 2 — DSA bs30 | 30 | 29.78 | 903.0 | 33.2 |
| 3 — DSA bs64 | 64 | 47.38 | 1604.1 | 39.9 |

At the same bs30 the decode-token cost (ITL 32.6 vs 33.2 ms) is **within single-batch noise** of
DSA — consistent with the 1.05× same-batch kernel ratio (§3). bs64 has 1.78× the aggregate
single-batch output (1604 vs 919) but higher ITL (39.9 ms) — the larger-batch trade. The
authoritative throughput is the §1 `bench_serving` ladder, not these biased single-batch numbers.

## 6. Artifacts

New run under `runs/20260612/`. Raw multi-GB traces (`*.nsys-rep`, `*.sqlite`,
`*.trace.json.gz`) are regenerable and git-ignored; committed: per-case `serve.log`/`bench.log`/
`result.jsonl`, the directional `serving/*.jsonl` + `.meta.json` sidecars, `case1_ds/nsys/kern_sum.csv`,
the per-case torch `decode_summary.txt`, `cmp_case1_vs_case2.txt`, `breakdown.md`, the stage
drivers (`_env.sh`, `stage{1..4}_*.sh`), and the parsers (`summarize_torch.py`, `summarize_nsys.py`,
`compare_decode.py`). The Loop-8 frozen run `runs/20260609/` is retained unchanged as the "before".

## 7. Method notes / deviations

- **Common max batch** is derived analytically from each server's `max_total_num_tokens` /
  (4096+512), not by a destructive OOM probe: DS @0.7 → bs30, DSA @0.8 → bs64 (capped). DS gained
  one batch slot vs Loop-8 (bs29→bs30) from the Loop-10 memory work (shared per-width DSGraphState
  + compact buffers freed a little KV pool).
- **nsys OSL 64** (vs torch OSL 512): the per-decode-step kernel mix is stationary, so 64 steps
  give identical per-kernel decode attribution while keeping the trace finalizable in seconds.
  The trade is that the OSL-64 nsys whole-capture rollup is prefill-weighted (§2), so decode-% is
  read from the torch DECODE traces and nsys serves the timeline/serialization + transport-kind
  confirmation only.
- **Directional sweep, not AC-11:** 1 trial / 180s window / 60s warmup. The `benchmark_compare.py`
  `--ac11` mode (3 trials, 600s window, gates) would refuse these; the legacy single-trial mode
  is the right reader. For publication-grade DS-vs-DSA SLO numbers, re-run the locked sweep.
- **Classifier fix:** the copied `summarize_torch.py` had no rule for the Loop-10 radix kernels
  (`_radix_hist`/`_radix_scan`/`_emit`/…), dumping ~29k µs into "other"; `breakdown.md` uses a
  corrected classifier (`other` < 15 µs in every case) and splits the shared output sampler from
  the DS-only radix stack so the +28,931 µs DS delta is honest.
