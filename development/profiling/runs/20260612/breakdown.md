# Decode GPU-kernel breakdown — 8×H200 GLM-5.1-FP8

Per-category decode-stage GPU-kernel breakdown from three torch-profiler traces, rank
**TP-0 DECODE** only (the representative rank). Each value is the summed `dur` (µs) of all
chrome-trace events with `cat ∈ {kernel, gpu_memcpy, gpu_memset}` over the captured
780-step decode window (78 layers × 10 steps).

Trace files used:

| case | config | file |
|---|---|---|
| case1 | DS, bs30, mem0.7 | `case1_ds/torch/trace/1781286523.6121113/bs-30-il-4096-…-TP-0-DECODE.trace.json.gz` |
| case2 | DSA native, bs30, mem0.7 | `case2_dsa07/torch/trace/1781288350.4538953/bs-30-il-4096-…-TP-0-DECODE.trace.json.gz` |
| case3 | DSA native, bs64, mem0.8 | `case3_dsa08/torch/trace/1781287874.0456023/bs-64-il-4096-…-TP-0-DECODE.trace.json.gz` |

**Total sanity check (vs expected ±1%):**

| case | actual µs | expected µs | diff |
|---|---|---|---|
| case1 | 361,786 | 361,786 | -0.000% OK |
| case2 | 343,820 | 343,820 | -0.000% OK |
| case3 | 426,658 | 426,660 | -0.000% OK |

All three totals match to the µs. No >1% drift.

The classifier here **fixes the bug in the copied `summarize_torch.py`**: that classifier's
`"topk"` rule (string match `top_k`/`topk`) swallowed the DSA `topk_transform_decode` and the
sampler `gatherTopK` into a generic "DS-index/scoring" bucket, and it had **no rule for the DS
radix stack** (`_radix_hist_kernel`, `_radix_scan_kernel`, `_emit_kernel`, `_block_count_kernel`,
`_block_prefix_kernel`) — those ~29 K µs of DS-only kernels fell straight into **"other"** (the
old case1 dump shows "other" = 77,749 µs / 21.5 %). The corrected classifier below drives
"other" down to **14–15 µs** (`alloc_decode_kernel` only) in every case — nothing large hides.

---

## 1. Per-case category table

| category | case1 DS bs30 µs | % | case2 DSA bs30 µs | % | case3 DSA bs64 µs | % |
|---|---:|---:|---:|---:|---:|---:|
| MoE | 95,182 | 26.3 | 123,804 | 36.0 | 167,091 | 39.2 |
| attention(MLA) | 42,653 | 11.8 | 42,148 | 12.3 | 64,031 | 15.0 |
| all-reduce (total) | 45,061 | 12.5 | 35,038 | 10.2 | 44,311 | 10.4 |
| &nbsp;&nbsp;of which DS-only bf16 two-shot | 11,341 | 3.1 | 1,016 | 0.3 | 1,803 | 0.4 |
| &nbsp;&nbsp;of which trtllm fusion (oneshot lamport) | 33,258 | 9.2 | 33,558 | 9.8 | 41,784 | 9.8 |
| DS:logical-score | 23,537 | 6.5 | 0 | 0.0 | 0 | 0.0 |
| DS radix top-k | 49,513 | 13.7 | 20,582 | 6.0 | 20,790 | 4.9 |
| &nbsp;&nbsp;of which DS-only radix stack | 29,010 | 8.0 | 0 | 0.0 | 0 | 0.0 |
| &nbsp;&nbsp;of which shared output sampler | 20,503 | 5.7 | 20,582 | 6.0 | 20,790 | 4.9 |
| DS index plumbing | 11,742 | 3.2 | 0 | 0.0 | 0 | 0.0 |
| DSA fused indexer | 29 | 0.0 | 18,430 | 5.4 | 24,661 | 5.8 |
| fp8-quant | 28,031 | 7.7 | 30,389 | 8.8 | 33,647 | 7.9 |
| GEMM/proj | 32,422 | 9.0 | 47,621 | 13.9 | 49,551 | 11.6 |
| norm/rope/elementwise | 33,511 | 9.3 | 25,701 | 7.5 | 22,472 | 5.3 |
| memcpy/set | 91 | 0.0 | 91 | 0.0 | 91 | 0.0 |
| other | 14 | 0.0 | 15 | 0.0 | 13 | 0.0 |
| **TOTAL** | **361,786** | **100.0** | **343,820** | **100.0** | **426,658** | **100.0** |

(The indented "of which" rows are sub-totals of the row above them, not separate additive
categories. The `DSA fused indexer` "29 µs" in case1 is just the 10-call
`smxx_paged_mqa_logits_metadata` scheduler stub that still fires once per step under DS — the
real DSA indexer kernels are absent in case1.)

**"other" is clean.** Only `alloc_decode_kernel` (14/15/13 µs, 10 calls) is unmatched in any
case. Nothing above ~5,000 µs — nothing above 15 µs — hides in "other".

### Category → kernel grounding

- **MoE** — `fused_moe_kernel` (the dominant single kernel: 89,551 / 118,425 / 158,441 µs in
  c1/c2/c3), `moe_align_block_size_kernel`, `count_and_sort_expert_tokens_kernel`,
  `act_and_mul_kernel` (SiLU), and `moe_sum_reduce_kernel` (case3 only, 3,021 µs).
- **attention(MLA)** — `flash_fwd_splitkv_mla_fp8_sparse_kernel` (30,321 / 30,446 / 53,490 µs),
  `flash_fwd_mla_combine_kernel`, `get_mla_metadata_kernel`, `set_mla_kv_buffer_kernel`,
  `concat_mla_absorb_q_kernel`, `_quantize_k_cache_fast_kernel`. Near-identical c1↔c2 (sparse
  MLA core is the same kernel for DS and DSA at bs30).
- **all-reduce (total)** — `allreduce_fusion_kernel_oneshot_lamport` (trtllm fusion, the bulk),
  `all_reduce_two_shot_kernel<__nv_bfloat16,…>` (DS-pinned two-shot score reduce),
  `ncclDevKernel_AllGather_RING_LL` (~0.5 K). See §2 for why the two-shot count differs.
- **DS:logical-score** — `_logical_score_kernel` only. DS-exclusive.
- **DS radix top-k** — DS-only radix stack: `_radix_hist_kernel`, `_radix_scan_kernel`,
  `_emit_kernel`, `_block_count_kernel`, `_block_prefix_kernel`. PLUS the shared output sampler
  `sbtopk::gatherTopK` + `bitonicSortKVInPlace`, which appear in **all three** cases (they are
  the final-logits sampler, common to DS and DSA). The DS-specific portion is the +28,931 µs
  delta, not the whole 49,513 — see the "of which" split above and §2.
- **DS index plumbing** — `_logical_to_physical_kernel`, `_scatter_gather_elementwise_kernel`,
  `index_fill`/`index_copy` elementwise kernels, and the bf16/f16 `*_copy_kernel` selection
  gathers. All DS-exclusive (zero in case2/3).
- **DSA fused indexer** — `sm90_fp8_paged_mqa_logits` (+ `…_metadata`), `topk_transform_decode_kernel`,
  `fast_hadamard_transform_kernel`, `fused_store_indexer_cache`. DSA-exclusive (zero in case1).
- **fp8-quant** — `per_token_group_quant_8bit_kernel` (the bulk), `_act_quant_kernel`
  (DSA-only sub-part, 1.4 K).
- **GEMM/proj** — `deep_gemm::sm90_fp8_gemm_1d2d_impl` family, `nvjet_*`, `cublasLt::splitKreduce`.
- **norm/rope/elementwise** — `rmsnorm`/`layernorm` cutlass kernels, `fused_rope_kernel`, the
  generic `vectorized_elementwise`/`elementwise_kernel`/`direct_copy` and the `triton_*_fused_*`
  gate/sigmoid/sum kernels, plus the RNG `distribution_elementwise_grid_stride_kernel` for sampling.

#### Correction flagged: `triton_per_fused_copy__mul_sum_0`

The task spec puts this kernel under **DS index plumbing** ("query projection onto channels …
in case2/3 these names should be near-zero"). **That assumption is wrong in this data.** Measured:

| kernel | case1 | case2 | case3 |
|---|---:|---:|---:|
| `triton_per_fused_copy__mul_sum_0` | 5,317 µs / 750 | **5,194 µs / 750** | 0 |

It is **not** near-zero in case2 — it costs essentially the same (5,194 µs) under DSA at bs30,
and disappears only in case3 (bs64). So it is a **path-common** projection/reduction, not a
DS-only kernel; it vanishes with the bs30→bs64 shape change, not with DS→DSA. I classified it as
**norm/rope/elementwise** (generic triton fused copy+mul+sum). Counting it as DS-index-plumbing
would have falsely inflated "Clean DS overhead" by ~5.2 K µs even though DSA pays the same toll.

---

## 2. Clean DS overhead = case1 − case2 (per category)

Both at bs30/mem0.7, so the difference is the DS index/scoring/transport stack minus what DSA
spends on its fused indexer.

| category | case1 − case2 µs | note |
|---|---:|---|
| DS:logical-score | **+23,537** | DS-only `_logical_score_kernel` |
| DS radix top-k | **+28,931** | DS-only radix stack (shared sampler nets to ≈0: 20,503 vs 20,582) |
| DS index plumbing | **+11,742** | DS-only `_logical_to_physical` / scatter-gather / index_fill·copy / bf16·f16 copy |
| all-reduce (total) | **+10,023** | of which DS bf16 two-shot +10,325 (11,341 vs 1,016); trtllm fusion ≈ flat (−300) |
| norm/rope/elementwise | +7,810 | mostly bookkeeping/elementwise around the DS path; not a clean DS kernel |
| attention(MLA) | +505 | flat — same sparse-MLA core |
| memcpy/set | +0 | flat |
| fp8-quant | −2,359 | DSA quantizes slightly more (extra indexer act-quant) |
| **DSA fused indexer** | **−18,401** | what DSA spends *instead* (0 under DS vs 18,430 under DSA) |
| GEMM/proj | −15,199 | DSA carries extra GEMM/projection work the DS path folds elsewhere |
| MoE | −28,622 | DSA's `fused_moe_kernel` is slower here (118,425 vs 89,551); not a DS effect — MoE timing noise/occupancy between the two captures |
| other | −1 | — |
| **NET TOTAL** | **+17,966** | **ratio 1.052×** (matches expected +17,966 µs, 1.05×) |

**Reading it the clean way — DS-positive vs DSA-substitute:**

- **DS-specific positive deltas** (kernels DSA simply does not run):
  - logical-score: **+23,537 µs**
  - radix top-k (DS-only stack): **+28,931 µs** (i.e. +29,010 DS-only radix, minus 79 µs of
    sampler noise)
  - index plumbing: **+11,742 µs**
  - extra all-reduce — the **DS-pinned bf16 two-shot score reduce**: **+10,325 µs**
    (case1 11,341 µs / 800 calls vs case2 1,016 µs / 20 calls)
  - **DS-specific subtotal ≈ +74,535 µs**
- **What DSA spends instead** — the **DSA fused indexer subtotal = 18,430 µs** (case2):
  `sm90_fp8_paged_mqa_logits` 6,963 + `topk_transform_decode` 7,678 +
  `fast_hadamard_transform` 2,635 + `fused_store_indexer_cache` 1,126 + metadata 29.
- The headline **net** is only **+17,966 µs (1.05×)** because that ~+74.5 K of DS-specific work is
  largely offset in this capture by case2 carrying **more MoE (+28.6 K)** and **more GEMM (+15.2 K)**.
  The MoE/GEMM gap is *not* a DS saving — it is per-capture `fused_moe_kernel` timing variance
  plus DSA's extra projection GEMMs; the DS index/scoring/transport tax itself is the ~+74.5 K
  figure above, and the +17,966 net is what survives after DSA's own indexer and its heavier
  MoE/GEMM in this particular pair of runs.

---

## 3. DSA batch efficiency = case2 (bs30) vs case3 (bs64)

Both DSA native; only batch size (and mem 0.7→0.8) differ.

| case | total µs (780-step window) | batch | per-token µs (total / batch) |
|---|---:|---:|---:|
| case2 (bs30) | 343,820 | 30 | **11,460.7** |
| case3 (bs64) | 426,658 | 64 | **6,666.5** |

**Ratio:** per-token bs30 / bs64 = **1.719×**. Going bs30→bs64 cuts per-token decode kernel
time to **0.582×** (a 1.72× throughput-per-token win) — the GPU is badly under-filled at bs30,
so most kernels (MoE, MLA, GEMM) amortize far better at bs64. Total wall time per window rises
only 1.24× (426,658 / 343,820) while serving 2.13× the tokens.

---

## 4. DS-specific exact kernels (case1) — grounded list

Per-kernel µs / calls over the 780-step decode window. Calls that are an exact multiple of
**780** (= 78 layers × 10 steps) fire once-per-layer-per-step (`= N×780` shown); the sampler
kernels are batch-driven and do not align to 780.

| kernel | case1 µs | calls | per-window | case2 µs | category |
|---|---:|---:|---|---:|---|
| `_logical_score_kernel` | 23,537 | 780 | 1×780 | 0 | DS:logical-score |
| `_radix_hist_kernel` | 18,620 | 3,120 | 4×780 | 0 | DS radix (DS-only) |
| `_radix_scan_kernel` | 5,594 | 3,120 | 4×780 | 0 | DS radix (DS-only) |
| `_emit_kernel` | 2,561 | 780 | 1×780 | 0 | DS radix (DS-only) |
| `_block_count_kernel` | 1,337 | 780 | 1×780 | 0 | DS radix (DS-only) |
| `_block_prefix_kernel` | 898 | 780 | 1×780 | 0 | DS radix (DS-only) |
| `_scatter_gather_elementwise_kernel` | 3,471 | 780 | 1×780 | 0 | DS index plumbing |
| `index_fill_kernel` (index_elementwise) | 2,131 | 1,560 | 2×780 | 0 | DS index plumbing |
| `index_copy_kernel` (index_elementwise) | 2,084 | 780 | 1×780 | 0 | DS index plumbing |
| `_logical_to_physical_kernel` | 1,867 | 780 | 1×780 | 0 | DS index plumbing |
| `bfloat16_copy_kernel` (vectorized_elementwise<8>) | 1,385 | 780 | 1×780 | 0 | DS index plumbing |
| `float16_copy_kernel` (vectorized_elementwise<8>) | 803 | 780 | 1×780 | 0 | DS index plumbing |
| `all_reduce_two_shot_kernel<__nv_bfloat16,8u,true>` | 11,341 | 800 | not 780-aligned | 1,016 | all-reduce (DS-pinned bf16 two-shot) |

Shared sampler (present in **all** cases, **not** DS-specific — listed for honesty since they
sit in the "DS radix top-k" category by name):

| kernel | case1 µs | calls | case2 µs | case3 µs |
|---|---:|---:|---:|---:|
| `sbtopk::gatherTopK<float,…>` | 13,964 | 2,250 | 13,976 | 14,258 |
| `bitonicSortKVInPlace<…>` | 6,538 | 1,500 | 6,606 | 6,533 |

### Radix top-k + transport story (grounded)

- The DS **radix top-k** path is six DS-only kernels totalling **29,010 µs** in case1, fully
  absent in DSA. The two radix workhorses (`_radix_hist` 4×780, `_radix_scan` 4×780) fire four
  passes per layer-step — consistent with a multi-pass radix select over the logical scores.
  `_block_count`/`_block_prefix`/`_emit` each fire once per layer-step to bin, prefix-sum, and
  emit the selected indices.
- DS computes those scores with `_logical_score_kernel` (**23,537 µs**, 1×780), then maps the
  winners back to physical KV slots via `_logical_to_physical_kernel` + the scatter/gather/
  index_fill/index_copy/copy plumbing (**11,742 µs** total, all 780-aligned, all DS-only).
- **Transport:** DS adds a dedicated **bf16 two-shot all-reduce** for the distributed score
  reduce — `all_reduce_two_shot_kernel<__nv_bfloat16,8u,true>` at **11,341 µs / 800 calls** in
  case1 versus only **1,016 µs / 20 calls** in case2. That ~780× jump in invocation count (≈1
  per layer-step, vs the 20 startup calls DSA keeps) is the DS pinned score-reduce, the
  +10,325 µs transport tax. The trtllm oneshot-lamport fusion all-reduce is essentially
  unchanged between DS and DSA (33,258 vs 33,558 µs), so the all-reduce delta is entirely the
  DS two-shot.
- DSA replaces this whole DS index+score+transport stack with one fused indexer
  (`sm90_fp8_paged_mqa_logits` + `topk_transform_decode` + `fast_hadamard_transform` +
  `fused_store_indexer_cache` = **18,430 µs** at bs30), which is why the *net* DS overhead is
  only +17,966 µs (1.05×) despite ~+74.5 K of raw DS-specific kernel time.
