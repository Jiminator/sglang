CORE_RISKS:
- The memory argument is not yet a fixed-point calculation. `max_tokens` grows with the KV pool, so raising `mem_fraction_static` also grows the TokenLabelTable and score buffers. Int8 does not simply “save 4 GB”; per-slot/head scales add back memory, and the larger pool may eat much of the gain.
- P99 TTFT may not fall below 22 s even if admission improves. At 4096 ISL, conc 64, TTFT can be prefill-compute-bound, scheduler-bound, or prefix-cache-miss-bound. The draft assumes queueing is dominant but does not require proving that.
- Compact signatures can hurt the already weak selector. DS recall is 75% / 5% / 0%; int8 noise or narrower `label_dim` can reorder near-threshold top-k tokens and make quality materially worse while still passing TPS.
- Higher `mem_fraction_static` may reintroduce generation OOM through transient buffers, not just static table allocation. `compute_token_scores` produces fp32 score tensors over token slots; larger pools increase scratch and possible all-reduce payload.
- Per-slot int8 scales may reduce bandwidth less than expected. For `label_dim=16`, loading int8 signatures plus fp16/fp32 scales plus extra multiply may save HBM but cost kernel time, possibly threatening the 30 TPS/req margin.
- The Tier 2 FlashMLA `top_k > 2048` direction is probably not one-loop feasible. The decode path has a hard `indices.shape[-1] == dsa_index_topk` assertion, and the kernel/scheduler metadata likely assumes the native DSA budget shape.

MISSING_REQUIREMENTS:
- A concrete HBM budget equation for weights, KV pool, TokenLabelTable, scales, `written`, score scratch, FlashMLA metadata, CUDA graph pools, allocator fragmentation, and safety headroom.
- Quantization spec: scale granularity, scale dtype, zero-scale handling, rounding/clamp rule, saturation behavior, invalidation semantics, and whether scales are updated atomically with signatures.
- Selection-equivalence requirements for int8 vs fp16 DS. Top-k does not need exact match, but the acceptable overlap/recall delta must be explicit.
- Label-dim compatibility with the existing Loop-5 mask. If `label_dim` is narrowed, the draft must say whether slicing the existing mask is valid or whether mask regeneration is required.
- CUDA graph requirements for the compact scoring path: no dynamic allocation, no host sync, no dtype-dependent shape change, no hidden `.contiguous()` allocation in captured replay.
- A TTFT attribution requirement: admission wait, prefix-cache lookup, prefill compute, first decode step, and scheduler delay need to be separable.
- Failure behavior when `usage.prompt_tokens` is missing or inconsistent. The AC-12 gate should fail closed, not silently fall back to a proxy.
- DSA non-regression requirement. Since V3.2 has native DSA, DS flags must not allocate DS tables or alter decode behavior when DSA is selected.

TECHNICAL_GAPS:
- The footprint → admission → SLO chain is plausible but not yet sound. It needs evidence that the freed table memory converts into enough admitted concurrency at the chosen pool size, and that admitted concurrency is the TTFT limiter.
- Raising pool size can make DS selection more expensive. If the active path scans physical slots or all-reduces `[bs, max_tokens]` scores, the lifted operating point may improve queueing but degrade decode latency.
- Int8 scoring is CUDA-graph-safe only if the scales are preallocated, static-shaped, and read entirely on device. The current Triton kernel expects fp16/fp32 signatures; adding int8 must avoid Python-side dtype dispatch inside captured paths.
- Quantize-on-write can increase prefill cost. For TTFT, this matters because label writes happen during prompt processing, exactly on the critical path to first token.
- “Tighter slot model” is underspecified. The current table is intentionally physical-slot-indexed to match `out_cache_loc`; shrinking it without changing allocator/data authority risks stale reads or out-of-bounds writes.
- HTTP 200 at 64K is too weak as a servability bar. It does not prove useful latency, budget correctness, non-OOM stability, or recall.

ALTERNATIVE_DIRECTIONS:
- Stronger first direction: make the HBM fixed point explicit before choosing int8 vs `label_dim`. The target should be “pool slots/admission achieved with headroom,” not `mem_fraction_static=0.8` by itself.
- Try a small `label_dim` sweep only if the existing mask semantics support it. It is the lowest implementation-risk footprint lever, but highest quality-risk lever.
- Prefer int8 with fp16 per-slot/head scales over fp32 scales unless evidence shows fp16 scale error hurts selection. This preserves most of the memory win.
- Consider page-level or two-stage labels: select candidate pages cheaply, then refine tokens within selected pages. Tradeoff: larger selector redesign, but it attacks the physical-slot memory coupling directly.
- Avoid offloading the table for the hot decode selector. CPU/NVMe offload is likely incompatible with per-token decode latency, though it may be useful for cold diagnostics.
- If TTFT is prefill-bound after admission improves, shift effort to prefill scheduling/chunking/cache-hit behavior rather than more selector compression.
- Treat DEC-2 opt-in framing as a serious endpoint: DSA default for DeepSeek-V3.2, DS compact path shipped as experimental or for models without trained sparse indexers.

QUESTIONS_FOR_USER:
- Is long-context quality a release gate, or is the single done-criterion truly only P99 TTFT and TPS?
- Can DS ship for V3.2 if it meets SLO but remains far below DSA on NIAH recall?
- Are production serve flags allowed to differ from the fixed Loop-5 scripts, especially around CUDA graphs and overlap scheduling?
- What minimum HBM safety margin is required after warmup and after the full 320-prompt run?
- Is mask regeneration allowed if narrower `label_dim` is chosen, or must the existing safetensors mask be reused exactly?
- Are server-side admission-wait and prefill timing metrics already available, or must the benchmark infer them from logs?

CANDIDATE_CRITERIA:
- At the chosen operating point, log and assert: KV pool slots, table GB, scale GB, peak allocated/reserved HBM, free HBM after warmup, and no monotonic memory growth over the full run.
- For SLO validation, require NUM_PROMPTS=320 at conc 16/32/64 with P99 TTFT < 22 s and per-request generation TPS >= 30, using the real cache-hit distribution.
- Add TTFT breakdown criteria: P99 admission wait, P99 prefill compute, and P99 time-to-first-decode-step. The SLO claim should say which component improved.
- For int8, compare compact vs fp16 DS on identical cached prompts/queries: top-k overlap@2048, selected-token recall, score error distribution, and NIAH non-regression.
- Negative tests: stale slot invalidation, max physical slot boundary, zero labels/zero scales, mismatched table size, missing `usage.prompt_tokens`, and DS disabled while DSA remains default.
- CUDA graph criterion: compact path must run under the intended production graph settings, or the draft must explicitly declare graph-disabled serving as the measured shipping mode.
