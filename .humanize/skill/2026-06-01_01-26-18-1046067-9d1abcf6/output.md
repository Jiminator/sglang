CORE_RISKS:
- The biggest unproven assumption is that `top_k > 2048` fixes recall. First measure the needle’s DS score rank at 4K/16K/64K. If the needle is usually rank 20K+, a 4096/8192 decode path will not help.
- DSA’s 100% recall with the same 2048 kernel proves the decode kernel is sound, but it does not prove DS is budget-limited. It may be scorer-limited.
- Larger budgets can still hurt or do nothing: more distractors enter the softmax, needle may still be unselected, latency may collapse, or FP8 dequant/noise may change attention behavior.
- Tier-2.A is understated. The assert in [dsa_backend.py](/sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa_backend.py:2150) is only the visible cap; FlashMLA metadata, `compute_dsa_seqlens`, graph buffers, page mapping, and fixed-shape capture also assume a budget ABI.
- Tier-2.B is too vague. “Learned/query-aware” needs a training source, artifact format, layer/head placement, runtime cost budget, and compatibility with the existing label table.
- Tier-2.C risks polluting the experiment. 128K admission failures, KV memory pressure, and radix behavior can mask whether recall changed.

MISSING_REQUIREMENTS:
- Define “competitive recall”: exact NIAH lengths, number of trials/seeds, needle placements, prompt template, judge rule, sampling params, and pass threshold.
- Add selector telemetry before kernel work: per-request needle rank, selected_contains_needle, valid_lengths, score distribution, and recall@K for K = 512/1024/2048/4096/8192.
- Record channel-mask provenance: calibration set, layer range, label dim, weights, fp16 vs int8, hash/path, and whether it was recalibrated for V3.2 FP8 long-context.
- Require fail-fast config behavior: DS `top_k > 2048` must error unless an opt-in non-`flashmla_kv` decode path is explicitly selected.
- Require TP correctness checks: all ranks must produce identical selected-index hashes after `all_reduce_token_scores`; missing process group must remain fatal.
- Require CUDA-graph criteria per budget bucket: fixed `[bs, max_top_k]` shape, no capture-time allocation, no host sync, deterministic replay, and separate capture for 2048/4096/8192 if needed.
- Add perf guardrails: TTFT, decode TPS/request, GPU memory, graph replay success, and admission at conc-1/16 at minimum.
- Add default-path non-regression: DSA, fp16 non-DSA, DS default 2048, dense DS <=2048, and MMLU must remain unchanged within agreed tolerance.

TECHNICAL_GAPS:
- `flash_mla_sparse_fwd + dequantize_k_cache_paged` is a good prototype path, not yet a production plan. It likely changes the FP8 KV path into a dequantized KV path, so numerical tolerance, temporary memory, graph capture, and latency must be proven.
- `flash_mla_sparse_fwd` accepts variable topk-shaped indices, but production still needs a fixed max shape for CUDA graphs. Variable budget should mean “fixed configured max_top_k with padded entries,” not dynamic tensor shapes.
- Padding semantics are unspecified. If selected indices contain `-1`, duplicated pads, or out-of-range logical positions, the sparse kernel path must define masking or safe replacement before launch.
- Top-p/nucleus is not a drop-in for the current DS config. [config.py](/sgl-workspace/sglang/python/sglang/srt/layers/attention/double_sparsity/config.py:5) explicitly rejects `selection_mode`, `top_p`, `min_top_k`, and `max_top_k`.
- Top-p over DS scores is not obviously meaningful unless scores are calibrated into a probability-like distribution. Softmax over 64K/128K scores also adds cost and graph-shape constraints.
- Learned selection needs an artifact contract: where weights live, how they are loaded, whether they are TP-sharded, and whether they score from existing token labels or require new label writes.

ALTERNATIVE_DIRECTIONS:
- Lead with B as a falsification probe: instrument baseline DS rank curves first. If the needle is often rank <=8192 but >2048, Tier-2.A has evidence. If not, fix selector quality before kernel work.
- Use `flash_mla_sparse_fwd + dequantize` as an opt-in experimental decode path to validate recall at 4096/8192 before writing a new FP8 adjustable-budget kernel.
- Consider DSA-as-teacher distillation: use DSA selected indices or needle inclusion labels to train/calibrate a DS scorer that still outputs the existing `[bs, 2048]` ABI.
- Try hybrid selection within 2048: DS top-score budget plus protected recency/global/strided anchors. This may improve NIAH without kernel work, but must be checked against MMLU and non-NIAH tasks.
- Scope Tier-2.C out unless 128K is a hard deliverable. If kept, make it a separate smoke gate after recall experiments, not part of the recall hypothesis.

QUESTIONS_FOR_USER:
- Is Loop 7 expected to deliver production-ready code, or is DEC-3-style measured evidence enough?
- What recall target counts as “competitive”: DSA parity, 16K material uplift, nonzero 64K, or a fixed percentage?
- Are learned artifacts allowed, including DSA-teacher data or new calibration files?
- Is a slower opt-in DS research path acceptable if it proves recall, or must it preserve Loop-6 throughput targets?
- Should 128K servability be a required Loop-7 gate or moved to its own loop?

CANDIDATE_CRITERIA:
- Baseline report: for DS default 2048, record NIAH recall plus needle-rank histograms and recall@K curves for 4K/16K/64K on the same 8xH200 TP=8 node.
- Tier-2.B gate: new selector is flag-gated; default selected indices remain equivalent; 16K recall improves materially over 5%; 64K is characterized; MMLU and dense DS <=2048 do not regress.
- Tier-2.A gate: opt-in decode supports configured fixed budgets 4096 and 8192; default DSA `dsa_index_topk` assert remains untouched; `top_k > 2048` without opt-in decode fails at startup.
- Kernel correctness: sparse decode output matches a reference sparse attention implementation on small deterministic cases for fp16/bf16 and FP8-KV/dequant path within defined tolerance.
- Graph safety: replay test proves zero allocation, stable shapes, no host sync, and identical selected-index hashes across eager/captured paths.
- Perf floor: report TTFT, decode TPS, memory, and admission for 2048/4096/8192 at conc-1 and conc-16; recall wins that exceed the agreed slowdown/memory budget do not pass.
- 128K, if included: `/generate` with 128K prompt returns first token without HTTP 400/OOM, logs selected valid lengths and memory headroom, and clearly separates servability from recall.
