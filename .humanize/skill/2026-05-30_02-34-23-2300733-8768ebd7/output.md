AGREE:
- The main reframe is now correct: success is admitted KV capacity plus HBM headroom sufficient to hit absolute client SLO, not “mem_fraction=0.8” as a goal by itself.
- Tier-2 is properly gated. Given the FlashMLA `indices.shape[-1] == dsa_index_topk` constraint, recall work involving `top_k > 2048` is not a small config tweak.
- Keeping compact DS behind a flag, fp16 as default until hardware validation, and DSA non-regression as a negative criterion is sound.
- AC-7 is the right fix direction: actual prompt tokens must replace the word-count proxy, and the DS-fair gate definition should not drift.

DISAGREE:
- AC-4 is internally inconsistent: positive says conc 16 and 64, but the project scope is conc 16/32/64. A conc-32 miss fails the SLO claim.
- DEC-6 should not be a pending MVP decision. The verified target is fixed Option B. Different production flags can be characterized later, but cannot be used to satisfy this goal.
- AC-2 conflates compaction classes. Same-`label_dim` int8 quantization can be checked against fp16 selected-token equivalence; narrower `label_dim`, page-level, or two-stage models change the selector and cannot honestly be held to “preserve numerics.”
- “Target factor” is too vague. Int8 plus per-slot/head scales may only save ~1.6-1.8x depending scale dtype, which may not be enough to restore nominal admission.
- “Documented attribution method if server lacks the split” is too weak. The SLO report needs measured attribution or it should say attribution is unavailable and avoid root-cause claims.
- Radix-on is under-enforced. `development/serve_double_sparsity.sh` defaults DS to radix-off unless a fixture artifact is supplied, so AC-4 must prove radix-on from sidecars/server args.
- The strict SLO says `P99 TTFT < 22 s`; current comparator-style logic using `<= 22` would be a false pass at the boundary.

REQUIRED_CHANGES:
- Rewrite AC-4 to require all concurrencies 16/32/64, full 4096 ISL / 512 OSL / ~55% cache / `NUM_PROMPTS=320`, strict `P99 TTFT < 22.0`, and `per_request_output_tps_p50 >= 30`.
- Define SLO trial aggregation before running: all trials pass, or median pass with worst-trial disclosed. Do not allow failed trials to disappear behind a summary.
- Add a pre-coding feasibility gate: compute required freed HBM, scale overhead, target `max_total_num_tokens`, and expected achieved concurrency at conc 64 before picking the compaction lever.
- Split AC-2 validation paths: quantization equivalence for same `label_dim`; regenerated/saved mask artifact plus quality/perf validation for narrower `label_dim` or structural selector changes.
- Lock Tier-1 decode ABI explicitly: `top_k == dsa_index_topk == 2048`, output width padded/valid for FlashMLA, no Tier-1 attempt to bypass that assert.
- Require actual TTFT attribution data: client queue/admission wait vs prefill compute at minimum.
- Make artifact routing explicit. Existing AC-12 harness writes `development/results`; Loop-6 acceptance says `runs/<date>_dsv32_loop6/`.
- Remove DEC-6 from the MVP path or state it is post-MVP only.

OPTIONAL_IMPROVEMENTS:
- Add microbenchmarks for compact label write and selector scoring; compaction can trade OOM for prefill latency.
- Include NVML/torch residual memory in HBM accounting, not only named tensors.
- Reword AC-6 so a 70K failure is “characterized ceiling,” not a positive pass.
- Add config tests for the compact flag path and DSA boot proving no DS table allocation.

UNRESOLVED:
- DEC-1: whether Tier-2 recall R&D is worth pursuing after the Tier-1 SLO result.
- DEC-4: exact compaction lever and required HBM headroom, but this should be constrained by the feasibility calculation above.
- DEC-2: shipping default may remain DSA with DS opt-in, but it must not weaken the MVP success definition.
- DEC-5: topology matters for execution logistics; the SLO claim still needs the fixed TP=8 Option-B target.
