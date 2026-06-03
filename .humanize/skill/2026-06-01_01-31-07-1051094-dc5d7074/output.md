AGREE:
- The measure-first pivot is reasonable. Existing evidence already points scorer-limited, but rank/recall@K diagnostics are the right way to quantify whether wider top_k has any oracle upside.
- Leading with Tier-2.B before Tier-2.A is technically justified. At 4K words, selecting about half the tokens and still missing 25% of needles is hard evidence that a larger budget alone is not the first lever.
- Keeping Tier-2.C servability separate from recall is correct. “Can serve 64K/128K” and “recalls the needle” are different failure classes.
- Preserving the default `flashmla_kv` DSA path and making lifted-budget decode opt-in is mandatory and the plan correctly states that.
- Fixed configured `max_top_k` plus padding is the right CUDA-graph direction if Tier-2.A is pursued. Dynamic top_k shapes would be a distraction.

DISAGREE:
- M0 is over-specified as “selector telemetry” and under-specified as an oracle. The selector does not naturally know the needle. Needle-rank requires harness-provided logical token positions, definition for multi-token needles, and rank computed before sequence-order sort.
- `recall@4096/8192` cannot be treated as normal selector output today. `config.top_k` is capped by the 2048 DSA ABI/validator/default decode assert. Higher-K curves must be score-only offline/oracle diagnostics, not a decode run, unless the opt-in ABI is already added.
- “Reproduce 75/5/0” is sloppy because the 64K evidence was unservable at `mem_fraction_static=0.6`, not necessarily a true 0% recall result. The plan must separate “HTTP/admission failure” from “served but missed.”
- M1 is too vague. The existing scorer is already query-aware. “Improved/query-aware/learned scorer” is not a plan unless it names concrete scorer changes and their data/artifact requirements.
- Making M1 “learned” is likely too heavy for one loop. Learned artifacts imply calibration data, artifact versioning, config/schema changes, reproducibility, and owner approval. Keep learned scoring out of the core loop unless explicitly chosen.
- The M2 prototype has a serious missing data-shape step. `dequantize_k_cache_paged` returns a compact dequantized KV tensor for a flattened page table; `flash_mla_sparse_fwd` then needs indices in that compact domain. Decode currently builds physical-slot `page_table_1`. Passing physical indices into compact KV would be wrong.
- M2 also misses that current `dequantize_k_cache_paged` allocates `torch.empty` internally. That conflicts with “zero-alloc replay” unless the plan includes an `out=`/scratch-buffer variant or another allocation-free decode dequant path.
- Padding entries are unsafe unless specified. `-1` padding may be acceptable to the attention kernel, but it cannot be fed blindly into dequantization because it can index the wrong physical slot.
- “top_k>2048 fails fast unless opt-in path selected” requires a real config/validator/backend ABI change. The existing config explicitly rejects the Twilight-style fields, and relying on `SGLANG_DS_ALLOW_TOPK_MISMATCH` would be too ad hoc for this plan.

REQUIRED_CHANGES:
- Define M0 as a harness/debug-oracle mode: source of needle token positions, multi-token needle rank rule, per-layer/per-step sampling, and score-only recall@K curves for K > 2048.
- Replace “baseline 75/5/0” with exact outcomes: 4K served recall, 16K served recall, 64K admission/servability status at the stated mem fraction.
- Make the A-vs-B gate an oracle-uplift gate: pursue Tier-2.A only if score-only recall@4096/8192 shows material recoverable recall over 2048. Needle rank merely being in `(2048, 8192]` is not enough.
- Split M1 into concrete non-learned candidates first: e.g. channel weighting/normalization changes, head aggregation changes, deterministic anchor budget experiments. Put learned/distilled scoring behind a separate owner decision.
- For M2, specify the decode ragged/compact mapping: physical selected slots -> flattened dequant input -> compact per-request indices for `flash_mla_sparse_fwd`.
- For M2, require allocation-free dequant scratch and q-padding scratch before claiming CUDA-graph safety.
- Add explicit opt-in ABI for lifted decode budget and validator behavior. Do not use the existing mismatch env var as the main mechanism.
- Add tests for invalid padding, duplicate selected indices, valid_lengths, R23 tie-break, fp8 dequant correctness, and graph replay allocation behavior at 4096/8192.

OPTIONAL_IMPROVEMENTS:
- Record compact score summaries instead of full score distributions: needle score, kth thresholds, rank, selected margin, percentiles, and valid token count.
- Add a small “oracle dense-within-window” diagnostic: if needle is forced into the selected set within 2048, measure whether answer recall recovers. That separates selector miss from downstream attention/model behavior.
- Make hybrid anchors an ablation, not a default candidate, because they can improve NIAH while stealing budget from semantically relevant tokens.
- Keep 128K in a separate mini-loop unless the owner requires it for Loop 7. First make 64K servability unambiguous.
- Update the decision record after the owner resolves A-primary vs B-primary, because the current strategic gate is internally contradictory.

UNRESOLVED:
- Human owner must decide whether to override the existing decision record that names Tier-2.A as selected direction. Technically, B-first is better supported by the evidence.
- Recall gate hardness needs owner choice: strict target versus R&D-style “measured, characterized, no regression.”
- Learned scorer artifacts need explicit approval. They are not a small scorer swap.
- The owner must decide whether opt-in slower research decode is acceptable, or whether Loop-6 throughput constraints apply even to Tier-2.A experiments.
- 128K servability scope needs a yes/no decision for this loop; otherwise it should be documented as out of scope.
