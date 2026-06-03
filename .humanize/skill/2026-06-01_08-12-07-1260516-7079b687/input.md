# Ask Codex Input

## Question

Read `development/loop7/plan.md` in this repo. It already contains 8 inline review comments wrapped in `<comment>...</comment>` tags (7 critiques + 1 positive note), written in a blunt Linus-Torvalds register. For grounding, also read the files the plan cites: `runs/20260530_dsv32_loop6/ds_on_v32_decision.md`, `runs/20260528_dsv32_mvp/ac12_analysis.md`, `python/sglang/srt/layers/attention/dsa_backend.py` (the `_forward_flashmla_kv` / `_forward_flashmla_sparse` paths and the `indices.shape[-1] == dsa_index_topk` assert), `python/sglang/srt/layers/attention/dsa/dequant_k_cache.py` (`dequantize_k_cache_paged`), and `python/sglang/srt/layers/attention/double_sparsity/{selection_kernel.py,config.py,metrics.py}`.

The plan is for "Loop 7": DeepSeek-V3.2 FP8 double-sparsity (DS) long-context RECALL R&D in SGLang. Six decisions are already RESOLVED (DEC-1 measure-first→B→A-if-evidence; DEC-2 recall floor=recorded+characterized, strict=stretch; DEC-3 128k→its own loop; DEC-4 production-ready bar for landed code; DEC-5 non-learned selector first; DEC-6 slower research decode path OK first). Do NOT re-argue the resolved decisions — assess whether the plan HONORS them.

Your task, in two parts:

PART 1 — VERDICT ON THE EXISTING 8 COMMENTS.
For each existing `<comment>` (identify it by its lens tag, e.g. "[SEQUENCING / blocker]", "[DON'T RE-OPEN DECIDED SCOPE / major]", "[EVIDENCE / major]", "[EVIDENCE / minor]", "[COST HONESTY / major]", "[MINIMUM LEVER / major]", "[CLEAN SEPARATION / minor]", "[ON-RAILS / positive]"): say AGREE / PARTIALLY AGREE / DISAGREE in one line plus one sentence of concrete reasoning grounded in the code or the cited docs. If any existing comment is wrong or overstated, say so plainly.

PART 2 — NEW, INDEPENDENT CRITIQUES.
Add critiques the existing 8 comments do NOT already make (no duplicates, no restating). Apply the same six lenses: SEQUENCING, MINIMUM LEVER, CLEAN SEPARATION, DON'T RE-OPEN DECIDED SCOPE, COST HONESTY, DURABLE EVIDENCE. Hunt specifically for things a fresh reviewer would catch, e.g.:
- AC/task coverage gaps: are there ACs with no task, or tasks with no AC? Is the "zero hot-path cost when oracle disabled" claim (AC-1) actually testable/enforced, and by which task?
- TP correctness: the selector does an all-reduce of token scores across TP=8 ranks (`all_reduce_token_scores`); does any AC/test pin cross-rank determinism of selected indices for the new scorer variants (AC-3) or the lifted-budget path (AC-4)? `selector.py` raises `DoubleSparsityTPMisconfigured` / `DoubleSparsityRebindError` — are those failure modes covered?
- Oracle validity: AC-1.1 "force the needle into the selected 2048" — by what mechanism, and does forcing it perturb the very scores being measured? Is the score-only recall@K computed on the SAME all-reduced scores the live selector uses, or a re-derived copy that could drift?
- Determinism/seed: NIAH recall numbers (75/5/0) — does the plan pin sampling params / seeds / needle placements so the baseline is recomputable, or could run-to-run variance swamp a "material uplift"?
- Statistical power: the ac12 baseline is 20 trials/length (e.g. 16K = 1/20 = 5%). Is N=20 enough to call a "material" 16K uplift, or does the plan need a stated trial count / confidence?
- Tier-2.B ABI honesty: AC-3 says non-learned scorer "byte-identical when flag off" but anchor-budget/normalization change selection granularity — is "byte-identical default" actually guaranteed by the gating, and is there a test?
- config/ABI surface: AC-4 adds a new opt-in lifted-budget config field, but `config.py` rejects unknown fields and reserves Twilight names — does the plan say which exact field name/validator path, and does it avoid colliding with the reserved Twilight ABI?
- Anything else genuinely missed.

OUTPUT FORMAT for Part 2: for each new critique emit exactly one block:
<comment>CRITIQUE — [LENS / severity] ANCHOR: "<short exact quote or section heading from plan.md where this attaches>". <1-4 sentence Linus-voiced, file-grounded critique>. FIX: <smallest change>.</comment>

Prioritize signal over volume — a handful of sharp, genuinely-new findings beats a long list. If a lens has nothing new worth saying, skip it rather than invent a nit.

## Configuration

- Model: gpt-5.5
- Effort: xhigh
- Timeout: 3600s
- Timestamp: 2026-06-01_08-12-07
- Tool: codex
