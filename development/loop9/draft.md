# Loop 9 Draft — Close the DS-on decode-overhead gap on GLM-5.1 (perf loop)

> Written 2026-06-10, after **Loop 8 closed** (disk `development/loop8/` = roadmap Loop 10, GLM-5.1
> DS bring-up; AC-4 closed R12, DS landed default-OFF). That loop produced a one-batch profiling
> characterization (`development/profiling/`) that **attributes** exactly where DS-on decode spends
> its extra time vs DSA-native. This loop's whole job is to **drive that measured gap down**.
> Feed this through `gen-plan` once scope is confirmed. **This is a draft of a draft — open for
> improvement.**

---

## What this work is (and is NOT) — read first

**This is not frontier LLM development. There is no novelty here.** We are reproducing the results
of a **2-year-old paper** (Double Sparsity, arXiv:2408.07092, `development/past_implementations/double_sparsity_paper_2408.07092.pdf`)
on top of a **fully open-source codebase** (SGLang), and checking whether it can be made to perform
on **MLA models that are well behind the frontier** (GLM-5.1). The contribution, if
any, is engineering: making an existing sparse-attention idea cheap enough to be worth its opt-in
slot on a model that already ships a *trained* sparse indexer (DSA). This is purely educational and the algorithm theoretically has no hope of beating the highly optimized DSA algorithmn.theres a reason why the algorithm was introduced 2 years ago and nobody has actually decided to use it in a frontier model, we just want to see how it would perform and learn about basic performance engineering skills when doing so.

## Objective

Take the **measured DS-on decode overhead** from `development/profiling/results.md` and **close the
gap** against the DSA-native baseline — by attacking the DS-specific kernels directly. The scope of
a single idea may be anything from a **small perf fix or debug** up to a **full rewrite / large
refactor / algorithm redesign** of a key component — *provided* it:

1. **Keeps the core concept of Double Sparsity** (offline channel-importance mask → per-token
   signatures → query·signature scoring → top-k token selection → sparse MLA decode over the
   selected tokens). Reference implementations: `development/past_implementations/` (DoubleSparse,
   Twilight, sglang-last-with-double-sparsity, the paper, and the as-built study `study/`).
2. **Introduces no additional theoretical lossiness.** The set of tokens DS selects (and therefore
   its recall/quality) must be **equivalent to today's** — pure speed/memory wins only. Numerics may
   move only where they provably do **not** change selection (verify, don't assume — §"Guardrails").

**We are not chasing the SLO in this loop.** The target is the **Case-1 (DS-on) one-batch
profiling benchmark** we just built — nothing more. SLO re-validation, recall R&D, multi-node, knob
compat, nvfp4 — all deferred (roadmap untouched except to mark this the active direction).

---

## The gap we are closing (from `development/profiling/results.md`)

One-batch profiling (GLM-5.1-FP8, 8×H200 TP=8, `bench_one_batch_server`, ISL 4096 / OSL 512) gave
three **frozen reference points** (DSA does not change as we optimize DS — so we re-profile **only
Case 1** going forward):

| Case | Config | decode GPU-kernel µs / 10-step window (torch TP-0) | aggregate decode tok/s | role |
|------|--------|------|------|------|
| **1 — DS** | DS ON, mem 0.7, bs 29 | **632,239** | 459 | the thing we are making faster |
| **2 — DSA@same** | DSA, mem 0.7, bs 29 | 342,857 | 876 | **apples-to-apples target** (same bs/mem; only DS flags differ) |
| **3 — DSA@best** | DSA, mem 0.8, bs 64 | 422,236 | 1555 | best-DSA aggregate target |

The deficit is **two separable, multiplicative penalties**, and DS pays both:

### Penalty A — DS index/scoring tax ≈ 1.9× (Case 1 − Case 2 = **+289,382 µs** at the SAME bs 29)
DS does **not** add an index onto a dense path — it **replaces DSA's tightly-fused fp8 indexer
(~17.2k µs/window)** with a much heavier stack. The DS-specific additions (DSA issues **zero** of
these — confirmed: absent from Case 2/3 nsys kern_sums):

| DS-specific kernel group | Δ µs (Case1−Case2) | source site |
|---|---|---|
| `ncclDevKernel_AllReduce_Sum_f32_RING` (DS-only per-layer cross-TP score reduce) | **+124,873** (780 calls = 1/layer/step) | `all_reduce_token_scores` — `double_sparsity/selection_kernel.py:561` (called :1058,:1071,:1447) |
| generic PyTorch top-k/sort stack (mbtopk / radixSort / sbtopk<long> / scan_by_key / searchsorted) | **+138,602** | `select_topk_sequence_order` / `_topk_by_score_then_pos` — `selection_kernel.py:606`/`:585` |
| `_logical_score_kernel` (DS channel-score compute) | **+63,107** | `_logical_score_kernel` / `_logical_score_triton` — `selection_kernel.py:58`/`:1230` |

What DSA spends **instead** (and DS does not): `sm90_fp8_paged_mqa_logits` (6.9k) +
`topk_transform_decode_kernel` (7.7k) + `fast_hadamard_transform` (2.6k) ≈ **17.2k µs** — a fused
fp8 indexer ~**19× cheaper** than DS's logical-score + torch-topk + extra-all-reduce path. Core
sparse-MLA decode attention is ~31–42k µs in **both** Case 1 and Case 2 (clean parity) — so the gap
is the **index/scoring + its synchronization**, not attention math. Every DS-specific kernel fires
once per layer per step and **serializes** on the single CUDA-graph decode node sequence (added
critical-path latency, not hidden work).

### Penalty B — batch-efficiency ≈ 1.78× (DS is locked to bs 29)
DSA at bs 64 does only 1.23× the work for 2.2× the tokens (≈1.8× more GPU-efficient per token). DS
**can't reach bs 64**: its per-rank `TokenLabelTable` shrinks the KV pool and caps the decode batch
at ~29 (mem 0.7). This penalty is **memory-driven**, and shrinking the table further was already
rejected once (DEC-4, int8 was the floor) because coarser signatures **are** added lossiness — so
**Penalty A is the primary, clearly-lossless target**; Penalty B is secondary and only pursued via
**lossless** means (see Idea 4).

**Net:** DS bs29 459 → DSA bs29 876 → DSA bs64 1555 tok/s = **3.4× best-vs-best** ≈ 1.9× (A) × 1.78× (B).

---

## Candidate ideas (a menu — `gen-plan` picks/sequences; each is one implement→measure cycle)

Ordered by expected payoff-per-risk. Each idea names the pain-point kernel it kills and its lossiness
posture. **These are starting points, not a fixed list** — a better idea found while reading the code
or the reference impls replaces a worse one here.

1. **Kill the DS-only f32 ring all-reduce (Penalty A, +124,873 µs — biggest single item).**
   DS computes per-rank *partial* token scores (signatures are head/TP-sharded) then SUM-reduces them
   across the attention TP group in **fp32, once per layer per step** — 780 serialized NCCL calls
   DSA never makes. Levers, cheapest-first: (a) **fold it into the existing fused all-reduce path**
   (`enable_flashinfer_allreduce_fusion` is already ON — the trtllm fusion all-reduce shows up shared
   in both cases) instead of a standalone f32 ring; (b) **batch the reduction** — reduce all layers'
   scores in fewer, larger collectives instead of 78 separate ones; (c) **lower the reduce dtype**
   (fp32→bf16 halves traffic) — *numerics-sensitive, gate behind selection-equivalence*. (a)/(b) are
   lossless reshaping; (c) needs the recall-oracle check. *Scope: medium refactor of the
   score-reduce path.*

2. **Replace the generic PyTorch top-k/sort stack with a fused top-k (Penalty A, +138,602 µs).**
   `_topk_by_score_then_pos` does two `argsort`s + gathers over the full KV width to pick 2048 and
   re-sort to sequence order; this explodes into mbtopk/radixSort/sbtopk/scan/searchsorted. DSA does
   the equivalent in one `topk_transform_decode_kernel` (~7.7k µs). Lever: a **fused selection kernel**
   (adapt DSA's `topk_transform`, or a Triton/CUDA top-k that emits sequence-ordered indices directly)
   producing **bit-identical** selected indices to today's deterministic (value-desc, then pos-asc)
   tie-break. *Lossless if output indices match — verify against `selection_recall_oracle.py`.
   Scope: new kernel / large refactor of `selection_kernel.py` selection path.*

3. **Fuse the logical-score compute (Penalty A, +63,107 µs).**
   `_logical_score_kernel` computes query·signature scores as a separate Triton pass over the full
   table. DSA fuses scoring into `sm90_fp8_paged_mqa_logits`. Lever: fuse scoring into the
   selection kernel (compute-and-select in one pass, killing the intermediate score scratch), and/or
   keep signatures fp8 end-to-end so scoring rides the fp8 path DSA uses. *Numerics-sensitive (int8
   dequant, head-agg) — gate behind selection-equivalence. Scope: kernel redesign.*
   > Ideas 1–3 together target the full ≈+326k µs DS index/scoring subtotal. Even partial wins move
   > Case 1 toward Case 2's 342,857 µs floor. The ceiling for A alone is ~1.84×→~1.0× on the index tax.

4. **Lossless batch-efficiency wins only (Penalty B — secondary).**
   Do **not** coarsen signatures (that's added lossiness, DEC-4). Instead: (a) any scratch/device-buffer
   that Ideas 1–3 eliminate **frees KV-pool memory** → admits a larger decode batch at the same mem
   fraction — couple the wins; (b) audit the `TokenLabelTable` for **layout/padding waste** that can be
   reclaimed without changing what's stored. Measure whether the freed memory lifts the bs-29 cap.
   *Lossless by construction (no signature change). Scope: memory-layout debug + re-measure admission.*

5. **Wildcard — redesign from the reference implementations.**
   If profiling after Ideas 1–3 still shows DS structurally heavier than DSA, consider a larger
   redesign of the selection path informed by `development/past_implementations/` (Twilight's selection,
   the original DoubleSparse kernels) — still DS-concept, still lossless. Last resort; only if the
   incremental kernel fusions plateau short of the goal.

---

## The iterate→measure protocol (the heartbeat of this loop)

For **each** idea, run exactly one cycle:

1. **Implement** the idea (DS path only; DSA-native default must stay byte-identical when DS is off).
2. **Verify losslessness FIRST** (before trusting any speed number): the DS selection output must
   match the pre-change path — use `double_sparsity/selection_recall_oracle.py` / the dense-DS and
   within-budget sanity checks. If selected tokens changed (beyond provably-irrelevant numerics),
   the idea **fails the no-added-lossiness bar** — revert or fix, do not bank the speedup.
3. **Re-profile Case 1 only**, reusing the **exact** setup in `development/profiling/plan.md` (Case 1
   server args: `$COMMON_ARGS --mem-fraction-static 0.7 --enable-double-sparsity --double-sparsity-config "$DS_CONFIG"`,
   bs 29, ISL 4096 / OSL 512). **One trial per run.** Cases 2 & 3 are **frozen references — do NOT
   re-run them** (DSA is unchanged; reuse `development/profiling/runs/20260609/case2_dsa07/` and
   `case3_dsa08/`). Both captures (torch per-stage decode + nsys `--cuda-graph-trace=node`) as in the
   plan, or torch-only if a quick read suffices and nsys adds nothing for that idea.
4. **Read the gap.** Compare the new Case-1 decode-GPU-kernel breakdown against the frozen Case 2
   (342,857 µs) and Case 3. **Success = the gap closed**, and specifically the **targeted DS-specific
   kernel group shrank** (e.g. after Idea 1 the `AllReduce_Sum_f32_RING` line should be gone/folded;
   after Idea 2 the mbtopk/radixSort stack should collapse toward DSA's `topk_transform` cost). Record
   per-idea: Case-1 total decode µs, the targeted-kernel Δ, aggregate decode tok/s, and the
   selection-equivalence result.
5. **Keep or revert.** Keep only ideas that (a) close the gap and (b) pass the lossless check. Stack
   kept ideas; carry the running Case-1 number forward as the new DS baseline for the next idea.

Reuse the committed parsers from the profiling run (`development/profiling/runs/20260609/`):
`summarize_torch.py`, `summarize_nsys.py`, `compare_decode.py` — and `development/profile_ds.sh`'s
kernel-aggregation block. Write each idea's result into a running ledger
(`development/loop9/results.md`, create it) with the same kernel-breakdown table shape as
`development/profiling/results.md`, one column added per idea.

---

## Acceptance criteria (draft — `gen-plan` formalizes)

1. **Gap closed and measured.** Case-1 DS-on decode GPU-kernel time is **materially lower** than the
   632,239 µs baseline and **measurably closer to Case 2's 342,857 µs** (DSA@same bs/mem), attributed
   to the specific DS kernel groups that shrank. Each landed idea has a before/after Case-1 profile.
   *(`gen-plan` sets the numeric bar — e.g. a target % of the +289,382 µs Penalty-A tax recovered;
   stretch = approach the Case-2 floor / reach into Penalty-B via Idea 4.)*
2. **No added theoretical lossiness.** Every landed change has a recorded selection-equivalence check
   (recall oracle / within-budget / dense-DS) showing DS selects the same tokens as before. Any
   numerics change is shown not to alter selection.
3. **Core DS concept intact.** Still offline channel mask → signatures → score → top-k → sparse decode.
   No silent fallback to dense or to DSA's indexer.
4. **DSA-native default un-regressed.** DS-off path byte-identical; the shipped model untouched.
5. **One trial per profiling run; Case 1 only re-profiled** (Cases 2/3 reused frozen). Ledger in
   `development/loop9/results.md`.

## Files to read first

- **The gap (the whole point):** `development/profiling/results.md` (+ artifacts under
  `development/profiling/runs/20260609/`, the `cmp_case1_vs_case2.txt` / `cmp_case2_vs_case3.txt` diffs).
- **The benchmark to reuse verbatim (Case 1 only):** `development/profiling/plan.md`.
- **The DS selection path to optimize:** `python/sglang/srt/layers/attention/double_sparsity/selection_kernel.py`
  (`all_reduce_token_scores`, `_logical_score_kernel`, `select_topk_sequence_order`), plus
  `token_label_table.py`, `token_label_write.py`, `selector.py`, `selection_recall_oracle.py`.
- **DSA's fused indexer (the cost target to emulate):** `python/sglang/srt/layers/attention/dsa_backend.py`
  (`sm90_fp8_paged_mqa_logits`, `topk_transform_decode`).
- **Lossless bar / DS concept:** `development/past_implementations/` (paper, study/, the three impls).
- **Doctrine:** `CLAUDE.md` (surgical changes, fix the data structure not the symptom, prove with numbers).

## Hardware / op-point

Single node 8×H200, TP=8, FP8 e4m3, page 64, fp8 KV, custom-all-reduce ON. **Never set
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments` for serving** (breaks custom-all-reduce-v2 IPC at GLM
TP=8 graph capture — BL-20260608). One TP=8 server at a time. Model + mask paths and the full Case-1
`DS_CONFIG` are in `development/profiling/plan.md`.

## Pending decisions (resolve in `gen-plan` discussion)

- **The numeric success bar for AC-1** — what fraction of Penalty A (the +289,382 µs / 1.9× tax) must
  be recovered to call the loop done? (Suggest: a concrete µs/ratio target on the Case-1 decode total
  and on each targeted kernel group, not a vibe.)
- **Ordering / stopping** — do all of Ideas 1–3, or stop once the bar is hit? Is Idea 4 (Penalty B)
  in scope this loop or its own follow-on?
- **Lossless-equivalence definition** — bit-identical selected indices, or a bounded recall-delta the
  user accepts as "no theoretical lossiness"? (Default: bit-identical selection; numerics free to move
  only where selection provably doesn't.)
- **Build vs adapt for Idea 2/3** — write a new fused DS kernel, or adapt DSA's `topk_transform` /
  `mqa_logits` to the DS signature layout? (DS signatures differ from DSA's indexer; check feasibility.)
