# Round 0 Summary — Loop 6 Milestone 1 (strategic gate + feasibility budget)

## Mainline objective (round contract)
Deliver **Milestone 1**: the two analyze-only, pre-coding gates that the whole loop depends on — AC-1 (strategic recall-R&D decision) and AC-2 (the binding HBM footprint-feasibility budget). No source/test/serve/bench code touched; no hardware run. This is the correct opening round: per loop discipline the spine starts "gate → feasibility → footprint", and AC-2's lever selection is binding on AC-3 (so footprint code must not start before this budget exists).

## What was done

### AC-1 — strategic gate (`analyze` → Codex, integrated by Claude)
Artifact: `runs/20260530_dsv32_loop6/ds_on_v32_decision.md`.
- **Decision:** pursue DS long-context-recall R&D on V3.2, but strictly **after** the engineering spine lands; direction = a **custom sparse-matmul DECODE kernel mirroring NSA/DSA with an adjustable `top_k`** (relaxing the `indices.shape[-1] == dsa_index_topk` hard cap); learned/query-aware selector secondary. DSA stays the production default; the DS compact path stays opt-in/reversible.
- **Rationale (evidence-grounded):** DS `top_k` is kernel-locked to the native DSA `index_topk=2048` (the shared `flashmla_kv` decode kernel asserts `indices.shape[-1]==dsa_index_topk`, not bypassable by `SGLANG_DS_ALLOW_TOPK_MISMATCH=1`); DS NIAH recall 75/5/0 at 4K/16K/64K vs DSA 100 at the **same** budget+kernel; dense (seq≤2048) DS recall=100% proves DS decode is sound → the gap is **selection quality** vs the trained DSA indexer, not budget size, and widening `top_k` is not an available lever without a new kernel.
- **Sequencing:** gated behind this doc AND a landed spine; must not block/regress the spine; legitimately deferrable to its own loop. Note: DS's value proposition is stronger on a model with no trained sparse indexer (deferred GLM-5.1 / 128k).

### AC-2 — footprint feasibility budget (`analyze` → Codex, integrated + verified by Claude)
Artifact: `runs/20260530_dsv32_loop6/footprint_feasibility.md`.
- **Grounded in real Loop-5 hardware anchors** (verified against `runs/20260528_dsv32_mvp/` boot logs and `token_label_table.py`): f=0.6 → table 1.55 GB, `max_total=53056`, headroom 37.78 GB, **serves** (admits 35.7/64); f≈0.70 → table 11.52 GB, `max_total=396096`, headroom 12.29 GB, **gen-OOM**; f=0.897 → table-alloc 31.18 GiB with 7.20 GB free, **boot-OOM**. Table formula `61·max_tokens·16·16·2`; KV ≈ 46.9 KiB/token.
- **Admission target:** 53056/35.7 = 1486 tok/admitted-req → **≥95K** tokens to admit conc 64, **~114K** with a 20% margin.
- **Per-lever budget** (freed-HBM, scale overhead, `f` needed, table bytes, predicted headroom, predicted conc-64) for: (i) fp16 + raise `mem_fraction_static` (no code), (ii) **int8 same-`label_dim`** (0.5625× fp16, ~1.78× win — scale overhead included), (iii) page-level/two-stage (~1/64×, NIAH-non-regression class).
- **Binding decision:** int8 is **predicted sufficient** → int8 is the selected **compaction** lever (build int8, not page-level). **But** the budget honestly rules the no-code fp16-f-bump *in*: the ~114K target sits at `f≈0.62`, far below the `f≈0.70` known gen-OOM, so the cheapest path is a **hardware fp16 mem-fraction sweep first** (`f≈0.612 / 0.625 / 0.650`); int8 is built only if that sweep fails (allocator residual / fragmentation / unmodeled gen memory). Page-level reserved for int8 insufficiency.
- **Primary equivalence gate fixed for AC-3.1:** `top-k overlap@2048 ≥ 0.99` vs fp16 (binding); secondary diagnostics (selected-token recall, score-error dist, rank displacement, NIAH trend) recorded-only.
- **Min-reversible-opt-in justification** + caveats (predicted budget; NVML/torch-residual HBM accounting + no-OOM long-generate is the hardware confirmation; conc-64 TTFT may be prefill-bound → admission-vs-prefill attribution required downstream).
- **Claude verification note appended** to the artifact: anchors confirmed verbatim from logs; corrected the Anchor-B label (`≈0.77-0.8` → `≈0.70`, matching the recorded "0.7 OOMs during generation"), which makes the fp16-sweep window conservative; flagged that the linear headroom fit overestimates real headroom by ~3.7 GiB at 396K, so hardware accounting is authoritative.

## Files created/modified
- `runs/20260530_dsv32_loop6/ds_on_v32_decision.md` (new, AC-1 artifact)
- `runs/20260530_dsv32_loop6/footprint_feasibility.md` (new, AC-2 artifact)
- `.humanize/rlcr/2026-05-30_06-27-19/goal-tracker.md` (completed truncated IMMUTABLE ACs; populated Active Tasks; logged 2 evolution entries; moved AC-1/AC-2 to Completed-pending-verification)
- `.humanize/rlcr/2026-05-30_06-27-19/round-0-contract.md` (new)
- `.humanize/rlcr/2026-05-30_06-27-19/round-0-summary.md` (this file)

## Tests added/passed
None — this is an analyze-only round (two decision/budget documents). No code, so no unit tests. The selection-equivalence unit test (`top-k overlap@2048 ≥ 0.99`) is **specified** here and will be implemented with the AC-3 footprint code.

## Remaining items (downstream, gated on this round)
- **Next round (AC-3/AC-4 sequencing):** per the AC-2 finding, the next hardware step should **sweep fp16 DS at `f≈0.612/0.625/0.65` first** (cheapest minimum lever) with full HBM accounting + no-OOM long generate (AC-4). Build the int8 compact `TokenLabelTable` (AC-3, flag-gated, fp16 default, CUDA-graph-safe) only if the fp16 sweep fails to admit conc-64 with generation headroom. **Do not** touch the FlashMLA `indices.shape[-1]==dsa_index_topk` assert in any spine work (AC-3.3).
- AC-5 client-SLO benchmark (with admission-vs-prefill attribution), AC-6 opt-in/DSA-default, AC-7/AC-8 hardening (soft), AC-9 within-budget-from-real-tokens (opportunistic), AC-10 recall R&D (gated on AC-1 + landed spine) — all later rounds.

## Note for review
The AC-2 budget makes a non-trivial, evidence-based recommendation that the **no-code fp16 mem-fraction bump may be the true minimum lever** (test-first), while still selecting **int8** as the binding *compaction* lever per the footprint ladder. This refines the AC-4/AC-3 execution order (sweep fp16 first; int8 conditional) and is logged in the Plan Evolution Log — it stays within the plan's Lower/Allowed bounds ("minimum lever", "0.7 acceptable as a conservative first step", "not mem_fraction=0.8 as a number in itself"). Flagged here explicitly for Codex's judgment.

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: Analyze-only round (two decision/budget documents); no code defect was discovered or solved, so no new lesson is warranted (lessons are added when a problem is solved, not for documentation rounds). Existing lessons were applied as cited context, not modified: BL-20260529-ds-longcontext-needle-recall-vs-topk (kernel-lock + selection-quality gap) grounds AC-1; BL-20260529-ds-vs-dsa-memfraction-admission-asymmetry (the mem-0.6 admission asymmetry, achieved 14.5/24.6/35.7, "0.7 OOMs during generation") grounds AC-2.
