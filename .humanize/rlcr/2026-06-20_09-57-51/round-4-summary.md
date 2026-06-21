# Round 4 Summary

## Work Completed
- **AC-2.3 RESOLVED** (the stalled Round-3 mainline), via a **cleaner direct method** instead of
  the shared-decode-step-id join Codex prescribed. The Round-3 stall was a score-vs-selection
  *step-misalignment* artifact (81/546), not a radix discrepancy. The direct proof sidesteps the
  pairing entirely: take each captured **post-reduce score row** (the authoritative top-k input the
  production radix consumed) and run BOTH top-k methods on the SAME row —
  - `select_topk_sequence_order` (exact torch reference == `torch.topk` semantics)
  - `blocked_topk_sequence_order` (the deterministic blocked/radix ALGORITHM the production Triton
    kernel `select_topk_sequence_order_triton` implements)
  — then compare selected-index sets. **Result: 624/624 identical**, and selector-width `[5120]`-vs-full
  **624/624 identical** on the same rows. The radix and selector-width AC-2.3 suspects are **retired on
  real GLM-5.1 score distributions** (not just on `topk_kernel.py` documentation). `verify_ac2_3.py` is
  fail-closed (nonzero exit on zero rows or any real mismatch).
- **Both real Round-3 review bugs fixed:**
  - `analyze_captures.py` is now **fail-CLOSED**: exits nonzero (rc=2) on zero score-capture groups,
    zero equivalence rows, or any unmatched join row. Verified it exits 2 on an empty capture dir.
  - `build_ledger.py` records **unambiguous generator-source provenance**: the generator file's
    git **blob hash** (commit-independent) + head-at-generation SHA + worktree dirty/clean marker, so
    the ledger source is pinned despite build_ledger emitting evidence one commit before its own commit
    exists. `run_meta.json` `git_sha_current` synced to HEAD; full SHAs.

## Files Changed
- `development/loop13/verify_ac2_3.py` (NEW) — the direct AC-2.3 proof; writes
  `evidence/ac2_3_radix_width_equivalence.json`.
- `development/loop13/evidence/ac2_3_radix_width_equivalence.json` (NEW) — 624/624 radix==torch.topk +
  624/624 width==full, on real captured rows.
- `development/loop13/analyze_captures.py` — fail-closed (nonzero exit on zero/unmatched rows).
- `development/loop13/build_ledger.py` — generator blob-hash + worktree provenance; regenerated
  `evidence/meta/arms/*.json`, `evidence/evidence_table.md`, `evidence/meta/run_meta.json`.
- `development/loop13/evidence/cheap_controls.json`, `evidence/findings.md` — AC-2.3 marked RESOLVED.
- `python/.../double_sparsity/selection_capture.py` — `req_pool_indices` retained (row identity).

## Validation
- `python3 development/loop13/test_reference_selectors.py` → **ALL 5 reference-selector tests pass**.
- `python3 development/loop13/verify_ac2_3.py` → **624/624** radix==torch.topk AND **624/624** width==full;
  wrote `evidence/ac2_3_radix_width_equivalence.json`; exit 0 (would exit 2 on any mismatch).
- `analyze_captures.py` on an empty dir → **exits 2** (fail-closed verified).
- Committed as `393966c02`; tree clean. **No server launched this round** (CPU-only analysis on already-captured
  artifacts); GPUs idle, one-TP=8-server-at-a-time invariant not exercised, not violated.

## Remaining Items (next mainline)
- **AC-6 production-path one-variable bisection** — the largest substantive gap. Reference-ceiling cliff
  (faithful raw-dot 0.013 vs faithful cosine 0.940 ≈ DSA 0.973) already names the candidate; the production
  path still needs a guarded diagnostic production-style cosine arm + per-variable arms (head_agg,
  fp8-vs-fp32 reduce, reduce dtype, radix, width) to attribute the 0.000 on the *served* path, not only the
  reference ceiling.
- **AC-2.1** forced-all physical-slot assertions; **AC-4** per-step length-cap garbage counters + per-example
  sample IDs/order (adapter/harness instrumentation, listed `fields_not_instrumented`, not faked).
- **AC-3.1** captured-row materialized-K proof; **AC-2.2** head-agg `pre_reduce` semantics confirmation.

## BitLesson Delta
- Action: update
- Lesson ID(s): BL-20260621-ds-capture-step-alignment
- Notes: The lesson originally flagged that score_capture and selection_capture use independent step
  counters so rows can't be paired at a decode step. Round 4 found the **cleaner resolution**: when the
  question is "does the production radix top-k match exact top-k," you don't need to pair captured selection
  rows to score rows at all — run BOTH the radix algorithm and exact `torch.topk` on the SAME captured
  *score* row and compare selected sets. The score row IS the authoritative top-k input, so this is
  conclusive and alignment-free. Added the corollary: prefer re-running both algorithms on one captured input
  over cross-instrument row joins when the captured input is the algorithm's direct argument.

## Goal Tracker Update Request
- Mark **AC-2.3 RESOLVED** (radix==torch.topk + width==full, 624/624 on real captured rows;
  `evidence/ac2_3_radix_width_equivalence.json`). The radix and selector-width suspects are retired.
- Promote **AC-6 production-path one-variable bisection** to the **next round's mainline** (largest remaining
  substantive gap; reference ceiling already names the cosine candidate).
- Keep **AC-2.2** (head-agg pre_reduce semantics), **AC-2.1/AC-4** (forced-all physical-slot assertions,
  garbage counters, sample IDs/order), and **AC-3.1** (captured-row materialized-K proof) active as
  instrumentation follow-ups.
- Note the two Round-3 review blockers (fail-open analyzer; stale ledger generated-SHA) are **CLOSED**.
