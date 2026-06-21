# Round 2 Contract

## Mainline Objective
Complete the close-out cleanup Codex's Round-1 review left open: finish stripping plan/workflow
markers from shipped DS files (the Round-1 sweep missed `Option B`, `Tier-2.A`, `Round 3`, `round 1`
and a stray blank line), and correct the false native-DSA band statement in the provenance doc — so
the branch is genuinely clean for final close-out.

## Target ACs
- **AC-1** (branch hygiene / clean shipped diff + comments per plan:417 Code Style). No other AC
  changes: AC-2..AC-10 are Completed and Verified; AC-8 was accepted in the Round-1 review. Do not
  re-litigate or re-run GPU work.

## Blocking Side Issues
- None. Both remaining items are [P3] documentation/comment fixes; they do not affect any AC's
  runtime behavior or the accepted AC-8 perf proof.

## Queued / out-of-scope
- No new code/logic changes. No re-benchmarking (AC-8 evidence stands: 256/256, 35.05 TPS / 22.90 s).
- Do NOT touch pre-existing base-code comments (surgical principle); only reword markers in the
  shipped DS diff files.

## Success Criteria
1. Reword `Option B` (config.py:8, page_table_adapter.py:9, validator.py:218/251/262) and `Tier-2.A`
   (validator.py:220/251) as durable technical language (e.g. "DSA index-topk operating point",
   "lifted-budget path").
2. Reword `Round 3` (calibrate.py:428) as the actual fix, and `round 1` (topk_kernel.py:78) as
   "first radix pass".
3. Remove the stray blank line flagged by `git diff --check` (metrics.py:281).
4. Fix DOUBLE_SPARSITY.md: state the native-DSA column is same-base context only; the accepted
   corrected-shape run is the DS run (256/256, 1 group, request_shape_ok=true). Remove the false
   "both DS and DSA meet the band" claim.
5. Comprehensive marker re-sweep over the shipped diff is clean (no AC-/DEC-/Milestone/Option/Tier/
   Round/round-N workflow markers, excluding DeepSeek-R1 model names + base-code Step/Phase);
   `git diff --check` clean; import gate + 114 unit tests pass; corrected branch re-pushed.

## Constraints
Push only to `Jiminator/sglang`; do not modify base-code comments; keep AC-8 evidence intact.
