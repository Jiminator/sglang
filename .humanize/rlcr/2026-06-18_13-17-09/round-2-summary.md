# Loop 12 — Round 2 Summary

## Mainline objective
Complete the close-out cleanup Codex's Round-1 review left open (both [P3], no AC gap): finish
stripping plan/workflow markers the Round-1 sweep missed, and correct a false native-DSA band
statement in the provenance doc. AC-8 was accepted in the Round-1 review (10/10 ACs).

## Work Completed

### [P3] Remaining plan/workflow markers — stripped
The Round-1 sweep targeted `AC-`/`DEC-`/`Milestone`/`Loop-N`/`[R-N]` but missed other workflow
labels. Reworded as durable technical language (no logic change):
- `Option B` → the DSA index-topk operating point / dropped: `config.py:8`,
  `page_table_adapter.py:9`, `validator.py` (×2, comment + user-facing error message).
- `Tier-2.A lifted-budget path` → `lifted-budget path`: `validator.py` (×2, comment + error message).
- `Round 3` → the actual fix description (flat-slicing would pick V/RoPE columns from later heads):
  `calibrate.py:428`.
- `round 1` → `first radix pass`: `topk_kernel.py:78` (radix-pass wording).
- Removed the stray EOF blank line in `metrics.py` (`git diff --check` was flagging it).

### [P3] Provenance doc native-DSA statement — corrected
`benchmarks/DOUBLE_SPARSITY.md` had claimed "both DS and DSA meet the band once the workload shape is
fixed", but native DSA's 46.50 s P99 TTFT is **not** in the ≤30.1 s band and there is no
corrected-shape DSA evidence. Fixed: the native-DSA column is now labelled **same-base context only**
(an earlier run made before the wrapper pinned the GSP grouping, explicitly "not in band"); the
**accepted** result is the DS run (256/256 completed, `request_shape_ok=true`, 35.05 TPS / 22.90 s).
Removed the false claim.

## Files Changed (v2 clone, R2)
- `python/sglang/srt/layers/attention/double_sparsity/{config,page_table_adapter,validator,calibrate,topk_kernel,metrics}.py` (comment/marker rewording + EOF blank line; no logic change)
- `benchmarks/DOUBLE_SPARSITY.md` (native-DSA note corrected)
- v2 commit `323cb7802`; branch re-pushed.

## Validation
- Final marker sweep over the full `<BASE>...HEAD` diff: **0** plan/workflow markers in added lines
  (DeepSeek-R1 model names + base-code Step/Phase preserved by design).
- `git diff --check` clean; AC-1 diff still 42 files, **0** dev-scaffolding, **0** dropped-module refs.
- `import sglang` OK; **114 unit tests pass** (no logic touched).
- Branch re-pushed to `Jiminator/sglang`; HEAD == remote `323cb7802`.

## Remaining Items
None. All 10 ACs pass; both Codex reviews' findings (R0 [P1] AC-8 workload shape + [P3] markers; R1
[P3] remaining markers + [P3] doc) are resolved. PR:
`https://github.com/Jiminator/sglang/pull/new/double-sparsity-v2`.

## BitLesson Delta
- Action: none
- Lesson ID(s): NONE
- Notes: R2 applied existing requirements (plan:417 code-style marker ban) and corrected doc prose;
  no new reusable technical lesson. The relevant lessons are already recorded
  (`BL-20260619-perf-parity-pin-request-shape`, `BL-20260619-ds-selector-width-ladder`,
  `BL-20260619-latest-main-base-drift`).
