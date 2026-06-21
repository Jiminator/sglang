# Round 3 Summary

## Objective
Fix the two blocking gaps from Codex's R2 review: **AC-5** (`total_tokens_mean` was numerically wrong) and
**AC-8** (stale `queue.md` + unresolved push). AC-5 and the AC-8 ledger/evidence work are complete; the push
is now owner-authorized to a fork but blocked on a working GitHub credential in this environment (see below).

## Mainline gap 1 — AC-5: `total_tokens` metric was mislabeled (FIXED)
Codex found it and I verified on the committed R2 artifact: the DS publishers emit
`sparsity_rate = 1 - selected/total` (pruned fraction), but `bench_serving` derived
`total = selected / sparsity_rate` (assumes the KEPT fraction) → reported `total_tokens_mean=3588.7` vs the
true `selected/(1-sparsity_rate)=4770.3`. `trial_evidence.py` PASSed only because `2048 < 3588` still held.

Fix (`8df44a59c`, Codex's plan):
- Added explicit `total_tokens` to `DoubleSparsityRequestStats` + `meta_info_for_request`; BOTH publishers
  (GLM `dsa_backend.maybe_publish_ds_request_summary` + DeepseekV2) set it from the host `seq_len`.
- `bench_serving` captures + aggregates `total_tokens` DIRECTLY (and emits a per-request `total_tokens`
  array), with a backward-compat fallback `selected/(1-sparsity_rate)` only when `total_tokens` is absent and
  `0 ≤ sparsity_rate < 1`.
- `trial_evidence.py` STRENGTHENED: refuses when the reported aggregate disagrees with the per-request
  `selected_tokens`/`total_tokens` arrays, or any row violates `sparsity_rate == 1 - selected/total`.
- 2 unit tests updated + green. trial_evidence proven to REFUSE the exact R2 mislabel (3588.7 vs 4774) and
  PASS a correct record.
Validated end-to-end: smoke + full re-run `results_r3/` → `total_tokens_mean ≈ 4765` (true seq-len), all 6
DS trials `trial_evidence.py` PASS with the consistency gate. Verdict CONCLUSION unchanged (2048 < true
total); only the number is corrected. Decode timing unchanged (host-side, zero GPU sync).

## Mainline gap 2 — AC-8: ledgers + evidence + push
- **Full re-run (`results_r3/`, HEAD 8df44a59c)** supersedes `results_r2/` (retired via `SUPERSEDED.md`):
  both comparators rc=3, verdict reproduces (DS PASS@16/32, FAIL@64; DSA also fails@64). `c805b4be5`.
- **Raw evidence committed losslessly** (`*.jsonl.gz`/`*.log.gz` + `EVIDENCE_SHA256.txt` raw+gz hashes +
  `REPRODUCE.md`). **Preflight CLEAN**: git tree clean; all evidence tracked; hash verify lossless; BOTH
  comparator replays from the decompressed committed artifacts rc=3; all 6 DS `trial_evidence.py` PASS.
- **Ledgers regenerated to ONE current state** (`3dc0cb4ef`): `queue.md` op-point mask row = REGENERATED
  (was GONE/regen-mandatory), task2/task3 DONE, task11 ACTIVE-until-push, R3 round-history added;
  `results.md` points at `results_r3`, close-out NOT marked complete until push.
- **PUSH (DONE):** owner-authorized push to the fork `Jiminator/sglang` `dev/double-sparsity-standalone`
  completed (fast-forward `cd2d1e7c1..2ce2adf4e`); public upstream NOT used. The first push was rejected by
  GitHub's 100 MB file limit on an accidental 252 MB tqdm-spam log
  (`results_v2/crash_evidence_r1/log_ds_c64.txt`, committed in R1; key lines already quoted in
  R1_DS_CRASH_FINDING.md). Purged it from history via `git filter-branch` (commits after `8e4407822` re-SHA'd,
  content intact; a pre-rewrite backup branch `loop11b-backup-pre-filter` was kept), then the fast-forward
  push succeeded. meta.json commit_sha run-time stamps (e.g. `8df44a59c`) stay internally consistent
  (DS == DSA) so the comparator gate holds.

## The verdict (unchanged)
DS PASS@conc16 (40.70 / 1.58s) + conc32 (34.05 / 3.00), FAIL@conc64 (26.91 < 30, 25.11s ≥ 22). DSA also
fails @64. Both comparators rc=3. Competitive-to-better than DSA at both op-points; ≤6% per-step tax.

## Files changed (R3)
- `metrics.py`, `models/deepseek_v2.py`, `layers/attention/dsa_backend.py`, `bench_serving.py`,
  `runs/20260616_mb/trial_evidence.py`, `test/.../test_double_sparsity_unit.py` — the total_tokens contract.
- `runs/20260616_mb/mb_r3.sh`; `results_r3/` (corrected verdict + .gz evidence + REPRODUCE + manifest);
  `results_r2/SUPERSEDED.md` (+ R2 evidence removed from tree). `development/{results,queue}.md` regenerated.

## Validation
- 2 unit tests green; trial_evidence catches the mislabel + passes correct records; full re-run all 6 PASS;
  both comparators rc=3 replayed from committed artifacts; preflight clean.

## Queued (not blocking)
Plan-workflow terminology in PRE-EXISTING comments (`batch_result_processor.py`, `benchmark_compare.py`);
new R3 code is terminology-clean. Clean in a focused pass.

## BitLesson Delta
- Action: add
- Lesson ID(s): BL-20260617-ds-total-tokens-explicit-not-rate-inverse
- Notes: publish derived quantities (total = selected/(1-sparsity_rate)) as EXPLICIT fields, not by inverting
  a rate whose convention can be misread; and make the fail-closed validator cross-check the aggregate
  against the per-request arrays + the metric contract, not just an ordering (selected < total).

## Goal Tracker Update Request
### Requested Changes:
- Mark AC-5 RESOLVED (R3): explicit total_tokens; all 6 DS trials trial_evidence PASS with consistency gate.
- Mark AC-8 evidence + ledgers RESOLVED (R3): results_r3 committed losslessly + replay-validated; ledgers current.
- Mark AC-8 PUSH DONE: pushed to owner fork `Jiminator/sglang` `dev/double-sparsity-standalone`
  (fast-forward `cd2d1e7c1..2ce2adf4e`); a 252 MB accidental log was purged from history to satisfy GitHub's
  size limit. AC-8 (and the loop's close-out) is now complete.
### Justification:
Every AC-5/AC-8 item is done + verified, including the owner-authorized push. The verdict (DS PASS@16/32,
FAIL@64) is published with correct, consistency-gated per-trial evidence that replays from committed artifacts.

## Round-4 Codex review — CLEAN (after iterative fixes)
The owner-authorized Codex review (`--base cd2d1e7c1`, since `origin/main` 11605767e shares no ancestor with
the branch; the loop's auto-detected base + Codex's bubblewrap sandbox are blocked in this env) flagged
findings across iterations, all fixed + pushed:
- [P3] `build_corpus.py` creates its gitignored output dir before writing (`da12616a5`).
- [P2] `benchmark_compare.py`: the report verdict now exposes `client_slo_verdict` (gating, matches exit) +
  `directional_verdict`; the directional ratio is labeled REPORT-ONLY (DEC-6, non-gating).
- [P2] `trial_evidence.py`: fails CLOSED on partial/length-mismatched DS arrays; contract over row-aligned arrays.
- [P2] `test_maybe_abort_on_ds_error`: mocks `update_finish_state` (the R1 finisher rename).
- Greened the comparator unit suite: added the locked Option-B field `disable_custom_all_reduce` to the
  fixtures (pre-existing gap) + updated `test_tps/ttft_gate_fail` + `test_too_few_trials` to DEC-6 (directional
  ratios report-only) + the 2-trial floor. Full suite: 383 passed (`9ab62e6ad`).
FINAL re-review: "I did not find any verified, high-signal correctness issues introduced by the diff." Branch
pushed to `Jiminator/sglang` at HEAD `9ab62e6ad`. The loop's deliverable is complete.
