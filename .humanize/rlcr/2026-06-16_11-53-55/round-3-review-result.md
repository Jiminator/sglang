# Round 3 Review Result

Mainline Progress Verdict: ADVANCED

Goal Alignment Summary:
`ACs: 10/10 addressed | Forgotten items: 0 | Unjustified deferrals: 0`

I updated the mutable section of `goal-tracker.md`: task8/task9/task11 were moved out of Active into
Completed and Verified, the plan version is now Round 3 review, and the pushed fork ref was corrected to the
actual verified remote HEAD `e0935e5a9` instead of the stale pre-closeout range text.

## Mainline Gaps

None.

The R2 AC-5 blocker is fixed in the verdict path. `DoubleSparsityRequestStats` now carries explicit
`total_tokens`, both GLM `dsa_backend` and DeepSeekV2 publish it from host sequence length, `bench_serving`
aggregates it directly, and `trial_evidence.py` refuses aggregate/per-request inconsistencies plus
`sparsity_rate != 1 - selected/total`.

I replayed `results_r3/` from a `git archive` of HEAD so ignored local raw files could not influence the
check. The archive contained 0 raw JSONLs/logs before decompression; after decompression, all 26 raw hashes
verified, all 6 DS `trial_evidence.py` runs passed, and both comparator replays returned rc=3:
production-envelope and same-memory both reproduce `client_slo_verdict=FAIL`, DS conc64 decode-TPS
`26.911753838576853`, P99 TTFT `25.109682296346875`, and corrected `total_tokens_mean=4765.478125`.

The R2 AC-8 blocker is fixed. `results.md`/`queue.md` point to `results_r3`, raw evidence is committed
losslessly as `.gz` plus hashes and reproduction commands, `git status --short` is clean, and
`git ls-remote jiminator dev/double-sparsity-standalone` returns `e0935e5a9`, matching local HEAD.

Focused validation:
- `python3 -m pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -k 'meta_info_shape or customized_info or record_selection'` → 3 passed.
- Archive replay from committed evidence only → hash check OK, 6/6 trial evidence PASS, both comparators rc=3.

## Blocking Side Issues

None.

## Queued Side Issues

- Some metric/comment wording is still stale around `sparsity_rate` semantics. `python/sglang/bench_serving.py`
  still has an old comment saying `total_tokens` is derived as `selected_tokens / sparsity_rate`, and
  `python/sglang/srt/layers/attention/double_sparsity/metrics.py` still describes/records the Prometheus
  histogram as selected/total while the per-request JSON contract is now the pruned fraction. This does not
  block AC-5/AC-8 because the published JSONL path uses explicit `total_tokens` and the replayed validator
  checks the pruned-fraction contract. Clean it in the queued terminology/observability pass.

## Goal Alignment Check

- AC-0/AC-6/AC-7/AC-UX: completed in earlier rounds; R3 did not regress them.
- AC-2/AC-3/AC-4/AC-9: completed and replayable from committed R3 artifacts; the honest DS FAIL@64 verdict is preserved.
- AC-5: resolved by explicit `total_tokens` plus consistency-gated per-trial evidence.
- AC-8: resolved by current ledgers, lossless committed evidence, reproduction validation, clean worktree, and verified push to the owner fork.
- Deferred items: only the 128k ISL / 1024 OSL workload remains out of scope per the immutable plan; no unjustified deferrals remain.

COMPLETE
