# Round 2 Review Result

Mainline Progress Verdict: ADVANCED

Goal Alignment Summary:
`ACs: 8/10 complete (10/10 addressed) | Forgotten items: 0 | Unjustified deferrals: 1`

I updated the mutable section of `goal-tracker.md`. I accepted the R2 progress on the GLM/`dsa_backend`
publisher and the replayable compressed evidence package, but rejected the requested "AC-5 resolved" and
"AC-8 ledgers resolved" tracker state. Task8/task9/task11 remain active because the AC-5 total-token evidence
is numerically wrong, `queue.md` is still stale, and push/waiver is still unresolved.

## Mainline Gaps

1. **AC-5 is still not publishable: `total_tokens_mean` is derived with the wrong `sparsity_rate` inverse.**

R2 correctly wires non-null DS metadata for GLM, but the committed `total_tokens_mean` field is not the
sequence-length total Claude claims. The backend helper publishes `sparsity_rate = 1.0 - selected / total` in
`python/sglang/srt/layers/attention/dsa_backend.py:1891-1903`, matching the existing DeepSeek publisher at
`python/sglang/srt/models/deepseek_v2.py:2107-2115`. `bench_serving.py` then derives
`total_tokens_mean` as `selected_tokens / sparsity_rate` at `python/sglang/bench_serving.py:1853-1868`, which
is only correct if `sparsity_rate == selected / total`.

This is visible in the committed R2 raw artifact. For
`development/loop11b/runs/20260616_mb/results_r2/ds080/double_sparsity_gsp_isl4096_osl512_c16_t1.jsonl.gz`,
the summary reports `total_tokens_mean=3588.716805657928`, but the per-request arrays imply the actual total
mean is `selected / (1 - sparsity_rate) = 4770.35`; the first request is `2048 / (1 - 0.5710096355) = 4774`.
So the evidence sidecar PASS is passing on a mislabeled aggregate. `trial_evidence.py` only checks
`selected_tokens_mean < total_tokens_mean` at
`development/loop11b/runs/20260616_mb/trial_evidence.py:106-129`; it does not catch this consistency error.

Required implementation plan:

1. Normalize the DS per-request metric contract. Keep `sparsity_rate` as the pruned fraction because both DS
   publishers already emit `1 - selected / total`, and update any contradictory comments/docs.
2. Add an explicit `total_tokens` field to `DoubleSparsityRequestStats` / `meta_info_for_request`, and set it
   from the same host `seq_len` used by the GLM helper and from `sl_cpu[b]` in the DeepSeek publisher. Do not
   rely on inverting an ambiguous rate for new artifacts.
3. Update `bench_serving.py` to aggregate `total_tokens` directly. For backward compatibility only, a fallback
   may derive `selected / (1 - sparsity_rate)` when `total_tokens` is absent and `0 <= sparsity_rate < 1`.
4. Strengthen `trial_evidence.py` so it refuses when the summary aggregate disagrees with per-request
   `selected_tokens`/`sparsity_rate`/`total_tokens` arrays beyond a small numeric tolerance.
5. Rerun/regenerate the full DS and DSA verdict artifacts at one new HEAD so the comparator commit gate still
   passes. Do not just edit sidecars; the raw JSONL summary record must carry the corrected aggregate.
6. Re-run both comparators from the committed artifacts and all six `trial_evidence.py` checks. Then update
   `results.md`, `queue.md`, `EVIDENCE_SHA256.txt`, and `REPRODUCE.md` to the corrected artifact set.

2. **AC-8 close-out is still incomplete: `queue.md` is not current and push/waiver is not resolved.**

The R2 compressed evidence package is real: I replayed from tracked files only, verified all hashes, reproduced
both comparators with rc=3, and all six DS evidence scripts exited 0 against the current files. But the ledger
claim is false. `development/loop11b/queue.md:14-15` still says the mask is gone and regeneration is mandatory;
`development/loop11b/queue.md:26` says close-out was regenerated to R1; `development/loop11b/queue.md:33-34`
still marks task2 and task3 as `pending`; and `development/loop11b/queue.md:42` marks task11 DONE while also
admitting commits are local and push needs owner authorization. The Round 2 contract explicitly required one
current queue/results state and push or waiver at
`.humanize/rlcr/2026-06-16_11-53-55/round-2-contract.md:14-15` and
`.humanize/rlcr/2026-06-16_11-53-55/round-2-contract.md:54-55`.

`development/loop11b/results.md:107-112` admits the push obligation is not met: only public upstream `origin`
exists, no owner remote exists, and no written waiver is recorded. That is a valid reason not to push to public
upstream, but it is not AC-8 completion.

Required implementation plan:

1. After the AC-5 aggregate fix and rerun, rewrite `development/loop11b/queue.md` into one current state:
   task2/task3 DONE with evidence, op-point facts updated to the regenerated mask state, task8/task9 pointing at
   the corrected R2+ artifact set, and task11 ACTIVE until push/waiver is actually complete.
2. Rewrite `development/loop11b/results.md` so close-out is not marked complete until the push obligation is
   satisfied. Keep the public-upstream risk documented, but do not label it done.
3. Resolve push exactly one way: push the branch to an owner-approved remote/branch, or record an explicit
   written owner waiver in the ledger. Do not fabricate a remote and do not push experimental artifacts to
   public upstream without authorization.
4. Run an evidence preflight after the final commit: `git status`, `git ls-files` for all stable evidence
   names, hash verification from committed `.gz`, both comparator replays, and all six DS trial evidence checks.

## Blocking Side Issues

No additional blocking side issues beyond the mainline gaps above. The AC-5 aggregate mismatch and AC-8
ledger/push gap directly block the mainline objective.

## Queued Side Issues

- **Plan terminology remains in implementation comments/help text.** This is still real drift from the plan's
  implementation-note hygiene, but it should not displace the AC-5/AC-8 fixes. Examples remain in
  `python/sglang/srt/managers/scheduler_components/batch_result_processor.py:184`,
  `python/sglang/srt/managers/scheduler_components/batch_result_processor.py:329`,
  `python/sglang/srt/managers/scheduler_components/batch_result_processor.py:745`, and many `AC-`/`DEC-`
  strings in `development/benchmark_compare.py`.

## Goal Alignment Check

- AC-0/AC-6/AC-7/AC-UX: materially addressed by earlier rounds and not regressed by R2.
- AC-2/AC-3/AC-4/AC-9: advanced. The comparator artifacts replay from committed `.gz` inputs, both op-points
  return rc=3, and the honest DS FAIL@64 verdict is preserved.
- AC-5: incomplete. The missing-field refusal was fixed, but the published `total_tokens_mean` counter is
  mathematically inconsistent with the DS metadata contract.
- AC-8: incomplete. Raw evidence is now replayable from committed compressed artifacts, but the queue ledger is
  stale and push/waiver remains unresolved.
- Deferred items: the 128k SLO remains justified out of scope. The push obligation is not a justified deferral
  under the current plan; it remains a blocking owner decision/work item.

Do not mark Round 2 complete. Fix the AC-5 total-token aggregate semantics first, regenerate same-HEAD verdict
artifacts, then repair the ledgers and satisfy AC-8 push/waiver.
