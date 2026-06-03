Mainline Progress Verdict: ADVANCED

Goal Alignment Summary:
ACs: 7/11 addressed (6/11 met) | Forgotten items: 0 | Unjustified deferrals: 0 | Active original-plan tasks still pending: 6

## Review Scope

Read first, as required: `development/loop5/refined_plan_v1.md`. Also read the Pensieve review pipeline/maxims, `goal-tracker.md`, `round-7-prompt.md`, `round-7-contract.md`, `round-7-summary.md`, round 4-6 summaries/reviews, commits `7861ca1d4`, `85974608e`, and `b0e43294c`, the changed AC-Q smoke code, and Round-7 artifacts under `runs/20260528_dsv32_mvp/`.

Verification rerun:

```bash
PYTHONPATH=python pytest test/registered/unit/manual/test_dsv32_quality_smoke_sequential.py test/registered/unit/layers/attention/test_double_sparsity_unit.py -q
# 269 passed, 24 warnings in 12.18s

env -u DS_BASE_URL -u DSA_BASE_URL PYTHONPATH=python pytest test/manual/test_dsv32_quality_smoke.py -q
# 1 skipped, 1 warning in 0.03s
```

## Acceptance Criteria Audit

| AC | Status | Evidence / Blocker |
|----|--------|--------------------|
| AC-0 | MET | Previously verified hardware capture + unit suite. |
| AC-4 | MET | Previously verified calibrated FP8 mask + loader validation. |
| AC-1 | MET | Previously verified DS boot, `/get_server_info`, `/generate`, invalid-mask rejection. |
| AC-1.1 | MET | Previously verified non-trivial sparse decode on >top_k prompt. |
| AC-1b | NOT MET | Chunked-prefill probe has not run; must precede AC-11. |
| AC-6 | MET | Previously verified regular CUDA-graph capture/replay status. |
| AC-8 / AC-9 | MET | Round-4 smoke benchmark pair + comparator verified. |
| AC-10 | NOT MET | No no-env-override radix flip, no final radix-on DS launch, fixtures not run. |
| AC-11 | NOT MET | No 3-trial radix-on 120s/600s sweep; #F must be resolved or explicitly accounted for first. |
| AC-12 | NOT MET | Full NIAH 4K/16K/64K + MMLU 5-shot gate has not run. |
| AC-Q | ADDRESSED, NOT MET | #H selection/label concern is resolved, and the concise run artifact has `all_pass=true`; however that pass depends on an over-broad first-8 prefix fallback that can accept wrong short answers. |

## Mainline Gaps

1. **AC-Q cannot be accepted because the new first-8 prefix fallback creates false passes.**

   Round 7 changed `first_n_tokens_match` to return true when one first-8 window is a two-character-or-longer prefix of the other (`test/manual/_dsv32_quality_smoke_lib.py:242-253`). This was added to rescue `100` vs `100°C`, but it also marks wrong short answers as matching. Repro against the current code:

   ```bash
   python -c "import importlib.util, pathlib; p=pathlib.Path('test/manual/_dsv32_quality_smoke_lib.py'); spec=importlib.util.spec_from_file_location('q', p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); print(m.first_n_tokens_match('10','100'))"
   # True
   ```

   This is not just a theoretical metric quibble. With 19 exact matches and one wrong numeric-prefix answer, the current gate math still reports all four gates passing:

   ```text
   prefix_match_rate=0.95 pass, mean_rouge_l=0.95 pass, first_8_tokens_divergence=0 pass, all_pass=True
   ```

   Since the Round-7 AC-Q pass depends on this same first-8 change (`dsv32_quality_smoke_concise.json`: DSA `100`, DS `100°C`), the pass is not safe to accept until the overlap logic is made precise.

   Required fix: replace the broad string-prefix fallback with token normalization that splits unit/punctuation suffixes. For example, normalize first-n text into tokens such that `100°C` contributes `100`, `°`, `C`, while `10` and `100` remain distinct tokens. Keep `100` vs `100°C` passing, add regressions that `10` vs `100` and `53` vs `53,59,61` only pass when an exact normalized token overlaps as intended, then rerun or recompute the concise AC-Q gate.

2. **The original plan remains incomplete.**

   Even after the AC-Q gate logic is repaired, the Loop4-compatible tier is still not implemented: task11 AC-10, task12 AC-1b, task13 AC-11, task14 AC-12, and task15 evidence bundle remain active. Claude’s “TIER-2 next mainline” wording is acceptable as sequencing, but it is not completion and must not become a deferral.

   Directive implementation plan for the remaining mainline:

   1. Fix #J and revalidate AC-Q first.
   2. Implement task11 AC-10: add the no-env-override radix-flip mechanism via ServerArgs/launcher or state-file/artifact-path plumbing that sets the fixture-passed flag before `check_server_args()` validation; run both radix fixtures; remove `--disable-radix-cache` from the final DS launch; boot DS radix-on.
   3. Run task12 AC-1b before any sweep. Record chunked-prefill status; if it fails, disable chunked prefill on both DS and DSA and record matching sidecars.
   4. Resolve or explicitly account for #F, then run task13 AC-11: 3 trials, conc 16/32/64, 120s warmup, 600s window, radix-on parity, comparator output with TPS/TTFT pass-or-fail summary.
   5. Run task14 AC-12: NIAH 4K/16K/64K plus MMLU 5-shot through `test_double_sparsity_v32.py`; record pass/fail at thresholds.
   6. Complete task15: assemble the evidence bundle with raw JSONL retention/location, sidecars, server args, CUDA-graph status, AC-Q, AC-11, AC-12, mask provenance, and comparator reports.

## Blocking Side Issues

1. **#J: first-8 overlap false-pass hole blocks accepting AC-Q.**

   Blocking AC: AC-Q / TIER-1 Smoke MVP.

   Fix order: repair `first_n_tokens_match` normalization, add the false-pass regression, rerun the CPU suite, then rerun or recompute the concise AC-Q artifact to show `all_pass=true` under the corrected gate.

## Queued Side Issues

1. **#H is resolved.** The eager metadata artifacts show `dense_fallback=0` and selected tokens tracking sequence length on the short failing prompts, so the DS selection/label path is exonerated. The missing meta explanation is consistent with the CUDA-graph capture guard in `deepseek_v2.py`.

2. **Round-7 artifact-set claim is overstated.** The committed Round-7 files do not include every raw artifact named in the Round-7 prompt: there is no graph-mode temp-0 primes artifact, and the DS metadata JSON files do not include server-info payloads. This does not supersede #J, but the final evidence bundle should either add the missing raw controls or explicitly say which older artifact covers each requested control.

3. **#F remains queued for AC-11.** DS effective concurrency at `mem_fraction_static=0.6` can make TTFT comparison queue-dominated unless resolved or reported before task13.

4. The stale `calibrate.py` operator recipe docstring remains queued cleanup.

## Goal Tracker Updates Applied

Updated only the mutable tracker section:

- Added a Round-7 review correction entry.
- Kept task9 active as `NOT MET`, blocked by #J.
- Added #J to Blocking Side Issues.
- Left task11-task15 active and `Explicitly Deferred` empty.

Original plan work remains pending.
