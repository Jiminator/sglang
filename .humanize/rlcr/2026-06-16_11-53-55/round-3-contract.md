# Round 3 Contract

## Mainline Objective (ONE)
Make the M-B verdict's AC-5 evidence NUMERICALLY CORRECT and finish AC-8 close-out honestly. Codex's R2
review found a real bug: the published `total_tokens_mean` is wrong because the DS publishers emit
`sparsity_rate = 1 − selected/total` (pruned fraction) while `bench_serving` inverts it as
`total = selected / sparsity_rate` (which assumes the KEPT fraction). Verified on the committed artifact:
reported `total_tokens_mean=3588.7`, correct value `selected/(1−sparsity_rate)=4770.3`. Fix the metric
contract, regenerate the full verdict at one new HEAD, and complete AC-8 (current ledgers + push/waiver).

## Target ACs
- **AC-5** (no-op refusal): `total_tokens_mean` is the true sequence-length total, consistent with the
  per-request `selected_tokens`/`sparsity_rate` arrays; all 6 DS trials `trial_evidence.py` PASS with a
  STRENGTHENED consistency check (aggregate vs per-request arrays within tolerance).
- **AC-8** (ledger/evidence/push): `queue.md` + `results.md` ONE current state (op-point facts updated,
  task2/task3 DONE, task11 ACTIVE until push done); evidence preflight; push to an owner-approved remote OR
  an explicit owner waiver recorded.

## Blocking issues (truly block the objective)
- **B1 — total_tokens metric semantics (AC-5).** Codex plan: (1) keep `sparsity_rate = 1 − selected/total`;
  (2) add explicit `total_tokens` to `DoubleSparsityRequestStats` + `meta_info_for_request`, set from
  `seq_len` (GLM `dsa_backend` helper) and `sl_cpu[b]` (DeepSeek publisher); (3) `bench_serving` aggregates
  `total_tokens` DIRECTLY, with a backward-compat fallback `selected/(1−sparsity_rate)` only when
  `total_tokens` absent and `0 ≤ sparsity_rate < 1`; (4) `trial_evidence.py` REFUSES when the summary
  aggregate disagrees with the per-request arrays beyond a small tolerance; (5) rerun the FULL DS+DSA verdict
  at one new HEAD (raw JSONL summary must carry the corrected aggregate — not a sidecar edit); (6) re-run both
  comparators + all 6 trial_evidence from committed artifacts.
- **B2 — AC-8 ledger + push.** `queue.md` is stale (op-point facts table still says mask GONE/regen-mandatory;
  close-out says "to R1"; task2/task3 pending; task11 DONE while push unmet). Rewrite to one current state;
  do NOT mark close-out/task11 DONE until push is satisfied. Resolve push exactly one way: owner-approved
  remote/branch, or an explicit OWNER waiver in the ledger (the agent cannot waive on the owner's behalf or
  push experimental artifacts to the public upstream — so this needs an owner decision; ask the owner).

## Queued (out of scope this round; documented, not blocking)
- Plan-workflow terminology in PRE-EXISTING implementation comments (`batch_result_processor.py:184/329/745`,
  AC/DEC refs in `benchmark_compare.py`). My new R2/R3 code is kept terminology-clean. Clean in a focused pass.

## Success criteria
1. Metric contract: `total_tokens` explicit on `DoubleSparsityRequestStats`/`meta_info_for_request`; both
   publishers (GLM helper + DeepSeek) set it from the host seq_len; bench_serving aggregates it directly
   (+ corrected fallback); trial_evidence consistency check added. py_compile clean.
2. Smoke (GLM DS, GRAPH): `total_tokens_mean` ≈ true seq-len mean (~4770, not ~3590); consistency check PASS.
3. Full re-run at one new HEAD → all 6 DS trials `trial_evidence.py` PASS with correct total; both comparators
   rc=3 from committed artifacts; verdict reproduces (DS PASS@16/32, FAIL@64).
4. `queue.md` + `results.md` ONE current state; EVIDENCE_SHA256 + REPRODUCE updated; evidence preflight clean.
5. Push: owner-approved remote pushed, OR explicit owner waiver recorded (asked via AskUserQuestion).

## Notes
- The fix touches shared production code (metrics.py, deepseek_v2.py, dsa_backend.py, bench_serving.py,
  trial_evidence.py). Keep additive + DS-gated; do not regress DeepSeek (adding total_tokens is additive) or
  native-DSA/non-DS paths. The verdict CONCLUSION is unchanged (2048 < true total); only the NUMBER is corrected.
- Re-run is the full ~4h (commit_sha gate needs DS+DSA at the new HEAD). Honest verdict unchanged.
