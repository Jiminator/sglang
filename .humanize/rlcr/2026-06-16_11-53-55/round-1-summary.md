# Round 1 Summary

## Objective
Make the M-B M4 verdict PUBLISHABLE and close out honestly. Round 0 delivered M-A + a directional verdict,
but Codex rejected the completion: the `--ac11` comparator REFUSED, same-memory was deferred, conc-64 was
admission-capped, the benchmark emitted no reuse/no-op evidence, AC-4 was not the spec'd tax guard, and AC-8
ledger/evidence/push were incomplete. All five gaps are now closed.

## How each Codex Round-0 gap was resolved
1. **AC-9 comparator refused + same-memory deferred →** DS + DSA re-run from ONE frozen HEAD (commit_sha
   99ac584ac). BOTH op-points now `--ac11` comparator-ACCEPTED: `ac11_production_envelope` (DS0.8/DSA0.85)
   rc=3 and `ac11_same_memory` (DS0.8/DSA0.8) rc=3 — honest absolute DS FAIL@64, directional PASS. (rc=3 =
   absolute SLO fail, the published verdict; rc=2 would be refusal — neither refused.)
2. **AC-2/3 admission-capped conc-64 →** the clean re-run reached nominal: running-req **peak 63** (≥61).
   Time-averaged achieved 58.9 is a real DS smaller-KV-pool effect at mem 0.8 (queue-bound TTFT), not a
   measurement failure — stated as such.
3. **AC-4 not the spec'd guard →** replaced sweep-derived TPOT with a DEDICATED controlled fixed-bs probe
   (distinct-prefix, GRAPH, mem 0.8 both sides): bs64 DS 39.83ms/DSA 37.70ms = 1.056; bs30 31.85/30.14 =
   1.057 — both ≤1.10 PASS; bs30 31 850µs ≪ 380 000. Loop-10 per-step parity held.
4. **AC-5/9 missing reuse/no-op evidence →** B1 extended `bench_serving` to emit per-request `cached_tokens`
   + DS no-op counters and added fail-closed `trial_evidence.py`. Every trial shows ~54% measured prefix
   reuse. No-op proven via direct evidence (0 dense_fallback across all 6 trials; top_k 2048 < 4096 ctx;
   4303 DS decode batches). The per-request DS meta_info aggregate is unwired for GLM (`Glm4MoeAttention`/
   dsa-backend never reaches DeepseekV2's `_publish_ds_request_summary`) — documented gap + recommended
   backend-side fix (`ac5_no_op_evidence.md`).
5. **AC-8 ledger/evidence/push →** results.md + queue.md regenerated to the R1 state; the stale `a4be98c4`
   capacity claim fixed to ld32 `35155ac4` (504640 reconfirmed); evidence committed as lossless `.meta.json`
   + `.evidence.json` sidecars + comparator md/json + `EVIDENCE_SHA256.txt` content hashes (raw 248MB jsonls
   gitignored); push recorded as pending owner direction (origin = public sgl-project upstream).

## The verdict (publishable)
Table-free DS on GLM-5.1-FP8 MEETS the client SLO (decode-TPS p50 ≥30, P99 TTFT <22s) at conc 16 (40.73 /
1.59s) and 32 (34.13 / 2.99s), and FAILS at 64 (26.98 <30, 25.08s ≥22). Native DSA ALSO fails @64. DS is
competitive-to-better than DSA at both op-points (TPS ratio 0.98–1.03, TTFT ratio 0.44–0.76). A throughput
FAIL is the honest, complete deliverable per the plan. Headline: `runs/20260616_mb/R1_HEADLINE_VERDICT.md`.

## Significant finding (real bug found + fixed)
The "clean re-run" surfaced a genuine DS server crash: the per-request error-abort path called
`req.check_finished()`, renamed upstream (#25725) to `update_finish_state()` — a semantic merge conflict that
crashed the whole TP=8 server when the DS selector sanitized a row. Fixed (99ac584ac). It escaped detection
because the abort path only runs on a selector error, which representative load never triggered; the
100%-identical-prefix tax burst (unrepresentative) was the first thing to trip it. Full analysis +
why-it-escaped in `runs/20260616_mb/R1_DS_CRASH_FINDING.md`.

## Files changed (R1)
- `python/sglang/srt/managers/scheduler_components/batch_result_processor.py` — check_finished→update_finish_state crash-fix.
- `python/sglang/bench_serving.py` + `runs/20260616_mb/trial_evidence.py` — B1 reuse + no-op evidence (committed 9af9d7835).
- `runs/20260616_mb/{ds_only,dsa_tax,mb_v2}.sh` — corrected sweep-first + distinct-prefix runners.
- `runs/20260616_mb/{R1_HEADLINE_VERDICT,R1_DS_CRASH_FINDING,ac5_no_op_evidence}.md`, `results_v2/ac11_*.{md,json}`, EVIDENCE_SHA256.txt.
- `development/{results,queue}.md`, `runs/20260616_ma/capacity_ds_evidence.md` (a4be98c4→35155ac4).

## Validation
- Both comparators accepted (rc=3) at matched commit_sha; verdict reproduces Round 0.
- DS verdict sweep: 0 selector errors, 0 crashes, peak 63, ~54% reuse/trial.
- AC-4 dedicated probe both bs PASS. AC-5 no-op proven by direct evidence.

## Remaining Items
- Push: pending explicit owner direction (origin = public upstream). Recorded, not skipped.
- Follow-up (out of scope): wire per-request DS summary backend-side so GLM populates meta_info.

## BitLesson Delta
- Action: add
- Lesson ID(s): BL-20260617-ds-abort-path-merge-drift
- Notes: A feature-branch error-handling path that calls a method renamed by an upstream merge is a semantic
  merge conflict invisible to git and to happy-path tests — it only fires on the error path. When a rename
  lands via merge, grep the feature branch's NEW call sites for the old name. Also: a 100%-identical-prefix
  load burst is NOT representative of production reuse and can trip reuse-path edges the real workload never hits.
