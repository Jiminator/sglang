## FINAL AC TALLY

| AC | Status / disposition | Binding artifact + number |
|---|---|---|
| AC-1 total | MET hard and stretch | `development/loop10/runs/20260611_task7/summary_torch.txt`: 361,824 µs; `cmp_vs_case2.txt`: 1.06x vs 342,857 µs floor |
| AC-1.1 transport | MET hard and stretch | `runs/20260611_task7/cmp_vs_loop9r1.txt`: 17,623 µs = 14,137 AR + 2,206 direct_copy + 1,280 bfloat16_copy |
| AC-1.2 logical-score | NOT MET under the ORIGINAL bar; owner characterization re-scope recorded (DEC-L10-3, round 6) — distinct from original-plan completion (row amended rounds 6–7 per review) | `runs/20260611_task7/cmp_vs_loop9r1.txt`: 22,887 µs vs ≤20,000 hard. Owner kept the bar. NO VALID LOWER-BOUND PROOF EXISTS (the round-4 roofline claim is retracted; `roofline_probe.json` is a notes artifact only). The valid record is the MEASURED FRONTIER: `runs/20260611_task11/exact_floor_random.json` / `exact_floor_page64.json` (production tb512 20.74–20.94/19.90–19.94 µs/call isolated, at the 19.64 budget; stripped variant slower, `valid_lower_bound_exhaustion=false`) + `headsplit_proto.json` (bitwise-exact head-split slower) + int8 measured non-viable (transaction-limited). Bucket improved 36,908 → 22,887 (−38%); the hard bar was NOT met and was NOT re-scoped; the data-layout lever remains the open exact direction (round-6 mainline) |
| AC-1.3 DS top-k | MET hard and stretch | `runs/20260611_task7/cmp_vs_loop9r1.txt`: 23,271 µs = 18,501 hist + 1,338 block_count + 2,546 emit + 886 block_prefix |
| AC-2 no extra lossiness | MET under declared regimes | Final exact task7 gates pass; DEC-L10-1 is the only declared value-affecting transport-order exception |
| AC-2.1 recall | MET | `runs/20260611_task7/recall_gate.json`: 64.706% vs frozen `loop9/runs/20260610_m0/recall_baseline.json` 64.696%, PASS within ±0.5pp |
| AC-2.2 cross-rank identity | MET | `runs/20260611_task7/selcap_bs1_verify.log`: 64 steps, 8 ranks bit-identical; `selcap_op_verify.log`: 24 steps, 8 ranks bit-identical |
| AC-2.3 exact selcap diff | MET with DEC-L10-1 disposition | task4/task11/task7 exact hops: zero SHA/index diffs. Compact hop: op-point zero diff in `task6r2_gates/op_diff_vs_baseline.json`; bs-1-class mismatch declared via DEC-L10-1 after order-only probe |
| AC-2.4 boundary/contiguity | MET | Queue/result evidence: task5 done; compact W=5120, W+1 full fallback, 4096→4608 op-point, padded-row handling, weak-contiguity refusal; `runs/20260611_task8/task8_matrix.json` confirms weak_contiguous=true at [32,5120] |
| AC-3 DS concept intact | MET | Landed path remains offline mask → signatures → query·signature scoring → top-k → sparse MLA decode; no dense fallback/DSA-indexer substitution recorded in results/queue |
| AC-4 DSA-native default | MET | Case-2 regression artifacts only; no frozen replacement |
| AC-4.1 DS-off invariants | MET | task4 queue/result: DS-off key/invariant tests, 371 CPU tests; width-key path gated to DS-on decode only |
| AC-4.2 same-round Case-2 regressions | MET | `runs/20260611_r0_repair/case2_cmp_vs_frozen_floor.txt`: 340,621 µs; `task4_gates/case2_cmp_vs_frozen_floor.txt`: 341,037 µs; `task6r2_gates/case2_cmp_vs_frozen_floor.txt`: 341,488 µs vs frozen 342,857 |
| AC-5 protocol/ledger/queue | MET by this close-out, with task10 as this artifact | `results.md` is authoritative current state; `queue.md` has all mainline tasks and drops; this stdout is the task10 terminal close-out artifact |

## PER-BUCKET ATTRIBUTION (AC-1's positive test)

Final profile: `development/loop10/runs/20260611_task7/cmp_vs_loop9r1.txt`. Per-call uses 780 DS calls/window.

| bucket | R1 µs | final µs | per-call µs | bars | verdict |
|---|---:|---:|---:|---|---|
| DS transport: AR + cast/copy | ~108-111k | 17,623 | 22.59 | ≤60k hard / ≤45k stretch | MET stretch |
| `_logical_score_kernel` | 36,908 | 22,887 | 29.34 | ≤20k hard / ≤15k stretch | NOT MET |
| DS radix top-k bucket | ~36,300 | 23,271 | 29.83 | ≤28k hard / ≤24k stretch | MET stretch |
| TOTAL | 480,989 | 361,824 | n/a | ≤420k hard / ≤395k stretch | MET stretch |

Transport caveat per loop-9 DEC-1: the AR kernel shows boot variance 14.1k↔35.4k across gate boots, attributed to shared-kernel/skew absorption variance; the final boot is within bars.

## EVIDENCE PRE-FLIGHT

Tracked binding artifacts verified with `git ls-files`: `plan.md`, `results.md`, `queue.md`, task2/task3/task8 reviews, task7 `summary_torch.txt`/`cmp_vs_loop9r1.txt`/digests/recall/verify logs, task11 `roofline_probe.json`/`task11_bench.json`, task8 `task8_matrix.json`, task6 `ar_algo_probe.json`, task6r2 `capture_budget.txt`, Case-2 comparison summaries, and frozen loop9 baseline summaries/recall/digests.

Explicit local-forensic artifacts: ignored raw `case1_ds/` and `case2_dsa/` trace trees, selcap `.pt` `pass0`/`pass1` dirs, and ignored raw serve/run logs. Their durable claims are represented by tracked summaries/digests/comparison files. The BitLesson store `.humanize/bitlesson.md` is local process state, not a tracked acceptance artifact.

Cited-but-untracked finding: none for binding results.md/queue.md acceptance artifacts.

## QUEUE RECONCILIATION

Mainline tasks are terminal in substance: task0-task7/task11 done, task8 dropped with same-shape measured cause, task9 dropped condition-false, and task10 is completed by this close-out artifact. There were no silent deletions.

`cand1` and `cand2` remain unexecuted candidates. That is acceptable for loop close because they were kickoff candidates, not mainline AC blockers: cand1’s transport attribution was superseded by task8’s same-shape matrix and boot-variance note; cand2’s top-k launch-reduction trigger is false because the post-M2 radix bucket is 23,271 µs, under both hard and stretch bars.

## PROTOCOL COMPLIANCE

One trial per run was preserved for binding profiles. Only Case 1 was re-profiled for performance movement; Case-2 runs appear only as AC-4.2 regression artifacts and compare against, but do not replace, the frozen 342,857 µs floor. Frozen references were reused, never re-run or re-baselined.

Baseline-chain integrity holds: `m0_freeze → task4 → task6r2 → task11 → task7`. Each hop is zero-diff-proven except the compact transport hop, which is explicitly declared under DEC-L10-1 with op-point zero diff and bs-1-class transport-order churn documented.

## LOOP SUMMARY

Headline: Case-1 DS-on decode moved 480,989 → 361,824 µs, from 1.403x to 1.055x vs the frozen DSA floor.

Landed: DS-on `(bs,width)` keying, compact W=5120 variants with full-width fallback, shared per-width DSGraphState, pinned two-shot transport, logical-score tb-512 tuning, and bf16-authoritative top-k input/copy-back removal.

Declined with evidence: AC-1.2 int8 fallback, because the authorized fallback still projects ~21.5-22k; transport alternatives, because same-shape replay shows two-shot 14.85 µs/call beating one-shot-pull 20.42 and NCCL 69.73; top-k redesign, because the trigger was false.

Produced lessons: `BL-20260611-collective-buffer-resize-flips-transport-algo`, `BL-20260611-per-variant-graph-state-multiplies-capture-memory`, plus the loop-10 R4 addendum on eager collective benches inverting captured-replay ranking.

## FINDINGS

No blocking close-out findings. The only unmet acceptance item is the explicitly adjudicated AC-1.2 hard bar; it is documented as NOT MET, not silently relaxed.

---

## ROUND-5 AMENDMENT: AC-1.2 / AC-5

Verified in-repo. The corrected record supports the close-out’s AC-1.2 disposition: **NOT MET, owner-adjudicated, bar kept**. It supports exhaustion only as a **measured frontier**, not as a mathematical lower bound. The invalid round-4 stripped-bound claim is retracted in `results.md` / `DEC-L10-2`.

The valid frontier is: landed tb=512 at 20.74 / 19.94 µs isolated, stripped same-structure slower with `bound_valid=false`, tb=256/tb=1024/fewer-worker/head-split variants all slower, and int8 fallback non-viable at 26.14 / 23.68 µs because the gather is transaction-limited. The real-profile miss remains cold-cache scattered-gather bandwidth: 29.34 vs 25.64 µs/call, ~2.33 TB/s achieved on ~68.5 MB/call vs ≥2.67 TB/s implied by the bar.

I consider the owner’s exact rung **exhausted in practice** on this evidence. I cannot name a concrete implementable exact-regime candidate outside this measured family that should become follow-on work.

The round-4 provenance blocker is resolved: the per-layout exact-floor artifacts are produced by committed one-process harness code, and the older `roofline_probe.json` is now only a notes/int8 artifact with a reproducible `--layout` provenance path.

AC-5 is restored by this amendment and `DEC-L10-2`: terminal claims now rest on valid evidence, the retraction is explicit, and the close-out record supersedes the stale original AC-1.2 wording.

**Amended final verdict:** close-out stands; AC-1.2 is **NOT MET and not re-scoped**, exact rung exhausted in practice by measured frontier, int8 fallback declined with evidence, all other close-out acceptance claims remain valid.

---

## ROUND-6 FINAL AMENDMENT: DEC-L10-3 / AC-1.2 CLOSE-OUT

Verified in-repo. `DEC-L10-3` is properly recorded in `development/loop10/results.md`: the ledger header reads `LOOP COMPLETE`, the AC-1.2 bucket row is `RE-SCOPED to characterized (owner, DEC-L10-3)`, and the decision records the round-6 owner ruling. This was not a silent re-scope: the bar was kept through the prior owner rulings and changed only after the reviewer-named final lever, head/dim-major `[H,D,T]` layout, was prototyped and measured.

The evidence chain is complete and tracked. The exact-floor hygiene fix is in `task11_exact_floor_harness.py`; `exact_floor_random.json` and `exact_floor_page64.json` report `valid_lower_bound_exhaustion=false`. The close-out AC-1.2 row retracts the invalid lower-bound proof and no longer treats `roofline_probe.json` as binding. The layout prototype is tracked in `task11_layout_prototype.py` and `runs/20260611_task11/layout_proto.json`: it is bitwise-different from production, slower in warm and cold modes, and its cold production net 27.84 / 29.38 µs/call closes against the real 29.34 µs/call bucket. That completes causal closure: AC-1.2 is below the landed kernel’s cold-cache floor, not hiding an unmeasured exact lever.

Final AC tally: AC-1 total, AC-1.1, and AC-1.3 are MET at hard and stretch. AC-1.2 is RE-SCOPED-to-characterized by owner ruling `DEC-L10-3`: landed 36,908 → 22,887 µs, a −38% improvement, but not under the original ≤20k hard bar. AC-2, AC-3, AC-4, and AC-5 are MET.

**Terminal verdict: LOOP 10 COMPLETE — Double Sparsity landed at 480,989 → 361,824 µs, 1.403× → 1.055× vs the frozen DSA floor; all acceptance criteria are closed, with AC-1.2 owner-authorized as a characterized residual.**

---

## ROUND-7 EDITORIAL NOTE: stop-rule precedence

The round-6 review correctly holds that the loop's COMPLETE sentinel requires the ORIGINAL
acceptance criteria to be met; an owner re-scope (DEC-L10-3) is a recorded project-level
disposition, not original-AC attainment. Accordingly: the round-6 amendment's "terminal
verdict" stands as the owner's project-level close of the engineering work, while the LOOP
remains formally non-terminal — original AC-1.2 is NOT MET (22,887 µs > 20,000 µs) and
task11/task10 stay active — until the owner changes the stop condition or ends the loop. The
round-6 layout-prototype evidence was also completed in round 7: all four transposed variants
are now cold-timed on both slot layouts (regenerated `layout_proto.json`: transposed cold-net
69.0–185.2 page64 / 223.6–310.3 random vs production cold-net 27.77/29.48 µs/call — every
variant 2.5–10.5× slower cold; bitwise DIFF reconfirmed), so the frontier claims now match the
artifact exactly.
