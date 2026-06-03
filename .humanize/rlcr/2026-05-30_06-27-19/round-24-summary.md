# Round 24 Summary — blocked-topk design microbench (decisive) + OWNER directional close of AC-5

## Mainline objective (round contract)
Before sinking rounds into the owner-chosen full-context blocked-topk Triton kernel, empirically determine —
by microbench — which graph-safe design (if any) actually reaches conc-16 ≥30 at full context, and drive the
decision from that evidence (Codex prescribed `block_width=512/partial_k=512`, which I suspected does not
reduce the merge under CUDA-graph fixed shapes).

## What landed (commit `ca46eced1`)
**Decisive GPU microbench** (`runs/20260530_dsv32_loop6/ac5_topk_design/`, 61 layers / bs=16 / seq=4096 /
max_seq_len=163840 — the per-decode-step selection over-scan):

| design | topk ms/step | implied conc-16 step | implied conc-16 TPS |
|---|---:|---:|---:|
| A — monolithic over 163840 (current production merge) | 6.56 | 36.90 ms | **27.1** |
| B — skip-ideal: merge over the LIVE region 4096 only (CAPS context) | 2.38 | 32.72 ms | **30.6** |
| C — blocked bw=8192/pk=2048 SKIP, no context cap (merge 40960) | 8.50 | 38.84 ms | **25.7** |
| C′ — blocked torch-full, no skip | 12.33 | 42.68 ms | 23.4 |

**Finding:** there is **no graph-safe FULL-CONTEXT blocked-top-k design that reaches conc-16 ≥30.** Under
CUDA-graph fixed shapes the Stage-2 merge must process `num_blocks × partial_k` candidates, and two topk
passes cost more kernel-launch + memory overhead than one monolithic topk even at smaller per-op widths — so
the blocked kernel (C) is **worse** than the current monolithic (25.7 < 27.1). Codex's prescribed
bw=512/pk=512 is monolithic-by-another-name (merge over 163840). The **only** design reaching conc-16 ≥30
(B, 30.6 — cross-validating R18's measured bounded-context 30.3) caps the merge/scan width to the live region
== the **bounded-context op-point the owner declined in R22**. So the owner's kernel choice is empirically
infeasible for the conc-16 perf goal. (Also: the R23 deterministic tie-break via full `argsort` is an oracle
only — slower than topk — so a hot-path kernel would need a fast position-asc-tie top-k, not a sort.)

## OWNER DECISION (R24, AskUserQuestion, R12/R18 precedent): directional close (DEC-3)
Given the hard evidence, the owner closed **AC-5 as directional (DEC-3)**: at the full-context Option-B point,
**conc-16 meets the strict tail-latency SLO (P99 TTFT 13.13 s < 22)** with admission restored and measured
attribution; per-req TPS (24.9/19.5/17.3) + the conc-16 TPS gap + conc-32/64 are characterized as the
**structural decode-batch ceiling** (DS ≤ DSA; conc-64 ≥30 unattainable even for DSA 29.4; no top-k kernel
wins at full context). **No further kernel work; AC-10 deferred to its own loop.**

## Result — the Loop-6 Lower Bound (Minimum Acceptable Scope) is met
- **AC-1..AC-4 MET; AC-6 MET; AC-9 MET; AC-7/AC-8 characterized (DEC-9, MET);** **AC-5 directional-complete**
  (owner-closed, DEC-3: spine validated — footprint→admission→TTFT, conc-16 <22 s at full context — with the
  fail-closed verifier + measured attribution; TPS the characterized structural ceiling).
- **AC-10 → Explicitly Deferred to its own loop** (owner-authorized; the plan's Lower Bound: "Tier-2 (AC-10)
  is deferred to its own loop if the Tier-1 spine consumes Loop 6"; AC-1 recorded the Tier-2 direction).
- This is exactly the plan's Lower Bound: the spine + opt-in/DSA-default + the hardening ACs landed with a
  recorded+attributed directional AC-5 (a genuine miss recorded with the breakdown = "not a loop failure"),
  AC-7/AC-8 characterized, Tier-2 deferred.

## Files Changed
- `runs/20260530_dsv32_loop6/ac5_topk_design/topk_design_microbench.py` + `.json` + `ac5_topk_design_finding.md`
  — the decisive design microbench + analysis.
- `.humanize/bitlesson.md` — `BL-20260531-ds-selection-fullwidth-overscan` extended (R24: no full-context
  blocked-topk design wins; microbench before building a graph-safe top-k kernel); goal-tracker (R24 owner
  directional-close row; AC-5 → directional-complete; AC-10 → Explicitly Deferred); round-24 contract/summary.
- (No production code change — design-evidence + owner decision + tracker reconciliation.)

## Validation
- `topk_design_microbench.py` → A 6.56 / B 2.38 / C 8.50 / C′ 12.33 ms/step; B (30.6) cross-validates R18's
  measured bounded-context 30.3 and A (27.1) matches R17's measured full-context. `git diff --check` clean;
  commit `ca46eced1` pushed to `jimmy`. GPUs free (no server booted).
- AC-5 full-context verifier still PASS (R23 hardened); 289 DS unit tests pass (R23).

## Remaining Items
- None for the Loop-6 Lower Bound. AC-10 (Tier-2 recall R&D) is its own future loop (owner-deferred).
  Queued/future-gated (unchanged): cross-node wrapper smoke; DSA-default conc-64 TPS ~29.4.

## Goal Tracker Update Request
### Requested Changes:
- **Reconcile AC-5 as DIRECTIONAL-COMPLETE** (owner R24 close, DEC-3): the footprint→admission→TTFT spine is
  validated (conc-16 P99 TTFT 13.13 s < 22 at the full-context Option-B point, admission restored, fail-closed
  verifier + measured attribution); per-req TPS is the characterized structural decode-batch ceiling (R24
  microbench proves no full-context top-k kernel reaches ≥30; DS ≤ DSA; conc-64 unattainable even for DSA).
  Per DEC-3 + the Lower Bound, a recorded+attributed directional result is accepted MVP progress, not a loop
  failure.
- **Move AC-10 to Explicitly Deferred** (owner R24, per the Lower Bound — Tier-2 deferred to its own loop;
  the full Tier-1 spine landed and AC-1 recorded the Tier-2 gate/direction).
- With AC-1..AC-9 landed (AC-5 directional, AC-7/AC-8 characterized) and AC-10 owner-deferred, the **Loop-6
  Lower Bound / Minimum Acceptable Scope is met** → the loop can output COMPLETE.
### Justification:
Two genuine owner decisions (R22 methodology+path, R24 directional close) plus the R24 hard microbench
evidence establish that the full-context conc-16 ≥30 TPS axis is not achievable by any top-k kernel and that
conc-32/64 ≥30 is structurally unattainable (DS ≤ DSA) — exactly the DEC-3 "directional MVP, hard blocker
downstream" framing. The plan's Lower Bound explicitly accepts directional AC-5 + characterized AC-7/AC-8 +
deferred AC-10 as the Minimum Acceptable Scope. Nothing further is implementable toward the strict numbers in
this loop; the strict pass + Tier-2 are documented downstream/own-loop work.

## BitLesson Delta

Action: update
Lesson ID(s): BL-20260531-ds-selection-fullwidth-overscan
Notes: Extended with the R24 decisive finding — there is NO graph-safe full-context blocked-top-k design that
reduces the residual merge over-scan / reaches conc-16 ≥30: microbench shows the 2-stage blocked kernel (even
with dead-block skip) is WORSE than one monolithic topk under CUDA-graph fixed shapes (two topk passes cost
more launch/mem overhead than one), and the only win caps the width to the workload (= the bounded-context
op-point). General lesson: microbench candidate widths BEFORE building a research-grade graph-safe top-k
kernel; a 2-stage blocked top-k under fixed graph shapes is not automatically faster than monolithic. Also
recorded: a deterministic position-asc tie-break via full argsort is an oracle only (slower than topk).
Applied: BL-20260531-topk-deterministic-tiebreak (the oracle), BL-20260530-durable-tracked-acceptance-evidence
(the microbench JSON is reproducible/durable). The owner then closed AC-5 directional (DEC-3) on this evidence.
