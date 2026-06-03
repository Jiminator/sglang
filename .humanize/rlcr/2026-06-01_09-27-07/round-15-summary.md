# Round 15 Summary — Loop 7

## Mainline objective (round-15-contract.md)
**task17 — the Tier-2.A landing disposition (deferred-with-evidence) that CLOSES AC-4.**
An `analyze` task: draft → `/humanize:ask-codex` review → integrate.

## Outcome: ACHIEVED — AC-4 closes via deferred-with-evidence; task16 explicitly deferred.

## Decision
Tier-2.A (the opt-in lifted-budget decode path) lands as an **opt-in, eager,
default-off research path with recorded served recall evidence**, and its
**production hardening (task16) is explicitly deferred to a follow-on**. This closes
AC-4 under the plan's "production-ready **or** deferred-with-evidence" branch (the
M4 dependency gate requires the disposition to exist; it now does).

## Justification (why deferred, not hardened)
The M0 oracle makes Tier-2.A **bounded-secondary**: a wider decode budget recovers
ONLY the **4K budget-limited** regime (proven served in R14: 75% → 95%, +20pp
material). **16K is budget-partial (~46% cap) and 64K is scorer-limited** — the
long-context goal that motivated the loop is served by the **landed,
production-ready Tier-2.B hybrid scorer** (AC-3 MET). The plan gates the
HIGH-COST/HIGH-RISK task16 kernel on *"the recall win justifies the heavy kernel"*;
a 4K-only win on a secondary lever does not. This is the plan-aligned,
theoretically-sound scoping decision given the measured evidence — not pragmatism
over correctness (the loop's correct primary lever, Tier-2.B, IS landed
production-ready).

## Why this is a VALID close (DEC-4/DEC-6 conditions, all satisfied)
1. **Recall evidence recorded** — M0 regime attribution + R14 served 4K recovery.
2. **DSA default untouched** — the `flashmla_kv` `dsa_index_topk` assert is unchanged;
   default-off decode byte-identical (default-off guard; full suite confirms).
3. **Research path gated out of production capture** — validator requires
   `--disable-cuda-graph` for the lifted path (the dequant allocates / not graph-safe).
4. **Eager number labeled as such** — the 95% is eager-mode; the graph number is a
   deferred-follow-on measure, not claimed here.

## Work Completed (`analyze`, Claude + ask-codex)
- **`development/loop7/m9_tier2a_disposition.md`** (new): the disposition record —
  the decision, the M0 + R14 evidence, the full landed opt-in surface (R10–R14), the
  DEC-4/DEC-6 conditions, the M0 bounded-secondary justification, and a **precise
  task16 follow-on scope** (alloc-free `out=` dequant, graph-safe fixed-shape compact
  remap, q-padding scratch, zero-alloc-replay + graph-mode recall re-measure,
  graph-captured TP=8 determinism, perf).
- **ask-codex review** (the `analyze` step): **"No high-signal invalidating issue
  found"** — the deferral is justified and the DEC-4/DEC-6 conditions are met.
  Integrated its two refinements: (a) added graph-captured TP=8 lifted-width
  determinism to the task16 follow-on (the R14 TP test is the eager/logical path);
  (b) corrected the wording so 16K (budget-partial/capped) and 64K (scorer-limited)
  are not compressed into one claim. Output cited:
  `.humanize/skill/2026-06-02_16-34-28-2810456-a017c2ed/output.md`.

## Files Changed
- `development/loop7/m9_tier2a_disposition.md` (new). Commit `b70f48d36`.
- **No production-code change** (a documentation/decision round).

## Validation
- Full DS unit suite (4 files) → **341 passed + 9 subtests** (unchanged — no code
  touched; confirms no regression).
- The disposition's claims are backed by committed artifacts (`m8`, the NIAH JSONs,
  `m0_oracle_finding_r4.md`) and the implementation gates Codex re-verified
  (validator eager-required, `_forward_lifted_budget` behind the default-off guard,
  the default `flashmla_kv` assert intact).

## AC status after R15
- **AC-4 → MET** (closed via deferred-with-evidence; the disposition record exists,
  recall evidence recorded, DSA default untouched, research path eager-gated).
- AC-1/AC-3/AC-5 MET (prior). **4/6 MET.**
- AC-2 PARTIAL (task20 final decision record), AC-6 PARTIAL (task19 perf consolidation).

## Remaining Items (active mainline)
- **task19 (AC-6, next mainline)** — final perf guardrails at conc-1/16 (TTFT, decode
  TPS/req, GPU mem, graph-replay, admission) + Tier-1 spine non-regression + the
  consolidated DS-vs-DSA recall/perf report.
- **task20 (AC-2)** — the final strategic-gate supersession decision record (supersede
  the Loop-6 Tier-2.A-primary ordering with the M0/R7/R8/R14 evidence + the AC-4 disposition).
- **task16** — Explicitly Deferred (follow-on; scope in `m9`).
- Evidence-hygiene queued: R8 oracle-sink provenance; plan-marker cleanup (pre-existing).

## BitLesson Delta
- Action: none
- Lesson ID(s): NONE
- Notes: A disposition/decision round — no new reusable engineering pitfall. The
  decision (defer a bounded-secondary lever's production hardening per the measured
  M0 evidence, closing via DEC-4/DEC-6 deferred-with-evidence) is a project-specific
  strategic call recorded in `m9`, not a cross-round engineering lesson. The
  evidence-methodology lessons it relies on (eager-vs-graph labeling, CP materiality,
  durable-tracked-acceptance-evidence) already exist.

## Goal Tracker
Updated directly (Plan Version 20): R15 Plan Evolution row; task17 → Completed and
Verified (pending R15 review); **task16 → Explicitly Deferred** with justification +
AC-impact (none on the long-context goal); Active Tasks now task19 + task20. No Goal
Tracker Update Request needed.
