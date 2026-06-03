# Round 15 Contract — Loop 7

## Mainline objective (EXACTLY ONE)
**task17 — write the Tier-2.A landing *disposition* record (deferred-with-evidence)
that CLOSES AC-4.**

Record the disposition for the opt-in lifted-budget decode path:
- The served 4K recall recovery is **proven and recorded** (R14: DS-lifted-4096
  95% vs DS-default-2048 75%, +20pp material, eager same-node, N=20).
- Production hardening (task16: alloc-free `out=`/scratch dequant, fixed-shape
  compact-builder scratch, q-padding scratch, CUDA-graph zero-alloc-replay proof,
  perf validation) is **explicitly deferred to a follow-on**, justified by the M0
  evidence that Tier-2.A is **bounded-secondary** (recovers only the 4K
  budget-limited regime; 16K budget-partial / 64K scorer-limited are NOT
  recoverable by a wider budget and are served by the landed Tier-2.B hybrid).
- The **DSA default is untouched** and the research path is **gated out of
  production CUDA-graph capture** (validator requires `--disable-cuda-graph`;
  default-off path byte-identical) — the DEC-4/DEC-6 conditions for a valid
  deferred-with-evidence close.

This is an **`analyze`** task: draft the disposition, review it via
`/humanize:ask-codex`, integrate the feedback, and land it as the disposition
artifact. AC-4 closes on this record existing (the plan's M4 dependency gate).

## Target AC(s)
- **AC-4** — closes it (the opt-in adjustable-budget decode, via the plan's
  "landed-or-deferred-with-evidence" branch).

## Blocking issues (truly block the mainline)
- **None.** The recall evidence (R14), the landed surface (R10–R14), and the
  eager-gating (R13 validator) all already exist; this round records the decision.

## Queued — explicitly OUT of scope this round (NOT closed/deferred)
- **task19 (AC-6)** — final perf guardrails (conc-1/16 TTFT, decode TPS/req, GPU
  mem, graph-replay, admission, Tier-1 non-regression). Next mainline; depends on task17.
- **task20 (AC-2)** — final strategic-gate supersession decision record. Depends on task19.
- **task16** — the production hardening itself: **explicitly DEFERRED** this round
  (added to the goal-tracker Explicitly Deferred table with justification + impact),
  carried to a follow-on per the disposition. NOT implemented this round.
- Evidence-hygiene queued: R8 oracle-sink provenance; plan-marker cleanup (pre-existing).

## Concrete success criteria
1. A disposition record (`development/loop7/m9_tier2a_disposition.md`) that records:
   (a) the served 4K recall evidence + the full landed surface (config ABI,
   validator gating, selector/backend lifted width, `_forward_lifted_budget`
   branch, the compact remap, all tests); (b) the **precise task16 follow-on scope**
   (the alloc-free dequant + graph-safe fixed-shape remap + capture/perf proof) so
   the deferral is well-specified, not vague; (c) the DEC-4/DEC-6 conditions
   satisfied — DSA default untouched, default-off byte-identical, research path
   eager-gated out of production capture; (d) the M0 bounded-secondary justification
   for deferral; (e) a clear "AC-4 closes via deferred-with-evidence" statement.
2. The record is **reviewed via `/humanize:ask-codex`** and the feedback integrated
   (the `analyze` routing); the ask-codex output is cited.
3. **goal-tracker** updated: AC-4 → **MET** (closed via deferred-with-evidence);
   task17 → done; **task16 → Explicitly Deferred** with justification + AC-impact
   analysis (impact: none on the long-context goal, which Tier-2.B serves; the 4K
   lever's production hardening is the only deferred item, recall evidence recorded).
4. **No production-code change required** (a documentation/decision round); the full
   DS unit suite still passes (no regression); no new plan-marker leakage.
5. Commit (the disposition doc + tracker, code/dev paths only; `.humanize/` excluded).

## Tag routing
- task17 is an **`analyze`** task → draft + `/humanize:ask-codex` review + integrate.
