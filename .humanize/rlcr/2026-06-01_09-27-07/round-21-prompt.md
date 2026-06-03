Your work is not finished. Read and execute the below with ultrathink.

## Original Implementation Plan

**IMPORTANT**: Before proceeding, review the original plan you are implementing:
@development/loop7/refined_plan_v1.md

This plan contains the full scope of work and requirements. Ensure your work aligns with this plan.

---

## Round Re-anchor (REQUIRED FIRST STEP)

Before writing code:
- Re-read @development/loop7/refined_plan_v1.md
- Re-read @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/goal-tracker.md
- Re-read the most recent round summaries/reviews that led to this round
- Write the current round contract to @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/round-21-contract.md

Your round contract must contain:
- Exactly one **mainline objective**
- The 1-2 target ACs for this round
- Which issues are truly **blocking** that mainline objective
- Which issues are **queued** and explicitly out of scope
- Concrete success criteria for this round

Do not start implementation until the round contract exists.

## Task Lane Rules

Use the Task system (TaskCreate, TaskUpdate, TaskList) with one required tag per task:
- `[mainline]` for plan-derived work that directly advances this round's objective
- `[blocking]` for issues that prevent the mainline objective from succeeding safely
- `[queued]` for non-blocking bugs, cleanup, or follow-up work

Rules:
- `[mainline]` work is the round's primary success condition
- `[blocking]` work is allowed only when it truly blocks the mainline objective
- `[queued]` work must be documented but must NOT replace the round objective
- If a new bug does not block the current objective, tag it `[queued]` and keep moving on mainline work

Before executing each task in this round:
1. Read @/sgl-workspace/sglang/.humanize/bitlesson.md
2. Run `bitlesson-selector` for each task/sub-task
3. Follow selected lesson IDs (or `NONE`) during implementation

---
Below is Codex's review result:
<!-- CODEX's REVIEW RESULT START -->
# Round 20 Review Result

Mainline Progress Verdict: ADVANCED

Round 20 materially advanced AC-6: `perf_closed_batch.py --stream` now records first-token
arrival timestamps and fails closed on HTTP-200 empty streams, and the committed TTFT JSONs
match the new `m11_perf_consolidation.md` TTFT table. I verified the headline values from
the JSON artifacts:

- DSA/native-NSA: 150.8 ms c1 short; 307.1 / 309.2 ms c16 short p50/p99; 1161.5 / 1322.1 ms c16 p770 p50/p99.
- DS-default: 183.3 ms c1 short; 371.7 / 374.0 ms c16 short; 1210.9 / 1400.2 ms c16 p770.
- DS-hybrid: 178.4 ms c1 short; 363.3 / 365.1 ms c16 short; 1218.1 / 1405.2 ms c16 p770.

The probe parses SGLang `/generate` SSE chunks, timestamps the first non-empty streamed
text, computes post-first-token decode TPS, and raises instead of recording an empty stream
as a completion (`development/loop7/perf_closed_batch.py:71-137`). `python -m py_compile
development/loop7/perf_closed_batch.py` passed.

I do **not** accept Loop-7 completion, and I do **not** accept full task19 verification yet.
The measurements are useful, but the source artifact task20 must cite is still missing
required exact provenance.

## Acceptance Criteria Audit

| AC | Status | Evidence / Blocker |
|----|--------|--------------------|
| AC-1 | MET | Prior R8 acceptance remains valid: fail-closed oracle, dedicated sink, oracle-off byte-equivalence + zero-alloc replay, separated baseline, stride=1 reference, and AC-1.1 force-inclusion evidence. |
| AC-2 | PARTIAL | Recall uplift evidence exists, but task20 final strategic-gate supersession decision record is still not written (`development/loop7/refined_plan_v1.md:165-167`, `:195-196`). |
| AC-3 | MET | Prior R7-R9 evidence remains accepted: graph-safe non-learned scorer/head/anchor variants, TP=8 determinism, within-budget parity, MMLU within 1.0 pp, and N=50 graph-mode recall matrix. |
| AC-4 | MET | Prior R18 review accepted production-ready lifted-budget decode and the consistent M9 disposition. |
| AC-5 | MET | 64K mem0.7 servability remains accepted; 128k remains out of Loop-7 scope. |
| AC-6 | PARTIAL | R20 records the missing TTFT values, but the per-run artifacts do not carry the required run provenance, and `m11` cites inconsistent commit SHAs. |

## Mainline Gaps

1. **task19 / AC-6 provenance is incomplete and internally inconsistent.**

   Evidence:
   - The R20 contract requires the fresh TTFT artifacts to include TTFT stats, streaming decode TPS, GPU memory, graph status, admission/served counts, exact launch args/config, commit SHA, and GPU type (`round-20-contract.md:48-51`).
   - The committed per-run TTFT artifacts contain the metrics, but not the required provenance. Example: `ttft_ds_default_c16.json` has label/mode/conc/completed, TTFT array, decode TPS, token counts, and wall time only (`development/loop7/ttft_ds_default_c16.json:1-36`). The DSA and hybrid files have the same shape (`development/loop7/ttft_dsa_c16.json:1-36`, `development/loop7/ttft_ds_hybrid_c16.json:1-36`).
   - `m11_perf_consolidation.md` says "R19 decode-TPS on `f9f6ec056`; R20 TTFT guardrails on `68969deb0`" (`development/loop7/m11_perf_consolidation.md:7`). In the actual history, `f9f6ec056` is R18, `68969deb0` is R19, and `30173f08b` is the R20 commit. The R20 summary itself names `30173f08b` as the round commit (`round-20-summary.md:67`).
   - This matters because `m11` is explicitly the source artifact for task20 (`development/loop7/m11_perf_consolidation.md:3-7`). A final decision record should not cite a source artifact with offset commit provenance.

   Required implementation plan:
   - Do not rerun measurements unless the exact run state cannot be reconstructed.
   - Add a `run_provenance` object to every `ttft_*.json` containing: server-code commit used for the live run, measurement-tool/artifact commit, whether the tree was dirty during measurement, GPU type/count, exact server launch command, effective DS/native-NSA config, mem fraction, graph/radix/overlap flags, admission count, graph evidence source, memory source/value, and artifact path.
   - Correct `m11_perf_consolidation.md` so the R19/R20 commit story is exact. If the servers were measured on parent commits with uncommitted probe changes, say that explicitly instead of labeling the parent as the artifact commit.
   - Keep the current TTFT table values if the provenance can be reconstructed; after the repair, move task19 to verified and let task20 cite `m11`.

2. **task20 / AC-2 remains unfinished original-plan work.**

   Evidence:
   - M4 requires a decision record superseding the strategic gate's Tier-2.A-primary ordering with M0 evidence (`development/loop7/refined_plan_v1.md:165-167`).
   - The task table makes task20 a coding task depending on task19 (`development/loop7/refined_plan_v1.md:195-196`).
   - Claude's summary correctly admits task20 is still remaining mainline work, so the loop cannot be marked complete.

   Required implementation plan:
   - After task19 provenance is repaired, write the final strategic-gate supersession record as the loop-close decision artifact.
   - Cite, in one coherent chain: M0 regime attribution, AC-1 oracle closure and stride provenance, AC-2 recall matrix/CI rule, AC-3 hybrid scorer non-regression, AC-4 production-ready lifted-budget disposition, AC-5 64K servability, and the corrected AC-6 `m11` perf/TTFT guardrails.
   - State exactly what changed from the Loop-6 gate: Tier-2.A was a sound primary choice before M0, but M0/R4-R8 evidence showed long-context 16K/64K is scorer-limited or budget-partial, making Tier-2.B the primary long-context path and Tier-2.A a bounded 4K lever.
   - Include the R8 stride/oracle provenance explicitly: cite the committed `oracle_stride_reference.json` plus the `selection_kernel.py` stride=1 call site, or archive/hash the raw sink used.

## Blocking Side Issues

None. The streaming probe itself is adequate for the missing TTFT measurement, and I did
not find a runtime issue that invalidates the recorded TTFT values.

## Queued Side Issues

1. Remove plan/workflow markers from production code/comments/tests before final cleanup/merge.
2. Learned/distilled selector work remains out of scope unless explicitly approved under DEC-5.

## Goal Alignment Summary

ACs: 6/6 addressed (4 met, 2 partial) | Forgotten items: 0 | Unjustified deferrals: 0

No original task is forgotten. No Explicitly Deferred item exists. The current active
mainline is now task19 provenance repair followed by task20 final decision record.

## Goal Tracker Update Requests

I updated `goal-tracker.md` directly:

- bumped Plan Version to 30 for Round 20 Review;
- added a Round 20 Review plan-evolution row accepting the TTFT values but rejecting full task19 verification until provenance is corrected;
- reactivated `task19` as partial/provenance repair;
- left `task20` active, waiting on the corrected AC-6 source artifact;
- removed the unverified AC-6 row from Completed and Verified;
- left Explicitly Deferred empty.

PENDING
<!-- CODEX's REVIEW RESULT  END  -->
---

## Goal Tracker Reference

Before starting work, **read** @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/goal-tracker.md to understand:
- The Ultimate Goal and Acceptance Criteria you're working toward
- Which tasks are Active, Completed, or Deferred
- Which side issues are blocking vs queued
- Any Plan Evolution that has occurred
- The latest side-issue state that needs attention

**IMPORTANT**: Keep the mutable section of `goal-tracker.md` up to date during the round.
Do NOT change the immutable section after Round 0.
If you cannot safely reconcile the tracker yourself, include an optional "Goal Tracker Update Request" section in your summary (see below).

## Mainline Guardrails

- Keep the mainline objective from @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/round-21-contract.md stable for this round
- Do not let queued issues take over the round
- If Codex reported several findings, classify them into:
  - mainline gaps
  - blocking side issues
  - queued side issues
- Only mainline gaps and blocking side issues should drive the next code changes

---

Note: You MUST NOT try to exit by lying, editing loop state files, or executing `cancel-rlcr-loop`.

After completing the work, please:
0. If the `code-simplifier` plugin is installed, use it to review and optimize your code. Invoke via: `/code-simplifier`, `@agent-code-simplifier`, or `@code-simplifier:code-simplifier (agent)`
1. Commit your changes with a descriptive commit message
2. Write your work summary into @/sgl-workspace/sglang/.humanize/rlcr/2026-06-01_09-27-07/round-21-summary.md

## Task Tag Routing Reminder

Follow the plan's per-task routing tags strictly:
- `coding` task -> Claude executes directly
- `analyze` task -> execute via `/humanize:ask-codex`, then integrate the result
- Keep Goal Tracker Active Tasks columns `Tag` and `Owner` aligned with execution

**Optional fallback**: if you could not safely update the mutable section of `goal-tracker.md` directly, include this section in your summary:
```markdown
## Goal Tracker Update Request

### Requested Changes:
- [E.g., "Mark Task X as completed with evidence: tests pass"]
- [E.g., "Add to Blocking Side Issues: bug Y blocks AC-2"]
- [E.g., "Add to Queued Side Issues: cleanup Z is non-blocking"]
- [E.g., "Plan Evolution: changed approach from A to B because..."]
- [E.g., "Defer Task Z because... (impact on AC: none/minimal)"]

### Justification:
[Explain why these changes are needed and how they serve the Ultimate Goal]
```

Codex will review your request and reconcile the Goal Tracker if justified.
