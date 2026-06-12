# Refine Plan QA

## Summary

Refined `development/loop11/plan.md` (the converged Loop 11 gen-plan output, annotated on 2026-06-12 with four planning-annotation comments from a Claude + Codex idea pass) into `development/loop11/plan_v2.md`. All 4 raw comments were classified as `change_request` — each directed annotation content into the plan proper (kernel design space, headroom levers, rejected-ideas record, kickoff queue candidates). All 4 were applied; the affected sections (`Feasibility Hints and Suggestions`, `Task Breakdown`, `Dependencies and Sequence`, `Claude-Codex Deliberation`) were updated consistently. No questions, no research requests, no new pending decisions; the plan remains `converged`. Ran in `discussion` mode; the only user interaction was correcting the output path (`development/loop1/plan_v2.md` → `development/loop11/plan_v2.md`, a confirmed typo — the literal path pointed into the unrelated old loop-1 directory). Note: the annotations were originally written with uppercase `<COMMENT>` markers, which the comment scanner does not recognize; the markers in the input file were normalized to lowercase `<comment>` (content unchanged) to enable extraction.

## Comment Ledger

| CMT-ID | Classification | Location | Original Text (excerpt) | Disposition |
|--------|----------------|----------|-------------------------|-------------|
| CMT-1 | change_request | Feasibility Hints and Suggestions › Conceptual Approach (line 111) | "de-risk the kernel-cost assumption with block-scale reassociation … Fold into task5's kernel design space." | applied |
| CMT-2 | change_request | Feasibility Hints and Suggestions › Conceptual Approach (line 122) | "two additional EXACT headroom levers … Add both to task0's probe matrix and treat them as M1 candidates alongside task3/task4" | applied |
| CMT-3 | change_request | Feasibility Hints and Suggestions › Conceptual Approach (line 126) | "further ideas CONSIDERED AND REJECTED during the planning idea pass, recorded so the loop does not rediscover them" | applied |
| CMT-4 | change_request | Task Breakdown (line 199) | "queue candidates to seed at kickoff beyond this table … (3) PARKED value-affecting insurance requiring an owner ruling on AC-6 …" | applied |

## Answers

None — no comments were classified as `question`.

## Research Findings

None — no comments were classified as `research_request`. (The annotation content itself was produced and code-verified earlier in the same planning session via repository exploration and a Codex consultation; no additional research was required to integrate it.)

## Plan Changes Applied

### CMT-1: Block-scale reassociation added to the absorbed-latent kernel design space

**Original Comment:**
```
Planning annotation (2026-06-12, Claude+Codex idea pass): de-risk the kernel-cost assumption with block-scale reassociation — compute `score[t] = Σ_blocks scale_b(t) · (v_h[block] · latent_fp8[t, block])`, applying the 4 per-128-block fp32 scales to per-block partial dots instead of dequantizing 512 elements per token. The real-arithmetic product is identical; only fp32 rounding reassociates, which the declared value-affecting record already covers. A further variant quantizes v_h itself per-block to fp8 and runs fp8 tensor-core dots (additionally value-affecting; gate separately if tried). Fold into task5's kernel design space.
```

**Changes Made:**
Converted into a new bullet "**Kernel-cost de-risk (block-scale reassociation, part of task5's design space)**" in the Conceptual Approach bullet list, immediately after the "Kernel shape" bullet it de-risks.

**Affected Sections:**
- Feasibility Hints and Suggestions: new bullet in Conceptual Approach
- Other: none — task5's table row already says "in-kernel fp8 dequant; gated on live-path selection equivalence + oracle recall", which subsumes the design-space variant

**Cross-Reference Updates:** None required.

---

### CMT-2: Served-envelope right-sizing + bounded selector-width capture (exact headroom levers)

**Original Comment:**
```
Planning annotation (2026-06-12, Codex idea pass — numbers are estimates, verify via task0): two additional EXACT headroom levers, in neither the draft menu nor the task table: (1) Right-size the served runtime envelope for the SLO workload — cap `max_running_requests` near 64 and `cuda_graph_max_bs` at 64 (no captured batch above the workload cap). Serve logs imply defaults near 2048 requests / bs512 capture; ReqToTokenPool alone drops ~1.55 GiB → ~50 MiB at context 202752, and the measured ~4.68 GB DS graph memory should fall materially. The frozen DSA baseline keeps its own production defaults — the client workload is identical, so the comparison stays honest, but record the envelope as part of the served DS config. (2) Bounded DS selector-width graph mode — stop auto-capturing the full-context-width DS graph variant (today capture adds the full width alongside the workload buckets; a full-width fp32 score scratch at large capture bs is hundreds of MiB before masks and graph objects), fail-closed if a live sequence exceeds the declared width cap. Add both to task0's probe matrix and treat them as M1 candidates alongside task3/task4 — they compose with each other, with the indexer gate, and with table elimination.
```

**Changes Made:**
Converted into the plan-prose paragraph "**Served-envelope right-sizing + bounded selector-width capture**" in Conceptual Approach, and propagated per the comment's directive: the task0 probe matrix gained a third axis "{default vs right-sized envelope: `max_running_requests`≈64, `cuda_graph_max_bs`=64, bounded selector width}", and the M1 milestone gained a "Candidate levers" line (queued at kickoff, promoted on task0 evidence, explicitly NOT part of the M1 completion gate so the owner's DEC-3 "full M1 = task3+task4+ladder" contract is unchanged).

**Affected Sections:**
- Feasibility Hints and Suggestions: new paragraph
- Task Breakdown: task0 row probe-matrix axis added
- Dependencies and Sequence: M1 milestone "Candidate levers" line added

**Cross-Reference Updates:** None — no task IDs or AC IDs changed; the levers target AC-1 through task0/M1 plumbing already in place.

---

### CMT-3: Rejected-ideas record from the planning idea pass

**Original Comment:**
```
Planning annotation (2026-06-12): further ideas CONSIDERED AND REJECTED during the planning idea pass, recorded so the loop does not rediscover them: (a) sparse score transport (per-rank local top-k union replacing the dense cross-rank reduce) — exactness fails because the top-k of SUMMED TP scores need not appear in any rank's local top-k, and the exact candidate union is 8×2048 = 16384 > the ~4608 live width, saving nothing; (b) page-level two-stage prefilter — with 4608/64 = 72 pages and k = 2048, even a perfect prefilter keeps ≥ 32 pages (≤ 2.25× bound, looser in practice) for new metadata and write complexity; only revisit at ≥16k contexts; (c) offload re-checked at bs64 — ~5.9 GB/step ≈ ~118 ms at 50 GB/s PCIe, worse than the bs30 rejection; (d) compressing the written-slot bitmap — ~11 MB/rank, immaterial next to table and graph memory; (e) lower/adaptive top-k — changes the recall@2048 contract and the apples-to-apples DSA comparison (quality bar, not a tuning knob).
```

**Changes Made:**
Converted into the plan-prose paragraph "**Also considered and rejected during planning (recorded so the loop does not rediscover them)**" in Conceptual Approach, directly after the DS-Offload rejection paragraph it extends.

**Affected Sections:**
- Feasibility Hints and Suggestions: new paragraph

**Cross-Reference Updates:** None required.

---

### CMT-4: Kickoff queue candidates beyond the task table

**Original Comment:**
```
Planning annotation (2026-06-12, Claude+Codex idea pass) — queue candidates to seed at kickoff beyond this table: (1) EXACT per-step tax reducers, pull in if the bs64 AC-4 guard runs tight: fuse the radix top-k emit with the logical→physical gather so winners are written as physical slots directly (drops a separate kernel; selected-index plumbing measured ~11.7 ms per 10-step window at bs30); bf16-primary score scratch when the served path is already bf16-authoritative (removes the fp32 scratch plane and the fp32→bf16 copy; must prove selection bit-identity vs the current conversion); workload-bound selector width ladder (e.g. 4096/4352/4608 instead of 5120-only — ~10% less selector work at the cap, more during early decode; only after the graph-memory controls land, since extra buckets add capture variants). (2) Fallback if absorbed-latent slips: trim the table-path label-write projection (avoid projecting full [K_nope|V] just to slice label channels) — exact only if the quantized-linear semantics are preserved; superseded by task6. (3) PARKED value-affecting insurance requiring an owner ruling on AC-6 mechanism compatibility BEFORE any work (selection reuse means not every (layer, step) re-runs query·signature scoring): cross-step lazy top-k refresh (score/reduce/top-k cost ÷ N, recent window force-included) and cross-layer selection sharing in small layer groups. (4) The draft's menu item 6 (serving-side admission/scheduling levers) deliberately remains a conditional queue candidate, not a task — justify with a measured queueing trace first; memory levers treat the cause.
```

**Changes Made:**
Converted into a numbered "Kickoff queue candidates beyond this table" list under Task Breakdown, immediately after the binding queue-population paragraph, so the loop's kickoff queue-seeding step picks it up. Item (3)'s "owner ruling on AC-6 BEFORE any work" was kept as a conditional gate attached to those candidates rather than opened as a new `DEC-N`: no decision is needed unless the insurance candidates are ever pulled from the queue, so the plan's converged status and empty pending-decision set are preserved. The Convergence Status subsection gained a traceability bullet recording the post-convergence refinement and this rationale.

**Affected Sections:**
- Task Breakdown: numbered candidate list added after the queue paragraph
- Claude-Codex Deliberation: Convergence Status traceability bullet added

**Cross-Reference Updates:** None — no task IDs or AC IDs changed; candidate items reference existing task6/AC-4/AC-6 identifiers, all of which exist.

---

## Remaining Decisions

None. All four `DEC-*` items in the plan were already resolved by the owner during gen-plan; this refinement introduced no new pending decisions. The one decision-like item (owner ruling on AC-6 mechanism compatibility for cross-step/cross-layer selection reuse, from CMT-4) is documented in the plan as a conditional gate that must be obtained before work starts on those specific queue candidates — it does not block the plan and is not opened as a `DEC-N` unless those candidates are pulled.

## Refinement Metadata

- **Input Plan:** development/loop11/plan.md
- **Output Plan:** development/loop11/plan_v2.md
- **QA Document:** .humanize/plan_qa/plan-qa.md
- **Total Comments Processed:** 4
  - Questions: 0
  - Change Requests: 4
  - Research Requests: 0
- **Plan Sections Modified:** Feasibility Hints and Suggestions (Conceptual Approach); Task Breakdown; Dependencies and Sequence (Milestones); Claude-Codex Deliberation (Convergence Status)
- **Convergence Status:** converged
- **Refinement Date:** 2026-06-12
