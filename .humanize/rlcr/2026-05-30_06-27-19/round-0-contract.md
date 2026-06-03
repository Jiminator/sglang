# Round 0 Contract

## Mainline Objective (exactly one)
Complete **Milestone 1 — the strategic gate + the pre-coding feasibility budget** (analyze-only, no code, no hardware): write the recall-R&D strategic decision doc and the binding HBM footprint-feasibility budget. These two artifacts GATE everything downstream — the binding lever selection in the budget determines what AC-3 (footprint code) builds, and the decision doc gates AC-10 (recall R&D). Per loop discipline, the spine starts at "gate → feasibility → footprint"; this round delivers the first two.

## Target ACs (1–2)
- **AC-1** — strategic gate: `ds_on_v32_decision.md` (pursue recall R&D after the spine lands, via a custom adjustable-`top_k` decode kernel; learned selector secondary).
- **AC-2** — feasibility budget: `footprint_feasibility.md` (HBM fixed-point per lever; **binding** lever selection; primary selection-equivalence metric + numeric threshold; minimum-reversible-opt-in justification).

Both are tagged `analyze` → executed via `/humanize:ask-codex`, then integrated by Claude.

## Blocking Side Issues in Scope
None. (One process issue handled in setup, not a code blocker: the goal-tracker IMMUTABLE Acceptance Criteria were truncated mid-AC-3.1 by the setup script and were completed this round, logged in the Plan Evolution Log — IMMUTABLE is only writable in Round 0.)

## Queued Side Issues Out of Scope
- AC-3 footprint code, AC-4/AC-5 hardware runs, AC-6–AC-9 hardening, AC-10 recall R&D — all gated on this round's outputs (AC-2's lever decision; AC-1's gate) and deferred to later rounds per the milestone sequence.
- Do **not** start footprint coding (AC-3) before the AC-2 budget artifact exists (AC-2 negative test).

## Round Success Criteria
1. `runs/20260530_dsv32_loop6/ds_on_v32_decision.md` exists and states the decision + rationale (kernel-lock / selection-quality evidence) + recall-R&D sequencing/consequence.
2. `runs/20260530_dsv32_loop6/footprint_feasibility.md` exists and records, per lever (fp16-baseline / int8-same-`label_dim` / page-level-two-stage): freed-HBM math, scale-storage overhead, target `max_total_num_tokens`, predicted achieved-conc@64 — grounded in the real Loop-5 anchors (f=0.6→1.55 GB/53056; f≈0.8→11.52 GB/396096 gen-OOM; f=0.897→31.18 GB boot-OOM; H200=139.8 GiB).
3. The budget makes the **binding** lever decision (int8 if predicted sufficient, else structural directly), names the **primary equivalence metric + numeric threshold** (top-k overlap@2048 ≥ 0.99) AC-3.1 must meet, and gives the one-line minimum-reversible-opt-in justification — and does **not** omit scale overhead or the larger-pool feedback.
4. goal-tracker.md, this contract, and round-0-summary.md are written; artifacts committed and pushed to `jimmy`.

## Out-of-Scope Guards
- Analyze-only round: no source/test/serve/bench code is modified.
- No hardware runs this round (no `runs/` benchmark artifacts beyond the two analyze docs).
- A code-only or analyze-only round is acceptable **if** the next round validates on hardware; two non-hardware rounds in a row with no new `runs/<date>_dsv32_loop6/` hardware artifact is a stall — so the next round must move to AC-3 footprint code (then AC-4 hardware).
