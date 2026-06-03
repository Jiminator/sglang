# Round 23 Contract

## Mainline Objective (exactly one)
**Fix the top-k exactness contract (deterministic tie-break) and the AC-5 verifier workload-volume gap —
the two prerequisites Codex R22 requires before the graph-safe blocked-topk kernel.** Codex showed (a) the
R22 `blocked_topk_sequence_order` "exact oracle" claim is FALSE on finite ties (blocked ≠ monolithic when
scores are equal — real after int8/scaled scoring), and (b) the AC-5 verifier still passes when each conc is
reduced to one internally-consistent request (no completed-count / duration / trial assertions; expected
constants read from the tamperable JSON). This round: define ONE deterministic ordering (score descending,
then logical position ascending) shared by `select_topk_sequence_order` + `blocked_topk_sequence_order` (the
contract the future graph-safe kernel must also honor) with finite-tie regressions; and tighten the verifier
to code-own the expected workload + assert completed/duration/trial identity. The graph-safe Triton kernel +
the post-kernel AC-5 rerun are the next round(s) (Codex's items 2-3, 5 — they build on this round's contract).

## Target AC(s)
- **AC-5 (task6)** — the done-criterion (owner-approved np64 methodology; owner-chosen full-context kernel
  path). `coding` (deterministic tie-break + tests; verifier hardening — data/CPU, no production hot-path
  change to the graph topk yet).

## Truly Blocking This Objective
- **The finite-tie exactness hole** (Codex R22 blocking issue 2): the oracle the Triton kernel will be
  written against is wrong on ties — must be fixed FIRST (Codex item 1) or the kernel can match a wrong
  oracle while diverging from the monolithic path on real tied scores.
- **The verifier workload-volume fail-open** (Codex R22 blocking issue 1): until `--verify` asserts the
  approved completed count + measured duration + trial identity from each sidecar/arrays, a degenerate
  one-request artifact passes — not acceptance-grade.

## Queued / Explicitly Out of Scope This Round
- **The graph-safe Triton blocked top-k** in `retrieve_topk_graph_safe` (Codex item 2) + its CUDA-graph
  zero-alloc tests (item 3) — the research-grade GPU kernel; it MUST honor this round's deterministic
  tie-break contract. Next round.
- **The post-kernel AC-5 rerun** (item 5) — depends on the kernel.
- **AC-10** — gated. **Cross-node smoke** — future-gated. **DSA conc-64 TPS ~29.4** — queued.

## Concrete Success Criteria
1. **Deterministic tie-break shared:** `select_topk_sequence_order` and `blocked_topk_sequence_order` both
   select the top-K by (score descending, then logical position ascending) — implemented via a shared helper
   (stable argsort). `blocked_topk_sequence_order` == `select_topk_sequence_order` on FINITE TIES:
   all-equal scores, ties crossing block boundaries, ties at the K boundary, ties mixed with `-inf` padding
   — new regressions that FAIL under the old code and pass now. 285+ DS unit tests still pass.
2. **Verifier workload-volume hardened:** `--verify` code-owns the AC-5 `EXPECTED_WORKLOAD` constants (JSON
   copy is documentation only, cross-checked against code), and asserts per conc: `completed` == the approved
   np64 count (192), `duration_s >= measurement_window_seconds`, sidecar `trial_id` present, plus the existing
   workload/operating-point fields. Tamper tests: reduced `completed`, shortened `duration_s`, and a
   coordinated `expected_workload`+sidecar mutation — each exit 1; clean PASS.
3. **Cleanup** (Codex queued 3): move the test file's `if __name__ == "__main__"` guard to the end so the new
   class is importable for direct invocation.
4. GPUs free (CPU/data round). Commit + push to `jimmy`; goal-tracker + round-23-summary + BitLesson Delta.

## Applicable BitLessons (confirm per-task via bitlesson-selector)
- `BL-20260527-torch-topk-aliasing-corrupts-input` (no aliasing in the topk/argsort outputs).
- `BL-20260530-durable-tracked-acceptance-evidence` (verifier must prove the artifact IS the claimed run —
  now extended to workload VOLUME/duration, code-owned expected constants).

## Out-of-bounds reminders
No change to the production graph-safe `retrieve_topk_graph_safe` hot path this round beyond what the shared
tie-break requires (the Triton kernel is next round). No ABI-lock / FlashMLA-assert / `top_k` change. No
plan-process tokens in code/comments. Do not change the DS-fair AC-12 gate. Must not exit by lying / editing
loop state / cancel.
