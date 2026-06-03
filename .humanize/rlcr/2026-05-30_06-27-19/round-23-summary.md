# Round 23 Summary — deterministic top-k tie-break + AC-5 verifier workload-volume

## Mainline objective (round contract)
Fix the two Codex R22 prerequisites for the owner-chosen graph-safe blocked-topk kernel: (1) the blocked
top-k "exact oracle" was NOT identical to the monolithic selection on finite ties; (2) the AC-5 verifier was
fail-open on workload VOLUME (a 1-request artifact passed). Codex requires the tie contract fixed FIRST so
the kernel is written against a correct oracle.

## What landed (commit `2715b7382`)
1. **Deterministic top-k tie-break (shared contract).** Codex counterexample: all-ones scores, K=3,
   block_width=4 → monolithic `[4,5,6]` vs blocked `[4,6,7]`. Added a shared `_topk_by_score_then_pos(vals,
   pos, k)` helper that selects the top-K by **(score DESCENDING, then logical position ASCENDING)** — a
   stable position-ascending sort then a stable score-descending argsort, so equal scores resolve toward the
   lower position. Both `select_topk_sequence_order` (monolithic) and `blocked_topk_sequence_order` (the
   oracle/eager fallback) now use it → **bit-identical, including on ties** (the all-ones case is `[0,1,2]`
   in both). 4 finite-tie regressions added (all-equal, ties crossing block boundaries, ties at the K
   boundary, ties mixed with `-inf`). **289 DS unit tests pass.** Moved the test-file `__main__` guard to the
   end (Codex queued #3 — direct class invocation now works).
2. **AC-5 verifier workload-VOLUME hardened.** Codex showed a 1-request artifact with self-consistent
   headlines passed. Now a **code-owned `EXPECTED_WORKLOAD`** constant is the authority (the JSON copy is
   documentation, asserted == code); `--verify` asserts per conc `completed == 192`, `duration_s >= window
   (300 s)`, and sidecar `trial_id` present, on top of the workload-identity + recompute-from-raw checks.
   **5 volume tamper tests each exit 1** (reduced completed=1 with consistent headlines, short duration,
   coordinated expected_workload+arrays mutation, JSON-doc tamper, trial_id removed); clean PASS.

## Result
The top-k exactness contract is now correct and shared — the graph-safe Triton blocked top-k (next round)
can be written against a correct oracle (bit-identical to the monolithic path incl. ties). The AC-5
full-context verifier is now fail-closed on metrics (recompute-from-raw + means), workload identity, AND
volume/duration/trial — Codex's R20/R21/R22 verifier gaps are all closed.

## Files Changed
- `python/sglang/srt/layers/attention/double_sparsity/selection_kernel.py` — `_topk_by_score_then_pos`
  helper; `select_topk_sequence_order` + `blocked_topk_sequence_order` use the shared deterministic tie-break;
  docstring updated.
- `test/registered/unit/layers/attention/test_double_sparsity_unit.py` — 4 finite-tie regressions; `__main__`
  guard moved to the end.
- `runs/20260530_dsv32_loop6/ac5_fullctx/ac5_fullctx_metrics_tool.py` + `ac5_fullctx_arrays.json` — code-owned
  `EXPECTED_WORKLOAD` + completed/duration/trial assertions.
- `.humanize/bitlesson.md` — new lesson `BL-20260531-topk-deterministic-tiebreak`; goal-tracker (R23 row);
  round-23 contract/summary (gitignored loop state).

## Validation
- `pytest test_double_sparsity_unit.py` → **289 passed** (9 subtests); direct `TestBlockedTopKExactness`
  invocation works. The all-ones K=3/bw=4 counterexample now matches in both selectors; ties-crossing-blocks
  and ties+(-inf) match.
- `ac5_fullctx_metrics_tool.py --verify` → PASS; 5 workload-volume tamper tests each exit 1 (incl. the exact
  Codex R22 reduced-completed gap). `git diff --check` clean; commit `2715b7382` pushed to `jimmy`. GPUs free
  (CPU/data round; no server booted).

## Remaining Items (the owner-chosen path)
- **Graph-safe Triton blocked top-k** in `retrieve_topk_graph_safe` (DSGraphState partial-score/partial-index
  scratch; per-block top-K; SKIP blocks entirely past each request's `seq_len`; merge under the now-correct
  deterministic tie-break; zero-alloc under CUDA-graph; ABI lock intact) + CUDA-graph replay/zero-alloc tests.
- **Full-context closed-batch conc-16 ≥30 TPS** proof, then the **full AC-5 client rerun** (np64-approved)
  with the hardened verifier.
- **Gated AC-10** — after AC-5 verified. Cross-node smoke (future-gated), DSA conc-64 TPS ~29.4 (queued).

## Goal Tracker Update Request
### Requested Changes:
- Mark Codex's R22 prerequisite blockers RESOLVED: the blocked-topk finite-tie hole (shared deterministic
  tie-break, bit-identical incl. ties, 4 regressions) and the verifier workload-volume fail-open (code-owned
  EXPECTED_WORKLOAD + completed/duration/trial asserts, 5 tamper tests). The AC-5 verifier is now fail-closed
  on metrics + identity + volume.
- AC-5/task6 stays Active for the owner-chosen graph-safe Triton kernel + the post-kernel rerun.
### Justification:
Codex required the tie contract fixed FIRST (before the kernel) and the verifier volume-anchored; both are
done and tamper-demonstrated. The remaining AC-5 work is the owner-chosen graph-safe kernel (writable now
against a correct oracle) + the np64 rerun.

## BitLesson Delta

Action: add
Lesson ID(s): BL-20260531-topk-deterministic-tiebreak
Notes: New lesson — when ≥2 implementations of a top-k/selection must agree (monolithic vs blocked vs a future
Triton kernel; an "exact oracle"), they DIVERGE on finite score ties unless they share an explicit
deterministic tie-break, because `torch.topk`/argsort break ties arbitrarily and differently across a
whole-array topk vs a per-block-then-merge topk (Codex's all-ones K=3/bw=4 counterexample). Fix: one shared
ordering (score DESC, then position ASC) via a stable pos-asc sort then a stable score-desc argsort; test ties
explicitly (distinct-score tests miss the divergence). Applied: BL-20260527-torch-topk-aliasing-corrupts-input
(fresh argsort outputs), BL-20260530-durable-tracked-acceptance-evidence (the verifier-volume hardening
extends "prove the artifact IS the claimed run" to completed/duration/trial with code-owned expected constants).
