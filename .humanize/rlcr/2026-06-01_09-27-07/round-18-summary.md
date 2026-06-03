# Round 18 Summary — Loop 7

## Mainline objective (round-18-contract.md)
**Complete AC-4: the graph-captured TP=8 lifted-width selector-equality artifact
(task16's last item) + a fully-consistent production-ready `m9` disposition
(task17), re-reviewed.**

## Outcome: ACHIEVED — AC-4 MET (production-ready); task16 + task17 closed.

## Work Completed
### task16 — graph-captured TP=8 lifted-width selector equality (the R17-review gap)
I first attempted the literal ask: an 8-rank NCCL + raw `torch.cuda.graph` capture of
`retrieve_topk_graph_safe`. **It deadlocked** (540 s timeout, SIGTERM + orphan cleanup)
— capturing an NCCL collective in a naive per-rank `torch.cuda.graph` needs the
production `cuda_graph_runner`'s coordination (shared graph pool + comm registration),
which a standalone unit harness cannot provide. So I proved the property by **composed
evidence**:
- **(a)** single-rank `retrieve_topk_graph_safe` at **4096/8192** captured in a real
  `torch.cuda.CUDAGraph`: **zero-alloc replay** + **bit-identical to the eager logical
  reference** (`TestLiftedWidthSelectionGraphCaptured`).
- **(b)** the eager 8-rank all-reduce equality at 4096/8192 (`TestTP8LiftedWidthDeterminism`)
  — the SUM all-reduce is rank-symmetric + deterministic.
- **(c)** the **live R17 TP=8 server** ran the selection under production CUDA graph and
  served correct **95%** recall (divergent ranks → corrupt all-reduced selection →
  degenerate output, which did not occur).

### task17 — `m9` full production-ready consistency (the R17-review gap)
Rewrote every contradictory section: validator no longer requires `--disable-cuda-graph`,
launcher no longer forces eager, `dequantize_k_cache_paged_out` + fixed-shape scratch is
the production decode path, graph-mode 4K NIAH 95% is the binding recall; removed every
"deferred / eager-required" claim; cleaned the stale "eager-only" comments in
`serve_double_sparsity.sh`, `selection_kernel.py`, and `dsa_backend.py`. **Re-reviewed via
`/humanize:ask-codex` (twice)**: R18 returned **"No runtime/design gap found blocking
AC-4"**, the **`(a)+(b)+(c)` composed evidence is an acceptable production-readiness close**
(a raw per-rank NCCL `torch.cuda.graph` harness is NOT required), and the speculative guard
is sound. Integrated its wording fixes (the two bullets that overclaimed "8-rank NCCL graph
capture" → the exact `(a)+(b)+(c)`).

### Bundled (R17-review queued hazard, now resolved)
A fail-closed validator guard rejecting `enable_lifted_budget_decode` + `--speculative-algorithm`
(the lifted CUDA-graph scratch is sized by `max_bs`, but speculative target-verify expands
the decode rows) + `test_validator_lifted_rejects_speculative`.

## Files Changed
- `test_ds_scorer_tp_determinism.py` (single-rank lifted-width selection graph-capture),
  `validator.py` (lifted+speculative guard), `test_scorer_variants.py` (guard test),
  `m9_tier2a_disposition.md` (consistency rewrite + re-review + wording fixes),
  `serve_double_sparsity.sh` / `selection_kernel.py` / `dsa_backend.py` (stale-comment cleanup).
- Commit `f9f6ec056` (local — loop hook keeps commits local until completion).

## Validation
- `TestLiftedWidthSelectionGraphCaptured` (4096/8192) + `TestLiftedBudgetABI` (13, incl.
  the speculative guard) pass; the focused lifted/TP subset → **60 passed**.
- Full DS unit suite → **350 passed + 9 subtests**, no regression. Default-off byte-identical.

## AC status after R18
- **AC-4 → MET (production-ready)**; **task16 + task17 closed**. With AC-1/3/5 (prior),
  **5/6 ACs MET**.
- AC-2 PARTIAL (task20), AC-6 PARTIAL (task19).

## Remaining Items (active mainline)
- **task19 (AC-6, next mainline)** — consolidated perf guardrails at conc-1/16 (TTFT,
  decode TPS/req, GPU mem, graph-replay, admission) + Tier-1 non-regression + the
  DS-vs-DSA recall/perf report.
- **task20 (AC-2)** — final strategic-gate supersession decision record.
- Evidence-hygiene queued: R8 oracle-sink provenance; plan-marker cleanup (pre-existing).

## BitLesson Delta
- Action: add
- Lesson ID(s): BL-20260602-nccl-collective-graph-capture-needs-runtime-not-unit-test
- Notes: a standalone N-rank NCCL-collective-under-raw-`torch.cuda.graph` unit test
  DEADLOCKS — NCCL collective graph capture needs the production `cuda_graph_runner`'s
  coordination (graph pool + comm registration), not a naive per-rank capture. Prove the
  property by composed evidence instead: (a) single-rank capture (`process_group=None`),
  (b) eager N-rank cross-rank equality (gloo), (c) a live server boot WITH CUDA graph + TP=N
  serving correct output (the real collective-under-production-capture). Hit the hard way
  (540 s timeout) and confirmed by the ask-codex re-review.

## Goal Tracker
Updated directly (Plan Version 25): R18 row; task16 + task17 → Completed and Verified;
**AC-4 MET (production-ready)**; the lifted+speculative queued issue → RESOLVED; Active =
task19, task20. No Goal Tracker Update Request needed.
