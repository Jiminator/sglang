# Round 22 Summary — AC-5 verifier hardened + owner decisions + blocked-topk foundation

## Mainline objective (round contract)
Harden the AC-5 full-context verifier to fail closed on the workload + operating-point IDENTITY (Codex R21
demonstrated a fail-open gap), and obtain the two owner decisions Codex says require explicit approval for
the loop to converge (np64-vs-np320 methodology; conc-16 full-context TPS kernel-vs-rescope).

## What landed
1. **Verifier hardened — fail-closed on workload identity** (commit `704be382f`). Codex R21 mutated the
   sidecar `mode`/`num_prompts`/`isl_total_tokens`/`osl_tokens` and `server_args.max_total_num_tokens` to
   garbage and `--verify` still passed. Now `ac5_fullctx_metrics_tool.py` embeds an `expected_workload` and
   asserts, on EVERY sidecar: `mode=double_sparsity`, sidecar `concurrency`==artifact key, ISL 4096 / OSL 512,
   `num_prompts`/warmup/window == the recorded np64-steady-state methodology, `chunked_prefill_size=8192`,
   `max_total_num_tokens=396096` — plus the existing flag invariants and the recompute-from-raw metric checks.
   **7 workload-metadata tamper tests each exit 1** (mode=baseline, num_prompts=320, isl=1, osl=1, max_total=1,
   conc-key mismatch, warmup=0); clean PASS.
2. **Owner decisions** (AskUserQuestion, R12/R18 precedent):
   - **(a) AC-5 methodology = np64 steady-state APPROVED.** The literal `NUM_PROMPTS=320` is rejected as
     cold-flood-misleading per the verified BitLesson (window cold-ramps; fixed-count floods the queue →
     P99 TTFT ≈ full 320-drain ~300s). The committed full-context AC-5 evidence's methodology is now
     **owner-approved** (recorded as plan evolution).
   - **(b) conc-16 full-context TPS path = implement the full-context blocked-topk kernel** (owner chose the
     exact research-grade kernel over the bounded-context rescope / directional-accept).
3. **Blocked-topk foundation** (commit `8ab6c7db0`): exact torch `blocked_topk_sequence_order` in
   `selection_kernel.py` returning the IDENTICAL ascending positions + valid_lengths as the monolithic
   `select_topk_sequence_order` (per-block top-min(K,bw) → merge → global top-K; exact because a global-top-K
   token has within-block rank ≤ its global rank ≤ K). This is the exactness oracle + eager fallback for the
   graph-safe Triton skip-kernel (whose value is skipping blocks entirely past each request's `seq_len`).
   **4 registered adversarial regressions (6 subtests)**: all-winners-in-one-block, masked/short sequences,
   block-boundary seq, padding (n not a multiple of bw), K≥block_width (single block), K>n. 285 DS unit
   tests pass. ABI lock untouched.

## Result
The AC-5 full-context evidence verifier is now fully fail-closed (metrics recompute-from-raw + workload
identity + operating point; 13 tamper tests across R21/R22 each exit 1). The AC-5 measurement methodology is
owner-approved (np64). The owner-chosen conc-16 TPS path (the full-context blocked-topk kernel) has its exact
algorithm + adversarial regression suite landed — the foundation the graph-safe Triton kernel must match.

## Files Changed
- `python/sglang/srt/layers/attention/double_sparsity/selection_kernel.py` — `blocked_topk_sequence_order`
  (exact, the oracle/eager-fallback).
- `runs/20260530_dsv32_loop6/ac5_fullctx/ac5_fullctx_metrics_tool.py` + `ac5_fullctx_arrays.json` — verifier
  hardened (`expected_workload` + per-sidecar workload-identity assertions).
- `test/registered/unit/layers/attention/test_double_sparsity_unit.py` — `TestBlockedTopKExactness` (4 tests).
- `.humanize/bitlesson.md` — `BL-20260530-durable-tracked-acceptance-evidence` extended (verifier must prove
  workload identity, not just a subset of flags); goal-tracker (R22 row + owner decisions); round-22
  contract/summary (gitignored loop state).

## Validation
- `ac5_fullctx_metrics_tool.py --verify` → PASS; 7 workload-metadata tamper tests each exit 1 (Codex's exact
  R21 gaps closed).
- `pytest test_double_sparsity_unit.py` → **285 passed** (281 + 4 new; 6 subtests). `git diff --check` clean;
  commits `704be382f` + `8ab6c7db0` pushed to `jimmy`. GPUs free (data/CPU round; no server booted).

## Remaining Items (the owner-chosen path)
- **Graph-safe Triton blocked top-k** in `retrieve_topk_graph_safe`: a zero-alloc kernel using DSGraphState
  partial-score/partial-index scratch that computes per-block top-K and SKIPS blocks entirely past each
  request's `seq_len` (sentinel-filled on device), then merges to the same result as the monolithic path
  (now oracle-tested). This is the actual perf win for full-context conc-16.
- **Full-context closed-batch conc-16 ≥30 TPS** re-measure after the kernel, then the **full AC-5 client
  workload rerun** (np64-approved) with the hardened verifier.
- **Gated AC-10** — after AC-5 verified. Cross-node smoke (future-gated), DSA conc-64 TPS ~29.4 (queued).

## Goal Tracker Update Request
### Requested Changes:
- Record the R22 **owner decisions** as accepted plan evolution: (a) AC-5 methodology = np64 steady-state
  (warmup120/window300) — supersedes the literal NUM_PROMPTS=320; (b) the conc-16 full-context TPS path =
  the exact full-context blocked-topk kernel (bounded-context rescope declined).
- Mark the **AC-5 full-context evidence verifier** as acceptance-grade/fail-closed — resolving Codex's R20/R21
  verifier blocking issues. AC-5 stays Active for the owner-chosen kernel + the post-kernel rerun.
### Justification:
The owner explicitly approved np64 and chose the kernel path, resolving the two decisions Codex flagged as
requiring owner approval. The verifier is now fail-closed on both metrics and workload identity (the specific
R21 gaps closed with tamper tests). The remaining AC-5 work is the owner-chosen graph-safe kernel + rerun,
whose exact algorithm + regression oracle landed this round.

## BitLesson Delta

Action: update
Lesson ID(s): BL-20260530-durable-tracked-acceptance-evidence
Notes: Extended with the R22 workload-identity instance — a verifier that validates only a SUBSET of the
sidecar (selected server flags) is still fail-OPEN on workload identity (Codex mutated mode/num_prompts/ISL/
OSL/max_total_num_tokens and it passed); the verifier must prove the artifact IS the claimed AC run (workload
+ full operating point) via an `expected_workload` asserted on every sidecar, not just that the metric arrays
are self-consistent (7 workload tamper tests each exit 1). Applied existing lessons:
BL-20260527-torch-topk-aliasing-corrupts-input (the blocked top-k uses fresh topk outputs, no aliasing),
BL-20260530-cold-flood-not-steady-state-slo (the np64 methodology the owner approved). The blocked-topk
exactness property (global top-K ⊆ union of per-block top-K) is captured in the `blocked_topk_sequence_order`
docstring + its regression suite; a standalone lesson will be added when the graph-safe Triton kernel lands.
