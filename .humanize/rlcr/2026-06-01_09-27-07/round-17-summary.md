# Round 17 Summary — Loop 7

## Mainline objective (round-17-contract.md)
**task16 (part 2) — wire the R16 graph-safe primitives into production CUDA graph,
relax the validator, prove zero-alloc backend replay, and confirm a live CUDA-graph
boot.** (The R15 review STALLED the loop, overriding the deferred close and requiring
task16 to be implemented.)

## Outcome: ACHIEVED — task16 COMPLETE; AC-4 closes via the production-ready branch.

## Work Completed (`coding` + `analyze`, Claude + ask-codex)
1. **`DSGraphState` + `allocate_graph_state`** (`cuda_graph.py`): lifted scratch
   (`lifted_page_table`, `lifted_compact_indices`, `lifted_valid_counts`,
   `lifted_compact_kv`, `lifted_q_padded`), allocated **only when
   `enable_lifted_budget_decode`**; threaded from both metadata sites in `dsa_backend.py`.
2. **`_forward_lifted_budget`** (`dsa_backend.py`): graph path — slice the scratch to
   the captured bs/width, run `build_lifted_compact_kv_fixed` (fixed-shape builder +
   alloc-free `out=` dequant into scratch), attend via FlashMLA with a q-padding
   scratch. The eager `build_lifted_compact_kv` stays as the non-graph fallback
   (resolved via `getattr(self, "forward_metadata", None)` so partial test stubs work).
3. **`_forward_flashmla_sparse`**: optional q-padding scratch param (write real heads,
   the pad tail stays 0 from allocation, trimmed output); default callers byte-identical.
4. **Validator**: removed the lifted `--disable-cuda-graph` rejection (path is now
   graph-safe); the default `flashmla_kv` `dsa_index_topk` assert is untouched.
   `serve_double_sparsity.sh`: `LIFTED_BUDGET=1` no longer forces eager.
5. **task17 (production-ready disposition)**: rewrote `m9_tier2a_disposition.md` from
   deferred → **production-ready**; recorded `m10_lifted_graph_finding.md`. Re-reviewed
   via `/humanize:ask-codex` (**"No invalidating design gap found"**); integrated its 3
   points (reframed the graph-captured TP=8 item as integratively-evidenced; added the
   fp8-op-point scope caveat; cleaned the stale deferred prose).

## Validation
- **Offline (GPU)**: the wired backend `_forward_lifted_budget` replays **zero-alloc**
  under a real `torch.cuda.CUDAGraph` at **4096 and 8192**
  (`TestLiftedBudgetBackendGraphSafe`), matching the eager reference.
- **LIVE (8×H200)**: server booted **WITHOUT `--disable-cuda-graph`**; the full forward
  (incl. lifted decode) **captured** ("fired up"); decode batches log **`cuda graph:
  True`** (#token 4416); **graph-mode NIAH 4K N=20 = 95% (19/20)** — matches the eager
  95% and confirms the **+20pp recovery over DS-default-2048 (~75%) holds in production
  graph mode**; served 20/20, 0 admission failures; **3.4× faster** than eager (13.8s vs
  46.8s); ~14.5 tok/s; ~70 MB lifted scratch at `--cuda-graph-max-bs 8`.
  (`ds_meta=None` under graph is the **expected** eager-only-meta behavior, confirming
  the decode ran captured.) `m10_lifted_graph_finding.md`, `niah_ds_lifted4096_graph.json`.
- **Non-regression**: default-off path byte-identical; full DS unit suite → **347 passed
  + 9 subtests**.

## Files Changed
- `cuda_graph.py`, `dsa_backend.py`, `validator.py`, `serve_double_sparsity.sh`,
  `test_lifted_budget_decode.py`, `test_scorer_variants.py` — commit `6453562e9`.
- `m9_tier2a_disposition.md` (production-ready), `m10_lifted_graph_finding.md` (new),
  `niah_ds_lifted4096_graph.json` (new) — commit `41e0af078`. (Both local — loop hook.)

## AC status after R17
- **AC-4 → MET (production-ready)**; **task16 + task17 done**. With AC-1/3/5 (prior),
  **5/6 ACs MET**.
- **AC-2 PARTIAL** (task20 final decision record), **AC-6 PARTIAL** (task19 perf consolidation).

## Remaining Items (active mainline)
- **task19 (AC-6, next mainline)** — consolidated perf guardrails at conc-1/16 (TTFT,
  decode TPS/req, GPU mem, graph-replay, admission) + Tier-1 non-regression + the
  DS-vs-DSA recall/perf report (the lifted graph-mode 14.5 tok/s / mem note feeds it).
- **task20 (AC-2)** — final strategic-gate supersession decision record.
- Open follow-on (documented, non-blocking): a standalone graph-captured 8-rank TP=8
  selector-equality artifact (the live TP=8 graph serving + eager equality + graph-safe
  selection evidence it); the bf16-store lifted branch is out of the fp8 AC-4 scope.
- Evidence-hygiene queued: R8 oracle-sink provenance; plan-marker cleanup (pre-existing).

## BitLesson Delta
- Action: update
- Lesson ID(s): BL-20260602-flash-mla-sparse-fwd-compact-decode-contract
- Notes: added the **production-wiring corollary** — allocate the fixed scratch once in
  the DS graph-state dataclass (only when opt-in), resolve it from the backend's own
  `self.forward_metadata` with a `getattr` eager fallback; q head-padding is alloc-free
  because the pad tail stays 0 from allocation and is never written (heads independent +
  output trimmed); bound the `[bs*width]` compact_kv footprint with `--cuda-graph-max-bs`;
  and VALIDATE with a wired-backend zero-alloc replay AND a live `cuda graph: True` boot
  (the host-syncing per-request meta is `None` under graph — its absence confirms capture).

## Goal Tracker
Updated directly (Plan Version 23): R17 row; task16 + task17 → Completed and Verified;
**AC-4 MET (production-ready)**; Active = task19, task20. No Goal Tracker Update Request needed.
