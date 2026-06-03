# Round 12 Contract — Loop 7

## Mainline objective (EXACTLY ONE)
**task14 (foundation) — implement & validate the lifted-budget decode *index core*:
the request-local physical→compact remap with padding-safety and within-row dedup,
plus the direct `flash_mla_sparse_fwd` wider-than-2048 kernel proof.**

Concretely, land a standalone, deterministic module that, given per-request selected
**physical** KV slots + `valid_lengths` (a fixed padded `lifted_budget_top_k` width),
produces:
- `page_table_1_flattened` — the concatenated VALID physical slots across the batch
  (the input `dequantize_k_cache_paged` consumes), with **no `-1`/pad** ever reaching it;
- request-local **compact-domain** indices (`request_base + selected_rank`, NOT physical
  slot values, NOT a global lookup) for `flash_mla_sparse_fwd`, with pad lanes masked to a
  safe sentinel;
- **within-row dedup** so a query row never sees a duplicate valid compact index
  (`flash_mla_sparse_fwd` would double-attend);
- **prefix-sharing safety**: the same physical slot appearing in two requests' selections
  maps to each request's OWN compact span (request-local base).

This is the correctness heart of the opt-in decode path — pure tensor logic, fully
CPU-unit-testable now — and the foundation the decode-branch wiring (next round) plugs into.
Paired with it: the **Codex-required direct `flash_mla_sparse_fwd` 4K-topk smoke/accuracy
test** (run on GPU if available) proving the kernel attends a >2048-wide (4096) compact
buffer and matches a reference sparse-attention within tolerance.

## Target AC(s)
- **AC-4** — the opt-in adjustable-budget decode path. This round lands its index core +
  kernel proof. AC-4 stays NOT MET (the served decode branch + recall evidence = next rounds).

## Blocking issues (truly block the mainline)
- **None.** The module is new and self-contained; it changes no existing runtime path. The
  R11 fail-closed seam `ds_lifted_budget_decode_available()` stays **`False`** this round, so
  no server can boot a half-wired path. Default DSA/DS-hybrid/oracle paths are untouched.

## Queued — explicitly OUT of scope this round (NOT closed/deferred; sequenced next)
- **task14 (wiring, next mainline)** — widen the selector budget to `lifted_budget_top_k`
  for the opt-in eager path; switch the opt-in decode to `flashmla_sparse` via
  `dequantize_k_cache_paged` (fp8→compact bf16) feeding this round's remap; flip
  `ds_lifted_budget_decode_available()` to `True` **gated eager-only** (validator still
  requires `--disable-cuda-graph` for it); R23 tie-break preserved.
- **task15 (remaining)** — served correctness/TP=8-equality at 4096/8192; reference-attention
  tolerance on the full served path.
- **task16** — production hardening (alloc-free `out=` dequant + CUDA-graph), gated behind the recall win.
- **task17** — Tier-2.A landing disposition record.
- **task19 / task20** — AC-6 perf consolidation + final strategic-gate decision record.
- Evidence-hygiene queued: R8 oracle-sink provenance; plan-marker cleanup (pre-existing).

## Concrete success criteria
1. A new module (e.g. `double_sparsity/lifted_budget.py`) exposes a pure, deterministic
   `build_compact_remap(...)` returning `page_table_1_flattened` + compact-domain indices
   + a valid mask, with **no `-1`/pad in `page_table_1_flattened`** and pad lanes masked in
   the compact indices.
2. CPU unit tests cover, each as an explicit case: request-local mapping; prefix-sharing
   (same physical slot in two requests → distinct compact spans); `-1`/pad masking before any
   dequant-domain output; **within-row duplicate** collapsed/asserted-unique; `valid_lengths`
   shorter than the padded width; a row with 0 valid entries.
3. The remap preserves the selector's deterministic ordering (the R23 score-desc/position-asc
   order of the selected positions is carried into the compact ordinals — rank = position in
   the selected list).
4. **Direct `flash_mla_sparse_fwd` 4K-topk smoke** (GPU if available): a tiny deterministic
   case dequantized via `dequantize_k_cache_paged`, attended with a 4096-wide compact index
   tensor, matches a reference einsum-softmax attention over the selected tokens within a
   stated tolerance. If no GPU is reachable, the test is written + skipped-with-reason and the
   proof is recorded as the explicit next-round gate (no silent skip).
5. `ds_lifted_budget_decode_available()` remains `False`; default-off path byte-identical;
   DSA `dsa_index_topk` assert + `SGLANG_DS_ALLOW_TOPK_MISMATCH` untouched.
6. No plan-marker leakage in production code/comments (domain names only).
7. Full DS unit suite passes.
8. `m7_lifted_budget_design.md` + `goal-tracker.md` updated; commit.

## Tag routing
- task14 is a **`coding`** task → Claude executes directly.
