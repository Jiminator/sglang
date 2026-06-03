# Round 18 Contract — Loop 7

## Mainline objective (EXACTLY ONE)
**Complete AC-4: land the graph-captured TP=8 lifted-width selector-equality
artifact (the last task16 evidence item) AND make `m9_tier2a_disposition.md` fully
consistent with the R17 production graph state (task17), re-reviewed.**

The R17 review reactivated task16/task17 for two precise reasons:
1. The graph-captured **TP=8 lifted-width selector** equality is unproven — the R17
   backend test uses preselected physical slots and bypasses the selector/all-reduce.
2. `m9` still has stale **"eager-required / validator rejects unless
   `--disable-cuda-graph` / launcher forces eager / only deferred item is hardening"**
   prose that contradicts the R17 landing.

## Target AC(s)
- **AC-4** (closes it: the production-ready landing with complete graph evidence) +
  **AC-3** (its TP=8 cross-rank equality requirement, now at lifted width under capture).

## Blocking issues (truly block the mainline)
- **None.** The lifted path is opt-in/default-off; the new test + the doc rewrite do
  not change the default runtime path.

## Queued — explicitly OUT of scope this round (NOT closed/deferred)
- **task19 (AC-6)** — final perf consolidation conc-1/16. Next mainline.
- **task20 (AC-2)** — final strategic-gate supersession decision record.
- Evidence-hygiene queued: R8 oracle-sink provenance; plan-marker cleanup (pre-existing).
- **Bundled defensive fix (queued, cheap)**: the R17-review lifted+speculative scratch
  sizing hazard — add a fail-closed validator guard (reject
  `enable_lifted_budget_decode` + speculative decode) since the lifted scratch is
  sized by `max_bs` while target-verify rows expand to `bs*num_draft_tokens`. Loop-7
  is non-speculative; this prevents a future undersized-scratch crash. Small, so
  bundled — but it does NOT replace the mainline.

## Concrete success criteria
1. **Graph-captured TP=8 lifted-width selector equality (GPU, 8 ranks)**: an 8-rank
   NCCL process group (one GPU each) captures the real graph-safe selector
   (`retrieve_topk_graph_safe`, including the all-reduce) at lifted widths **4096 and
   8192**, replays it, asserts **zero-alloc** (`assert_no_alloc_in_region`), and
   asserts **identical `selected_indices` + `valid_lengths` across all 8 ranks** +
   equal to the eager/logical reference. (The live R17 TP=8 graph server already
   served correct recall, evidencing the all-reduce captures; this is the explicit
   standalone artifact.)
2. **`m9` full consistency**: every section describes the R17 production state —
   validator relaxed (no `--disable-cuda-graph` requirement), launcher no longer
   forces eager, `dequantize_k_cache_paged_out` + fixed scratch is the production
   path, the graph-mode 4K NIAH 95% is the binding production recall. Keep the
   bounded-secondary 4K-only analysis; **remove every "production graph hardening is
   deferred / eager-required" claim**. The graph-captured TP=8 artifact (1) is cited
   as recorded, not a follow-on.
3. **`m9` re-reviewed via `/humanize:ask-codex`** (the `analyze` step); feedback integrated.
4. **Bundled**: a fail-closed validator guard rejecting lifted + speculative decode,
   with a unit test.
5. **Non-regression**: default-off byte-identical; full DS unit suite passes; no new
   plan-marker leakage.
6. `goal-tracker.md` updated (AC-4 MET production-ready; task16/task17 done); commit.

## Tag routing
- The TP=8 test + validator guard are **`coding`** → Claude. The `m9` disposition is
  **`analyze`** → draft + `/humanize:ask-codex` review.
