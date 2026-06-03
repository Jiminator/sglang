# Round 14 Summary — Loop 7

## Mainline objective (round-14-contract.md)
**task15 — the binding *served* recall evidence for the lifted-budget 4096 decode
path**, plus the lifted-width TP=8 determinism + backend-level decode correctness.

## Outcome: ACHIEVED — task15 DONE. The lifted-4096 budget materially recovers 4K served recall.

## Headline result (live, served, N=20)
Both servers EAGER on the same node (so the delta isolates the **budget**, not the
eager-vs-graph numerics gap):

| variant | hits/N | recall | 95% CP CI | admission_fail |
|---|---|---|---|---|
| DS-default top_k=2048 | 15/20 | 75% | [50.9%, 91.34%] | 0 |
| DS-lifted lifted_budget_top_k=4096 | 19/20 | **95%** | [75.1%, 99.9%] | 0 |
| **uplift** | | **+20 pp** | lifted 0.95 > base_hi 0.9134 → **MATERIAL** | |

This confirms the M0 oracle's 4K **budget-limited** attribution **on the served
decode path** (prompt ~4400 tokens → the 4096 budget keeps ~all of it → the needle
at oracle score-rank ~2208 lands inside). Tier-2.A stays **bounded-secondary**: M0
showed 16K budget-partial / 64K scorer-limited, which a wider budget cannot recover
— those are served by the landed Tier-2.B hybrid scorer (AC-3).

## Work Completed (`coding`, Claude)
1. **Live served recall sweep** (the payoff): booted DS-lifted-4096 (eager,
   int8/mem0.7, `--disable-cuda-graph`) and DS-default-2048 (eager, same node),
   ran NIAH 4K N=20 each via `niah_ds_baseline.py`, computed Clopper–Pearson CIs +
   directional materiality. A `/generate` smoke first confirmed the lifted path
   serves coherently with `double_sparsity` meta non-None (`dense_fallback=0`).
2. **Backend-level decode test** (`test_lifted_budget_decode.py::TestLiftedBudgetBackendDecode`):
   drives the actual wired `DeepseekSparseAttnBackend._forward_lifted_budget` (not
   just the helper) at **4096 and 8192** with prefix-sharing, a duplicate physical
   slot, and `valid_lengths < width`, vs an independent reference attention.
3. **Lifted-width TP=8 determinism** (`test_ds_scorer_tp_determinism.py::TestTP8LiftedWidthDeterminism`):
   8 gloo ranks through the production logical selector + all-reduce at
   `max_top_k ∈ {4096, 8192}`, `max_seq_len=8192`; identical `selected_indices` +
   `valid_lengths` across ranks (full-length request selects exactly the lifted width).
4. **Serve knob** (`serve_double_sparsity.sh`): `LIFTED_BUDGET` (+`LIFTED_BUDGET_TOP_K`)
   emits `enable_lifted_budget_decode`/`lifted_budget_top_k` in `DS_CONFIG` and forces
   `--disable-cuda-graph` (mirroring the `RECALL_ORACLE` eager handling).
5. **Matrix tool + finding**: `lifted_recall_matrix.py` (reuses the CP +
   directional-materiality methodology), `m8_lifted_recall_finding.md` (full
   provenance: commit, GPU, server args, DS configs, admission, artifacts).

## Files Changed
- `development/serve_double_sparsity.sh` (LIFTED_BUDGET knob).
- `test_lifted_budget_decode.py` (backend-level decode test), `test_ds_scorer_tp_determinism.py`
  (lifted-width TP=8).
- `development/loop7/`: `lifted_recall_matrix.py` (new), `m8_lifted_recall_finding.md`
  (new), `niah_ds_lifted4096.json` (new), `niah_ds_default2048_eager.json` (new),
  `ds_lifted_vs_default_recall_4k.json` (new), `m7_lifted_budget_design.md` (open items
  resolved).
- Commit `0ad20774a` (local — loop hook keeps commits local until completion).

## Validation
- Backend decode test → **2 passed** (GPU); lifted-width TP=8 → **2 passed** (8-rank gloo).
- Full DS unit suite (4 files) → **341 passed + 9 subtests** (was 337; +4 R14 tests).
- Live sweep: lifted 19/20, default 15/20, served 20/20 + 0 admission fails both;
  reproducible from the committed JSONs via `lifted_recall_matrix.py`.

## Provenance (Codex-required)
Commit `2ba4dafc1` (R13 wiring) + the R14 serve knob; 8× NVIDIA H200 (sm90), TP=8;
op-point int8 / mem 0.7 / page 64 / fp8-KV / flashmla_kv / radix-off / eager; N=20;
0 admission failures. Artifacts listed above.

## Remaining Items (active mainline, NOT queued-out)
- **task16 / task17** — the landed path is eager-required (the dequant allocates).
  AC-4 remains pending the task16 production-hardening **decision** (alloc-free `out=`
  dequant + CUDA-graph) + the task17 Tier-2.A landing disposition, which — per
  DEC-4/DEC-6 — may record this eager research recall + carry hardening to a follow-on
  with the DSA default untouched (the "deferred-with-evidence" close).
- **task19 / task20** — AC-6 perf consolidation + final strategic-gate decision record.
- Evidence-hygiene queued: R8 oracle-sink provenance; plan-marker cleanup (pre-existing).

## AC-4 status
task14 (wired) + task15 (served recall recovery) **DONE**. AC-4 remains NOT MET only
on the task16 hardening decision + the task17 landing disposition.

## BitLesson Delta
- Action: update
- Lesson ID(s): BL-20260602-eager-vs-graph-recall-differs-despite-identical-scorer
- Notes: added the **eager-only-variant corollary** — when the variant under test is
  itself eager-only (the lifted-budget dequant is not graph-safe), re-measure the
  baseline EAGER on the SAME node so the delta isolates the variable (the budget)
  rather than confounding it with the eager-vs-graph numerics gap; the eager number
  is still not the production-graph number. Backed by the R14 both-eager lifted-vs-default
  4K comparison.

## Goal Tracker
Updated directly (Plan Version 18): R14 Plan Evolution row added; task15 → **done**.
No Goal Tracker Update Request needed.
