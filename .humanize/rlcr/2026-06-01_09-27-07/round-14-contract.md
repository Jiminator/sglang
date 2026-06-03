# Round 14 Contract — Loop 7

## Mainline objective (EXACTLY ONE)
**task15 — produce the binding *served* recall evidence for the lifted-budget
4096 decode path, plus the lifted-width TP=8 determinism + backend-level decode
correctness Codex requires.**

The centerpiece is the live served NIAH 4K recall-recovery sweep: boot the DS
server eager (`--disable-cuda-graph`) with `enable_lifted_budget_decode=true`,
`lifted_budget_top_k=4096`, `top_k=index_topk=2048`, run NIAH 4K at N≥20, and
compare the lifted-4096 served recall to the DS-default-2048 served recall on the
same node with exact Clopper–Pearson CIs — stating whether the uplift exceeds the
baseline CI. The M0 oracle predicted 4K is budget-limited (score-only
recall@2048≈44% → recall@4096≈100%); this measures whether the *served* lifted
path recovers it. A recorded, characterized result (uplift OR a null/served-vs-
oracle gap) closes task15's recall question per DEC-2.

## Target AC(s)
- **AC-4** (the lifted-budget decode recall evidence) and **AC-2** (recall uplift
  measured DS-vs-DS same node, CI-judged).

## Blocking issues (truly block the mainline)
- **None known.** The serve script must emit the lifted knobs (a success
  criterion, not a blocker). If the live sweep surfaces a runtime bug in the wired
  lifted path, fixing it is in-scope mainline (it blocks the served evidence).

## Queued — explicitly OUT of scope this round (NOT closed/deferred)
- **task16** — production hardening (alloc-free `out=` dequant + CUDA-graph),
  pursued only if the recall win justifies; the path stays eager-required until then.
- **task17** — Tier-2.A landing disposition record (after the recall evidence + task16
  decision).
- **task19 / task20** — AC-6 perf consolidation + final strategic-gate decision record.
- Evidence-hygiene queued: R8 oracle-sink provenance; plan-marker cleanup (pre-existing).

## Concrete success criteria
1. **Backend-level decode test** (GPU): drive the actual wired method
   `DeepseekSparseAttnBackend._forward_lifted_budget` (not only
   `build_lifted_compact_kv`) on a minimally-constructed backend with deterministic
   fp8 KV, prefix sharing, a duplicate physical slot, and `valid_lengths < width`,
   at widths **4096 and 8192**, matching a reference attention.
2. **Lifted-width TP=8 determinism**: extend `test_ds_scorer_tp_determinism.py`
   with a lifted case for `max_top_k ∈ {4096, 8192}` and `max_seq_len ≥ max_top_k`
   through the logical production selector path with the 8-rank all-reduce;
   assert identical `selected_indices` + `valid_lengths` across all ranks.
3. **Serve knob**: `serve_double_sparsity.sh` emits `enable_lifted_budget_decode`
   + `lifted_budget_top_k` in `DS_CONFIG` and forces `--disable-cuda-graph` for the
   lifted path (mirroring the `RECALL_ORACLE` eager handling).
4. **Live served recall sweep**: NIAH 4K, N≥20, on the same node, for
   **DS-default top_k=2048** vs **DS-lifted lifted_budget_top_k=4096** (eager, int8,
   mem 0.7), with served-vs-admission separated and Clopper–Pearson 95% CIs;
   directional materiality (lifted point vs the default CI high). Recorded to a
   finding doc + JSON artifacts with server args, DS config, commit, GPU type,
   trial count, and admission status. **If the lifted path does not recover 4K**
   served recall, that null/characterized result is recorded honestly (served vs
   score-only-oracle gap noted) — it still closes the recall question.
5. **Non-regression**: default-off path byte-identical; full DS unit suite passes;
   no new plan-marker leakage in production code.
6. `m7`/a new `m8` finding + `goal-tracker.md` updated; commit.

## Tag routing
- task15 is a **`coding`** task → Claude executes directly.
