# Round 10 Contract

## Mainline Objective
**Land the Tier-2.A lifted-budget ABI (AC-4 / task13).** Add the opt-in config
fields `enable_lifted_budget_decode: bool` + `lifted_budget_top_k: int` to
`DoubleSparsityConfig`, with validator gating that **rejects `top_k > index_topk`
UNLESS `enable_lifted_budget_decode` is set** (and the opt-in backend path is the
mechanism) — explicitly NOT reusing `max_top_k` / Twilight fields /
`SGLANG_DS_ALLOW_TOPK_MISMATCH`; the default DSA `dsa_index_topk` assert is
untouched when the flag is off. Produce the **task13 design/disposition record**
(the exact ABI + the physical→`page_table_1_flattened`→compact-remap + padding-
safety + R23-tie-break design for the decode path, reviewed via `ask-codex`),
which the subsequent decode-path rounds (task14–17) build against. This is the
AC-4 foundation Codex's gap requires "first."

## Target ACs (1–2)
- **AC-4** (primary): the opt-in lifted-budget ABI + validator gating + design
  record (the foundation; the decode path/kernel/tests are task14–17).

## Blocking Side Issues In Scope
- None. This round adds opt-in config + validation + a design doc; default-off is
  byte-identical and the DSA assert is untouched.

## Queued Side Issues Out Of Scope (justified)
- **task14 lifted-budget decode path** (`flash_mla_sparse_fwd` + `dequantize_k_cache_paged`
  compact remap, eager research path), **task15 kernel/safety tests**, **task16
  production hardening**, **task17 landing disposition**: the decode kernel is the
  next rounds' work — this round lays the ABI it plugs into.
- **AC-6 perf consolidation (task19) + final strategic-gate decision record
  (task20)**: end milestone.
- **Plan-marker cleanup; R8 oracle-sink provenance note**: pre-merge hygiene.

## Bundled cheap hardening (Codex R9 queued #1 + claim-correction #4)
- Clamp `_force_include_anchor`'s temporary work shape to
  `A = min(anchor_budget, top_k, max_seq)` (the effective budget can never exceed
  the selected count, so this is bit-identical but bounds a pathological opt-in
  `anchor_budget` from over-allocating). Add an **over-budget (anchor_budget >
  top_k) GPU eager-vs-graph case** so the m6 over-budget-coverage claim is backed
  by a graph test (or trim the claim).

## Round Success Criteria
- `enable_lifted_budget_decode` + `lifted_budget_top_k` in `DoubleSparsityConfig`
  (`_ALLOWED_FIELDS` + dataclass + `__post_init__` validation + `parse_double_sparsity_config`);
  default off ⇒ byte-identical config behavior.
- Validator: `top_k > index_topk` is rejected UNLESS `enable_lifted_budget_decode`
  is set; `lifted_budget_top_k` is validated (e.g. > index_topk, fixed); the old
  `SGLANG_DS_ALLOW_TOPK_MISMATCH` is NOT the lifted-budget mechanism (a clear error
  steers to the new ABI). The default-off path leaves the existing
  `top_k == index_topk` assert + `SGLANG_DS_ALLOW_TOPK_MISMATCH` ablation behavior
  unchanged.
- A **task13 design/disposition record** (`development/loop7/`): the ABI, the
  compact-remap (physical slot → `page_table_1_flattened` → request-local compact
  KV index) + `-1`/pad masking + fixed-budget+padding + R23 tie-break plan, and the
  production-hardening disposition framing (land vs deferred-with-evidence). The
  design is reviewed via `/humanize:ask-codex` and the review integrated.
- Unit tests: ABI parse/validation (accept opt-in lifted budget; reject
  `top_k > index_topk` without the flag; reject the Twilight/max_top_k/env
  mechanisms). Anchor temp-shape clamp + over-budget graph case. All DS unit tests
  pass. Committed + pushed; goal-tracker + round-10-summary updated.
