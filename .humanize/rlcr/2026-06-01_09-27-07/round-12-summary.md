# Round 12 Summary — Loop 7

## Mainline objective (round-12-contract.md)
**task14 (foundation) — implement & validate the lifted-budget decode *index core*:**
the request-local physical→compact remap (padding-safety + within-row dedup +
prefix-sharing isolation), plus the direct `flash_mla_sparse_fwd` wider-than-2048
kernel proof.

## Outcome: ACHIEVED — index core landed + CPU-tested; kernel half proven on GPU.

## Context (why this is the right first chunk of task14)
The fp8 DS decode uses `flashmla_kv` (asserts `indices.shape[-1] == dsa_index_topk
== 2048`). The lifted path must instead use `flash_mla_sparse_fwd` (no cap) — which
is the **bf16** backend, so the fp8 KV is dequantized via `dequantize_k_cache_paged`
into a **compact** `[total_valid,1,576]` buffer, and the kernel attends it by
**request-local compact ordinals**, not physical slots. That compact remap is the
trap-laden correctness core (request-local spans, prefix sharing, `-1` masking,
within-row dedup) the m7 design + every Codex review flag — and it is pure tensor
logic, fully CPU-testable now. This round lands it + proves the kernel half.

## Work Completed (`coding`, Claude)
1. **`double_sparsity/lifted_budget.py::build_compact_decode_index`** — pure-tensor,
   deterministic. Given per-request selected physical slots (selector order, fixed
   padded width) + `valid_lengths`, it emits:
   - `page_table_1_flattened` — valid physical slots only, batch-major/selection-rank
     order, **never `-1`** (it is the literal input `dequantize_k_cache_paged` loads);
   - `compact_indices` — request-local ordinals `request_base + rank` for valid lanes,
     `-1` for pad lanes (the kernel masks `<0`/`>=s_kv`);
   - within-row **dedup keeps the highest selection rank** (stable value-sort,
     first-of-run) and counts drops; **prefix-sharing isolated** to per-request spans;
     selector order preserved.
2. **CPU unit tests** (`test_lifted_budget_decode.py::TestCompactDecodeIndex`, 8):
   request-local mapping, prefix sharing (shared slot → distinct per-request spans),
   no `-1` in flattened table, within-row dedup keep-first (+ keep-highest-rank when
   the first repeats later), zero-valid-row base accounting, `valid_lengths` prefix,
   order preservation.
3. **GPU kernel smokes** (`TestLiftedBudgetKernelSmoke`, 2, H200/sm90):
   - `flash_mla_sparse_fwd` attends a request selecting **3000 > 2048** rows inside a
     4096-wide padded budget and matches a reference attention — **no-cap proof**, plus
     `-1` pad masking + request-local spans;
   - full **fp8 → `dequantize_k_cache_paged` → `flash_mla_sparse_fwd`** pipe with
     prefix-sharing matches a reference attending the dequantized selected slots, and
     the compact rows are **bit-identical** to the full-dequant gather.
4. **Discovered + recorded a kernel ABI constraint** (a `width=8` smoke hit
   `Assertion params.topk % (2*B_TOPK) == 0`): the padded index width
   (`lifted_budget_top_k`) must be a **multiple of 128**; the kernel masks indices
   `<0`/`>=s_kv` (so `-1` pad lanes suffice). Captured in `m7_lifted_budget_design.md`
   + a new BitLesson; the next-round wiring must enforce `lifted_budget_top_k % 128 == 0`.
5. `m7_lifted_budget_design.md` updated (landed core + kernel proof + the confirmed
   contract; resolved the "`flash_mla_sparse_fwd` accuracy >512 unproven" open risk).

## Files Changed
- `python/sglang/srt/layers/attention/double_sparsity/lifted_budget.py` (new module).
- `test/registered/unit/layers/attention/test_lifted_budget_decode.py` (new: 8 CPU + 2 GPU).
- `development/loop7/m7_lifted_budget_design.md` (landed core + kernel contract + risks).
- Commit `d187f59f4` (local — loop hook keeps commits local until completion).

## Validation
- `TestCompactDecodeIndex` → **8 passed** (CPU).
- `TestLiftedBudgetKernelSmoke` → **2 passed** (GPU, H200/sm90).
- Full DS unit suite (`test_lifted_budget_decode`, `test_scorer_variants`,
  `test_double_sparsity_unit`, `test_ds_scorer_tp_determinism`) → **332 passed +
  9 subtests** (was 322+9; +8 CPU remap +2 GPU smoke), no regressions.
- No existing runtime path changed; `ds_lifted_budget_decode_available()` stays
  `False` (no half-wired path can boot); default DSA/DS-hybrid/oracle untouched.

## Remaining Items (active mainline, NOT queued-out)
- **task14 (wiring, next mainline)** — widen the selector budget
  `max_top_k`→`lifted_budget_top_k` for the opt-in eager path; route the opt-in
  decode to `flashmla_sparse` via `dequantize_k_cache_paged` feeding this round's
  remap; flip `ds_lifted_budget_decode_available()` to `True` **gated eager-only**
  (validator still requires `--disable-cuda-graph`); enforce `lifted_budget_top_k %
  128 == 0`; preserve the R23 tie-break.
- **task15 (remaining)** — served correctness + TP=8 selected-index equality at 4096/8192.
- **task16** — production hardening (alloc-free `out=` dequant + CUDA-graph), gated behind the recall win.
- **task17** — Tier-2.A landing disposition record.
- **task19 / task20** — AC-6 perf consolidation + final strategic-gate decision record.
- Evidence-hygiene queued: R8 oracle-sink provenance; plan-marker cleanup (pre-existing).

## AC-4 status
**task14 advanced** (index core + kernel proof). **AC-4 NOT MET** — the decode-branch
wiring, served recall evidence, TP=8 equality, and task16/17 remain.

## BitLesson Delta
- Action: add
- Lesson ID(s): BL-20260602-flash-mla-sparse-fwd-compact-decode-contract
- Notes: records the fp8 lifted-budget decode contract — dequant→compact buffer +
  request-local ordinal indices (not physical), `-1`/`>=s_kv` masking, within-row
  dedup, prefix-sharing isolation, and the `lifted_budget_top_k % 128 == 0`
  (`topk % (2*B_TOPK)`) kernel-block requirement — so the next-round wiring and any
  future `flash_mla_sparse_fwd` consumer honor it. Reusable, non-obvious, directly
  load-bearing for the upcoming decode-branch wiring.

## Goal Tracker
Updated directly (Plan Version 15): R12 Plan Evolution row added; task14 → "in
progress (index core + kernel proof done R12)" with the remaining wiring listed;
task15 → "partial (kernel smokes done R12)". No Goal Tracker Update Request needed.
