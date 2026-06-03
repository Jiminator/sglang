# Round 12 Review Result

Mainline Progress Verdict: ADVANCED

Round 12 advanced the stated contract. The new
`double_sparsity/lifted_budget.py::build_compact_decode_index` implements the
request-local physical-to-compact remap with pad masking, prefix-sharing
isolation, and within-row duplicate dropping, and the direct GPU smokes prove
`flash_mla_sparse_fwd` can attend a wider-than-2048 compact index width. This
does not complete Loop 7 or AC-4: the served decode branch, remaining safety
tests, production hardening/disposition, final perf consolidation, and final
decision record remain original-plan mainline work.

## Implementation Review

Accepted R12 work:

1. The compact remap is real and matches the Round 12 foundation contract.

   Evidence: `python/sglang/srt/layers/attention/double_sparsity/lifted_budget.py:106`
   builds a `final_valid` mask from `valid_lengths` and pad value, removes
   within-row duplicate physical slots, computes exclusive per-request compact
   bases from post-dedup counts, emits `page_table_1_flattened` from only
   `final_valid` lanes, and sets invalid compact lanes to `-1`. This keeps
   `page_table_1_flattened` pad-free for `dequantize_k_cache_paged` and keeps
   shared physical prefix slots isolated per request by construction.

2. The CPU tests cover the main index-domain hazards in the round contract.

   Evidence: `test/registered/unit/layers/attention/test_lifted_budget_decode.py:25`
   covers request-local bases, no `-1` in the flattened table, prefix sharing,
   duplicate dropping, zero-valid rows, `valid_lengths` shorter than the padded
   width, and input-order preservation.

3. The GPU kernel proof exists and passed locally.

   Evidence: `test_lifted_budget_decode.py:151` drives a 4096-wide compact index
   with 3000 valid rows in request 0 and matches a reference attention; line 185
   exercises the fp8 `quantize_k_cache` -> `dequantize_k_cache_paged` ->
   `flash_mla_sparse_fwd` pipe with prefix sharing and verifies compact rows are
   bit-identical to full dequant gather. On this review runner, CUDA was available
   on H200/sm90 and the full file reported `10 passed`.

4. The opt-in runtime path is still fail-closed.

   Evidence: `selection_kernel.py:457` still returns `False` from
   `ds_lifted_budget_decode_available()`. No existing runtime path can boot into
   the half-wired lifted path, and the default DSA/DS paths remain untouched in
   this commit.

No high-signal code-level bug was found in the Round 12 implementation itself.

Commands run:
- `pytest -q test/registered/unit/layers/attention/test_lifted_budget_decode.py::TestCompactDecodeIndex`
  -> `8 passed`
- `pytest -q test/registered/unit/layers/attention/test_lifted_budget_decode.py`
  -> `10 passed`
- `pytest -q test/registered/unit/layers/attention/test_lifted_budget_decode.py test/registered/unit/layers/attention/test_scorer_variants.py test/registered/unit/layers/attention/test_double_sparsity_unit.py test/registered/unit/layers/attention/test_ds_scorer_tp_determinism.py`
  -> `332 passed, 9 subtests passed`
- `git diff --check d187f59f4^ d187f59f4` -> no whitespace errors
- `ds_lifted_budget_decode_available()` probe -> `False`

## Mainline Gaps

1. **AC-4 task14 is only partially implemented: the index core is done, but the
   served lifted-budget decode branch is still missing.**

   Required implementation plan:
   - Keep `top_k` as the base DSA/indexer budget. Before flipping the seam,
     tighten validation so the lifted path requires `top_k == index_topk` and
     `lifted_budget_top_k > index_topk`; use `lifted_budget_top_k` as the fixed
     padded lifted selection width.
   - Enforce the R12-discovered kernel contract before enablement:
     `lifted_budget_top_k % 128 == 0`, with config/validator tests for reject and
     accept cases.
   - Special-case the backend/dtype validator for the lifted fp8 path: the default
     fp8 path still uses `flashmla_kv`, but lifted fp8 decode must be allowed and
     routed through dequantized compact KV plus `flash_mla_sparse_fwd`.
   - In the opt-in branch, require `--disable-cuda-graph` until task16 lands.
     Allocate the selector/output scratch to `lifted_budget_top_k` for that branch
     and call the existing top-k selection with the lifted width, preserving the
     established deterministic selection contract.
   - Convert the lifted logical selection to physical KV slots, call
     `build_compact_decode_index`, pass only `page_table_1_flattened` into
     `dequantize_k_cache_paged`, and pass `compact_indices` to
     `flash_mla_sparse_fwd` via the sparse backend. No `-1` or pad lane may reach
     dequant.
   - Flip `ds_lifted_budget_decode_available()` to `True` only after the branch is
     wired, eager-gated, and covered by tests. Do not weaken the default
     `flashmla_kv` `dsa_index_topk` assert.

2. **AC-4 task15 remains partial.**

   R12 supplied the CPU remap matrix and two kernel smokes, but the original plan
   still requires served-path correctness and TP equality.

   Required implementation plan:
   - Add served decode tests at 4096 and 8192 lifted widths comparing the wired
     path against a reference sparse attention on deterministic fp8/dequant cases.
   - Include prefix sharing, invalid padding, `valid_lengths`, and duplicate
     physical-slot cases in the served branch, including an interior `-1` produced
     by dedup rather than only suffix padding.
   - Add TP=8 selected-index equality tests for the lifted 4096/8192 path, using
     the same logical production-path shape as the existing scorer TP matrix.

3. **AC-4 task16 and task17 are still pending.**

   Required implementation plan:
   - Add an alloc-free `out=`/scratch variant of `dequantize_k_cache_paged` plus
     fixed-shape q-padding and compact-index scratch for the production path.
   - Prove zero allocation under CUDA-graph replay before allowing lifted-budget
     production graph capture.
   - Write the Tier-2.A landing disposition after served recall evidence exists.
     If hardening is carried forward, the disposition must explicitly record the
     recall evidence, the DSA default remaining untouched, and the research path
     staying gated out of production capture.

4. **AC-6 task19 and AC-2 task20 remain pending.**

   Required implementation plan:
   - After task17, run the existing Loop-7 serve/benchmark tooling at the chosen
     op-point for DS-default, graph-safe DS-hybrid, DSA, and the lifted research
     path if it is still relevant.
   - Record conc-1 and conc-16 TTFT, decode TPS/req, GPU memory, graph replay
     status, admission, radix/cache assumptions, and exact server configs.
   - Write the consolidated DS-vs-DSA recall/perf/non-regression report.
   - Write the final strategic-gate supersession decision record using the final
     M0/R4/R7/R8/R9/R10/R11/R12 evidence plus the AC-4 disposition.

## Blocking Side Issues

None. The newly discovered `%128` kernel width constraint would be blocking if
the lifted backend were enabled today, but the availability seam remains `False`;
therefore it is a required next-wiring condition, not a live side blocker.

## Queued Side Issues

1. Preserve or cite the R8 raw oracle-sink provenance before task20, or cite the
   hardcoded `stride=1` call site plus committed aggregate explicitly.
2. Remove plan/workflow markers and stale variant comments before final merge.
3. Learned/distilled selector work remains out of scope unless explicitly approved
   under DEC-5.

## Goal Alignment Summary

ACs: 6/6 addressed | Forgotten items: 0 | Unjustified deferrals: 0

| AC | Status | Review result |
|----|--------|---------------|
| AC-1 | MET | Prior R8 oracle-off and stride evidence remain accepted. |
| AC-2 | PARTIAL | Recall/MMLU evidence exists; task19 consolidation and task20 final record remain. |
| AC-3 | MET | R9/R10 graph-safe variant coverage remains accepted. |
| AC-4 | PARTIAL / NOT MET | R12 advanced task14 foundation and part of task15, but the served lifted decode branch, TP equality, task16 hardening, and task17 disposition are missing. |
| AC-5 | MET | 64K servability at mem0.7 remains verified. |
| AC-6 | PARTIAL | Final conc-1/16 perf guardrail report remains missing. |

The tracker represents every original plan task in Active, Completed, or the
empty Deferred table. There are no forgotten task IDs. The remaining task14,
task15, task16, task17, task19, and task20 work is active, not justified away.

## Goal Tracker Update Requests

No tracker edit was required. `goal-tracker.md` Plan Version 15 already records
R12 as task14 foundation progress, keeps task14/task15/task16/task17/task19/task20
active, and has no Explicitly Deferred items.

PENDING
