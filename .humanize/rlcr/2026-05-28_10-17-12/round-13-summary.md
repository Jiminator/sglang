# Round 13 Summary

## Context
Codex's Round-12 review converged the loop: **0 active plan tasks, 0 mainline gaps, 0 blocking
issues**; the sole unmet criterion is **AC-12** (a hard fail), which is an inherent DS limit with no
DS-preserving fix. The mechanical COMPLETE condition is therefore unsatisfiable by autonomous code
work, so I **escalated the disposition to the user** (AskUserQuestion). The user directed a close-out:
do the queued cleanups first, then test whether a significantly larger DS selection budget passes
NIAH ("if not, then there is a serious issue"), then document the remaining work for the next loop.

## Work completed (user-directed)

### 1. Queued cleanups
- **Comparator per-side `mem_fraction_static` hole** (Codex queued #1): `_validate_per_side_agreement`
  now requires `mem_fraction_static` **constant within each side** while keeping it ignored
  **across** sides (the sanctioned DSA 0.85 vs DS 0.6 TokenLabelTable asymmetry). New regression
  `test_per_side_mem_fraction_drift_refused` (per-side drift → exit 2; constant-per-side cross-side
  asymmetry → proceeds).
- **Stale `calibrate.py` recipe docstring** (Codex queued #3): corrected to the validated command in
  `calibration_provenance.md` (`--tp 8` + local `--dataset` corpus + `-v`) and made accurate —
  `--dtype` is a recorded forward-stability hint (load uses `torch_dtype="auto"`), `--tp` is recorded
  metadata (`device_map="auto"` shards across visible GPUs; no distributed group spawned).

### 2. NIAH selection-budget investigation — answers the user's question
**The DS selection budget cannot be significantly increased**, and the AC-12 NIAH failure is a
**selection-quality** limit, not a decode bug:

- **Budget is kernel-locked to the model's DSA `index_topk=2048`.** Booting DS with `top_k=8192`
  fails twice (`ac12_topk_sweep/boot_evidence_topk_locked.txt`): the validator refuses
  (`top_k != index_topk`), and with `SGLANG_DS_ALLOW_TOPK_MISMATCH=1` the **`flashmla_kv` decode
  kernel itself** asserts `indices.shape[-1] == self.dsa_index_topk` during CUDA-graph capture. DS
  reuses V3.2's native DSA decode kernel, which consumes exactly 2048 indices — so the budget can't
  be widened on this path. (Architectural constraint, not a bug.)
- **No serious decode bug.** A DS-only NIAH curve at `top_k=2048`
  (`ac12_topk_sweep/ds_recall_vs_length_topk2048.json`): recall **100%** at 1024 words (dense,
  budget ≥ seq), **100%** at 1536 (dense), **75%** at 4K (~50% selected), **5%** at 16K (~12.5%,
  from AC-12). DS recalls perfectly when its selection is complete (= a dense model), and recall
  tracks the selected fraction. Combined with MMLU == DSA (short context), DS decode is sound.
- **The DS-vs-DSA gap at the SAME 2048 budget is selection quality.** DSA recalls 100% at 16K with
  the same kernel + budget because its *trained* DSA indexer places the needle in its 2048; DS's
  *offline channel-mask* heuristic does not. On a model with a native trained sparse indexer, DS is
  capped at the native budget and selects worse, so it cannot match DSA's long-context recall, and a
  larger `top_k` is not an available lever. (`ac12_topk_sweep/analysis.md`.)

### 3. Next-loop issue list
`runs/20260528_dsv32_mvp/next_loop_issues.md`: the AC-12 disposition decision (accept smoke
milestone vs re-scope vs R&D), DS selector R&D (query-aware/learned indexer; a kernel accepting
`top_k > index_topk`), DS KV-budget/TokenLabelTable for 64K admission + AC-11 TTFT, the strategic
"is DS worthwhile on a model with native DSA?" question, and the cosmetic serve-header terms.

## Files changed
- `development/benchmark_compare.py` — per-side `mem_fraction_static` check.
- `test/registered/unit/development/test_ac11_comparator.py` — `test_per_side_mem_fraction_drift_refused`.
- `python/sglang/srt/layers/attention/double_sparsity/calibrate.py` — recipe docstring.
- `runs/20260528_dsv32_mvp/ac12_topk_sweep/` (`analysis.md`, `ds_recall_vs_length_topk2048.json`,
  `boot_evidence_topk_locked.txt`), `runs/20260528_dsv32_mvp/next_loop_issues.md`.
- Commits `ced03f374` (cleanups), `27434cee7` (investigation + docs). Both pushed.

## Validation
- **409 CPU tests pass** (the five-file suite; +1 comparator per-side mem-fraction regression).
- Hardware: confirmed the `top_k`-budget kernel lock (two boot failures with evidence) and ran the
  DS-only NIAH recall curve at `top_k=2048`. DS server shut down; node-0 GPUs freed (0 MiB);
  the pre-existing router was not touched.

## Remaining Items
- **AC-12 remains a recorded HARD FAIL** — now precisely characterized (selection-quality +
  kernel-budget-cap; no DS-preserving fix). The loop's COMPLETE condition stays unsatisfiable; the
  disposition (accept smoke milestone / re-scope / R&D) is the user's carried-over decision,
  documented in `next_loop_issues.md`.
- Queued (still non-blocking): DS selector/KV-budget R&D, AC-11 TTFT follow-up, cosmetic
  serve-header terms — all listed for the next loop.

## BitLesson Delta
- Action: update
- Lesson ID(s): BL-20260529-ds-longcontext-needle-recall-vs-topk
- Notes: Added a Round-13 addendum: the DS selection budget is **kernel-locked** to the model's DSA
  `index_topk=2048` (validator refusal + `flashmla_kv` `indices.shape[-1] == dsa_index_topk`
  assertion), so `top_k` cannot be raised on this backend; a DS-only recall curve proves decode is
  sound (dense ≤2048 → 100%, tracking the selected fraction); and the DS-vs-DSA gap at the same 2048
  budget is **selection quality** (offline channel-mask vs the trained DSA indexer), not budget size
  or a decode bug. This sharpens how to diagnose an AC-12 NIAH failure and bounds what can fix it.

## Goal Tracker Update Request

### Requested Changes:
- Confirm the **comparator per-side `mem_fraction_static`** and **`calibrate.py` recipe docstring**
  queued items RESOLVED (commit `ced03f374`, +1 regression, 409 CPU tests pass).
- Record the **AC-12 selection-budget investigation** outcome: budget kernel-locked to
  `index_topk=2048`; DS decode sound (dense=100%); AC-12 gap is selection quality vs the native DSA
  indexer — no DS-preserving fix and no raisable `top_k` on this backend (commit `27434cee7`).
- Keep **AC-12 as NOT met** (hard fail) and the loop4-compatible MVP incomplete; the disposition is
  the user's carried-over decision (see `next_loop_issues.md`). No immutable AC/threshold changed.

### Justification:
Per the user's Round-13 direction, this round cleared the two queued cleanups and answered the
selection-budget question empirically: DS cannot widen `top_k` (shared DSA decode kernel asserts the
2048 budget), and the AC-12 NIAH gap is a selection-quality limit (DS's offline channel-mask vs
V3.2's trained DSA indexer at the same budget), with DS decode proven sound (dense recall 100%). No
"serious issue" (decode/serving bug) exists — the limitation is architectural and inherent to running
Double Sparsity on a model that already has a superior native sparse indexer. AC-12 was not faked
green and no immutable AC was changed; the disposition is documented for the next loop.
