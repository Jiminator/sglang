# Round 18 Summary

Mainline: **AC-2.4 NIAH recall-oracle@2048 corroboration** for the production DS scorer (Codex required-plan
item #1). Diagnostic/guarded instrumentation only; no selection/adapter fix.

## Feasibility / approach
The recall-oracle is fully wired: the config-borne `recall_oracle` flag latches the cross-process sink on
the TP workers (`oracle_artifact_sink.py`), and the selector hook records the needle's score rank +
`recall_at_k` per (request,trial,layer,step) (`selection_kernel.py:1210` → `selection_recall_oracle.py`).
The loop7 driver is unusable (its `test/manual` helper was deleted + it targets DeepSeek-V3.2), so I wrote
a self-contained GLM driver. The design is FAIL-CLOSED and self-verifying: a wrong needle span makes the
server emit a `span_out_of_range` hard-failure marker, so an incorrect offline tokenization fails LOUD, not
silent.

## Work Completed
1. **`serve.sh ds_recall_oracle`** — production DS config + `recall_oracle:true`, eager (the host-side
   oracle record is illegal under graph capture). Also added `ds_reduce_fp32` + `ds_recall_oracle` to the
   mode-error string (R17-review queued nit).
2. **`niah_recall_oracle.py`** (new, self-contained) — GLM tokenizer; filler + unique magic-number needle
   near the middle + a recall question; needle span via raw-prompt offset mapping (`add_special_tokens=
   False`). An **alignment probe** measures the server-vs-offline token delta (a BOS prefix would shift all
   logical positions) on a representative prompt of each regime and asserts it is one consistent small
   offset, then shifts every span by it — **measured delta = 0** (GLM adds no BOS, so the offline span
   matches the server KV domain exactly). Per trial: `set_active_trial` → `/generate` (`ignore_eos`, a few
   decode steps) → `clear_active_trial`. After the sweep it reads the sink, **fails closed** on any missing
   trial record or `span_out_of_range`/`exception` marker, and reduces to per-regime recall@2048.
3. **GPU** — one TP=8 server (`ds_recall_oracle`, eager), launched with cwd=`evidence/` so the TP worker's
   oracle-dir default (`cwd/.sglang_ds_oracle`) matches the driver's `--oracle-dir` (env does NOT reach TP
   workers). 8 dense + 8 sparse trials. Torn down to 0 MiB.
4. **`build_ledger.py`** — `validate_recall_oracle_artifact()` fail-closes (both regimes present, non-zero
   records, `corroboration_only` label) before recording the per-regime summary in `run_meta.json`.
   `findings.md` + the evidence-table footer record the result.

## Result (`evidence/ac2_4_recall_oracle.json`, CORROBORATION ONLY — not exoneration)
Fail-closed checks passed: 8/8 trials produced records in BOTH regimes, 0 `span_out_of_range`, 0
`exception`, token delta 0.

| regime | prompt tok | recall@2048 | needle_worst_rank (min/median/max) | selected_contains_needle |
|---|---|---|---|---|
| dense  | ~1136 (< top_k) | **1.0**    | 54 / 777 / 1139   | 1.0 |
| sparse | ~4310 (> top_k) | **0.4103** | 72 / **2524** / 4313 | 0.4103 |

- **Dense** selects every token (recall trivially 1.0; needle always kept) → the dense regression is NOT a
  scorer-ranking failure; it is the H3 current-slot exclusion (AC-2.1/AC-4), independently corroborated.
- **Sparse**: the production raw-dot scorer (`scorer_norm=off`) ranks the needle inside the 2048 budget only
  **~41%** of the time (median worst-rank 2524 > 2048), so the needle is pruned out > half the time. This
  corroborates that the sparse collapse is **scorer-driven** (the raw-dot lock) — consistent with the sparse
  0.000 GSM8K and the cosine recovery. `selected_contains_needle_rate == recall@2048` in both regimes (the
  AC-1 oracle invariant holds — internal consistency check).

## Files Changed (committed `4a16c082a`)
- `development/loop13/serve.sh` (+`ds_recall_oracle` mode, mode-error string), `development/loop13/niah_recall_oracle.py`
  (new), `development/loop13/build_ledger.py` (`validate_recall_oracle_artifact` + run_meta wiring),
  `development/loop13/evidence/ac2_4_recall_oracle.json` (new), `development/loop13/evidence/findings.md`
  (AC-2.4 section), `evidence/evidence_table.md` + `evidence/meta/*` (regenerated), `.gitignore`
  (+`.sglang_ds_oracle/`).

## Validation
- CPU suite, explicit args: `ac4_garbage_counters` (production + ref_faithful + ref_cosine),
  `ac2_1_forced_all_assertions`, `ac6_bisection_matrix`, `ac6_corrob_ref_cosine_noinc`,
  `ac6_score_reduce_corrob`, `ac2_2_head_agg`, `ac4_sample_ids`, `verify_ac2_3 .sglang_ds_scorecap_sparse`
  (committed AC-2.3 artifact unchanged), `test_reference_selectors` (5/5) — **all exit 0**.
- `build_ledger.py` → provenance consistent; verified `validate_recall_oracle_artifact()` ABORTS ledger
  generation on an empty-regime artifact, then restored.
- The driver is itself fail-closed (exit 2 on missing record / hard failure / inconsistent token delta).
- One TP=8 server, eager, torn down to 0 MiB. No sink/`.pt`/`.humanize` raw artifacts committed. No
  selection/adapter **fix**.

## Remaining Items (for AC-8 COMPLETE)
- **AC-3.1** captured decode-row materialized fp32 `K_label` selected-index equality (the committed
  `ac3_1_materialized_k.json` is a synthetic CPU proof; the plan wants it on captured rows).
- **AC-4** serial cells (DSA-radix serial, production DS sparse serial, ref_faithful/ref_cosine serial — the
  reference eager serial runs are very slow) + selected-vs-total gaps.
- **AC-8** final root-cause writeup.

## BitLesson Delta
- Action: add
- Lesson ID(s): BL-20260621-niah-recall-oracle-fail-closed-span-self-verify
- Notes: To reuse a model-specific NIAH recall-oracle on a NEW model (DeepSeek-V3.2 driver → GLM-5.1-FP8),
  the server-side oracle (rank/recall computation) is model-agnostic; only the driver (tokenizer + NIAH
  prompt + needle-span) is model-specific. Two correctness hazards and their guards: (1) the needle's
  LOGICAL token span must match the server's KV domain — a constant BOS prefix shift is the usual gap.
  Don't guess: send an ALIGNMENT PROBE (one /generate), compute `delta = server_prompt_tokens -
  offline_tokens`, assert it is one consistent small offset across regimes, and shift every span by it
  (GLM measured delta=0; a model that prepends BOS would be 1). (2) Even with the probe, rely on the
  oracle's FAIL-CLOSED design as the backstop: the server validates each registered span against its actual
  positions and emits a `span_out_of_range` hard-failure marker, and the driver must exit non-zero on ANY
  such marker or any issued-trial-without-a-record — so a misaligned span fails loud rather than silently
  measuring the wrong tokens' rank. Cross-process path agreement matters too: env vars set at server launch
  do NOT reach TP worker subprocesses, so the worker resolves the sink/trial dir from ITS cwd default —
  launch the server with a known cwd and point the driver at that same absolute dir. Builds on
  [[ds-flag-must-be-config-borne-not-env]] and [[ds-control-must-exercise-pruning]].

## Goal Tracker Update Request

### Requested Changes (already applied to the mutable section):
- Plan Version → 20 (Round 18); added a 17-review row + the Round-18 evolution row.
- task4 → done (AC-2.3 R5; **AC-2.4 R18**): recall-oracle@2048 corroboration recorded (dense 1.0, sparse
  0.4103), ledger-wired with a fail-closed presence check.

### Justification:
Codex named AC-2.4 the #1 remaining close-out item. The corroboration was produced on the real GLM production
DS scorer by reusing the proven recall-oracle machinery (no new production code — a config-borne default-off
flag + a self-contained offline driver). The result reinforces the standing verdict from both sides: dense
recall 1.0 corroborates the dense regression is the H3 current-slot exclusion (not scorer ranking), and
sparse recall 0.41 (worst-rank median > 2048) corroborates the scorer-driven sparse collapse. The remaining
close-out (AC-3.1 captured materialized-K, AC-4 serial/selected-vs-total, AC-8) is the active sequence toward
COMPLETE — not deferrals.
