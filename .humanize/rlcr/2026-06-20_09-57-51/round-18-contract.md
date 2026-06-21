# Round 18 Contract

Round 17 was ADVANCED, no new blocker. AC-4 garbage counters are complete for all primary served DS arms.
Loop still NOT_COMPLETE on: AC-2.4 recall-oracle, AC-3.1 captured materialized-K, AC-4 serial cells +
selected-vs-total, AC-8 writeup. This round takes Codex's required-plan item #1, the most self-contained
named close-out artifact that reuses existing (proven) machinery: **AC-2.4 NIAH recall-oracle**.

Feasibility confirmed: the recall-oracle is fully wired — config-borne `recall_oracle` flag latches the
sink on TP workers (oracle_artifact_sink.py), the selector hook records the needle's score rank +
`recall_at_k` per (request,trial,layer,step) (selection_kernel.py:1210 → selection_recall_oracle.py
`oracle_payload_for_row`), and the design is FAIL-CLOSED: a wrong needle span emits a `span_out_of_range`
hard-failure marker, so an incorrect offline tokenization produces a LOUD failure, not a silently-wrong
artifact. The loop7 driver (`niah_oracle_sweep.py`) is unusable as-is (its `test/manual` helper was deleted
+ it targets DeepSeek-V3.2), but its model-agnostic helpers (`needle_logical_span`, `_generate_decode`,
`_read_sink`) are the template; only the NIAH prompt builder must be rewritten for GLM.

## Mainline Objective (exactly one)
**Produce the AC-2.4 NIAH recall-oracle@2048 corroboration artifact for the production DS scorer, dense AND
sparse**: a guarded `serve.sh` mode (production DS + `recall_oracle:true`, eager) + a self-contained GLM
NIAH driver/reducer (`development/loop13/niah_recall_oracle.py`) that registers needle spans, drives
single-request trials at dense (<2048 tok) and sparse (>2048 tok) context lengths, and reduces the
fail-closed sink into `evidence/ac2_4_recall_oracle.json` with per-regime recall@2048 — labelled
CORROBORATION ONLY (NOT scorer exoneration), wired into the ledger with a fail-closed presence check.

## Target ACs
- **AC-2.4** (primary): `recall_oracle` recall@2048 recorded for dense and sparse as corroborating evidence
  (explicitly not a generic selected-index equivalence proof).

## Blocking Side Issues (these ARE the mainline)
- The loop7 NIAH driver depends on a deleted `test/manual/test_double_sparsity_v32.py` helper and the
  DeepSeek-V3.2 tokenizer. A self-contained GLM driver (GLM tokenizer from the model dir; raw-prompt offset
  mapping, no chat template) is required. The server-side oracle (rank computation) is model-agnostic and
  reused unchanged.

## Queued Side Issues (documented, OUT OF SCOPE this round)
- AC-3.1 captured decode-row materialized fp32 `K_label` selected-index equality (needs latent-VALUE +
  query capture extension).
- AC-4 serial cells (DSA-radix serial, production DS sparse serial, ref_faithful/ref_cosine serial — the
  reference eager serial runs are very slow) + selected-vs-total gaps.
- AC-8 final root-cause writeup.
- `serve.sh` mode-error text omits `ds_reduce_fp32` (R17-review queued); `ac4_garbage_counters.py --arm
  <non-production>` defaults to the production capture dir if CAPDIR omitted (ledger catches it; reuse
  hardening). Plan-term comment cleanup.

## Approach
1. `serve.sh ds_recall_oracle`: production `ds` config + `"recall_oracle": true`, eager
   (`--disable-cuda-graph`, required for the host-side oracle records). Capture/trial/sink dir via
   `SGLANG_DS_RECALL_ORACLE_DIR` (default `$EVID/.sglang_ds_oracle`).
2. `development/loop13/niah_recall_oracle.py`: load the GLM tokenizer (`$MODEL/tokenizer.json`); build NIAH
   prompts (benign filler + a unique-magic-number needle at a known offset + a recall question) at a DENSE
   target (<2048 tok) and a SPARSE target (>2048 tok); compute the needle's logical token span via offset
   mapping (`add_special_tokens=False`); per trial `sink.set_active_trial(...)` then POST `/generate`
   (`ignore_eos`, a few decode steps); read the sink, FAIL CLOSED on any missing trial record or hard
   failure marker (`span_out_of_range`/`exception`); reduce to per-regime recall@2048 = mean of
   `recall_at_k["2048"]` over success records, plus `needle_worst_rank` summary. Write
   `evidence/ac2_4_recall_oracle.json` with a `corroboration_only: true` label and the per-regime numbers.
3. `build_ledger.py`: add a fail-closed presence/shape check for the AC-2.4 artifact (dense+sparse sections,
   record counts > 0) and reference it in the footer/NOT-instrumented surfaces. `.gitignore` the oracle dir.

## Concrete Success Criteria
1. `serve.sh ds_recall_oracle` exists (production ds + recall_oracle, eager). One TP=8 server, torn down to
   0 MiB. No `.pt`/sink/`.humanize` raw artifacts committed.
2. `evidence/ac2_4_recall_oracle.json` records, per regime (dense + sparse), recall@2048 over a non-zero
   number of fail-closed-verified oracle records (every issued trial produced a record; zero hard-failure
   markers), with an explicit corroboration-only label. The driver exits non-zero on any missing record /
   hard failure (verified the fail-closed path).
3. `build_ledger.py` references the artifact and fail-closes if it is absent/empty when AC-2.4 is rendered;
   `findings.md` records the recall-oracle corroboration result.
4. Tests pass; provenance consistent. Commit; round-18-summary with BitLesson Delta + Goal Tracker Update
   Request. No selection/adapter FIX (diagnostic/guarded instrumentation only). No exit by lying / editing
   loop state / cancel-rlcr-loop.
