# Round 19 Contract

Round 18 was ADVANCED but Codex rejected AC-2.4 closure: the recall numbers are coherent (dense 1.0, sparse
0.4103, 8/8 trials, zero failure markers) but the **fail-closed artifact contract is incomplete** — the
producer writes the canonical JSON before validating, `no_active_trial` markers are not treated as fatal,
the ledger's checks are too weak, and `serve.sh ds_recall_oracle` does not enforce the oracle-dir/CWD
agreement the worker depends on. Same evidence-integrity class as the R15 regression.

The R18 sink (`evidence/.sglang_ds_oracle/sink.jsonl`, 4992 records) is INTACT and the run was clean. I will
RE-RUN the sweep with the hardened driver (one GPU boot) rather than re-reduce offline: only a live sweep
can genuinely populate the issued-vs-recorded tracking AND the non-null server prompt-token samples that
Codex's hardened ledger check requires — a reduce-from-old-sink can't supply prompt_tokens (the sink records
only rank/recall) and would reconstruct `issued`. The re-run also proves the fixed serve harness works from
the repo root.

## Mainline Objective (exactly one)
**Repair the AC-2.4 recall-oracle fail-closed contract end-to-end (producer + consumer + serve harness)** so
a partial / failure-marker / wrong-source run can never leave a canonical artifact the ledger accepts, then
regenerate the committed artifact from the verified R18 sink and prove the guards with negative tests.

## Target ACs
- **AC-2.4** (primary): a fail-closed, provenance-checked recall-oracle@2048 corroboration artifact +
  reproducible serve harness.

## Blocking Side Issues (these ARE the mainline — the two R18-review blockers)
- `niah_recall_oracle.py` writes `ac2_4_recall_oracle.json` BEFORE the `problems` check; only
  `span_out_of_range`/`exception` are fatal (not `no_active_trial` or future markers); no count invariants.
- `build_ledger.py` `validate_recall_oracle_artifact()` checks only arm/label/regime-presence/non-zero
  records — not failure markers, issued-vs-recorded counts, recall-record parity, or source dir.
- `serve.sh ds_recall_oracle` documents that the worker resolves the oracle dir from server CWD but launches
  in the caller's CWD, so a normal launch splits the trial/sink paths.

## Queued Side Issues (documented, OUT OF SCOPE this round)
- AC-3.1 captured decode-row materialized fp32 `K_label` selected-index equality (synthetic proof insufficient).
- AC-4 serial cells (DSA-radix serial, production DS sparse serial, ref_faithful/ref_cosine serial) +
  selected-vs-total gaps.
- AC-8 final root-cause writeup.
- `ac4_garbage_counters.py --arm <non-production>` defaults to the production capture dir if CAPDIR omitted
  (ledger catches it); plan-term comment cleanup.

## Approach
1. **Producer (`niah_recall_oracle.py`)**: factor the reduce+validate into one function that builds the
   report IN MEMORY, runs ALL checks, and writes to `ac2_4_recall_oracle.json.tmp` then atomically renames
   to the canonical path ONLY if zero problems (so a failed run never leaves/clobbers the canonical JSON).
   Treat ANY non-zero `failure_markers` entry as fatal (read the actual marker keys, not a whitelist). Per
   regime require: `trials_issued == num`, `trials_with_records == trials_issued`, `oracle_records > 0`,
   `recall_at_2048_records == oracle_records`, `selected_contains_needle_records == oracle_records`, and
   assert `recall_at_2048 == selected_contains_needle_rate`.
2. **Re-run** the GPU sweep with the hardened driver via the fixed `serve.sh ds_recall_oracle` (from the
   repo root, proving the LAUNCH_CWD fix) → a fresh end-to-end artifact (same numbers: dense 1.0, sparse
   ~0.41) with genuine issued-tracking + the new `selected_contains_needle_records` + server prompt-token
   samples. One TP=8 server, eager, torn down to 0 MiB.
3. **Consumer (`build_ledger.py`)**: assert the same invariants before recording `run_meta` — exact
   dense+sparse regimes, ALL failure_markers zero, `trials_with_records == trials_issued`,
   `oracle_records == recall_at_2048_records == selected_contains_needle_records > 0`, `recall ==
   selected_rate`, non-null prompt-token samples, `index_topk == 2048`, `source_oracle_dir_basename ==
   ".sglang_ds_oracle"`.
4. **Serve harness (`serve.sh`)**: add `LAUNCH_CWD` (default caller `$PWD`); set it to `$EVID` for
   `ds_recall_oracle`; launch the server via a subshell `( cd "$LAUNCH_CWD" && exec python3 -m
   sglang.launch_server ... )` keeping `$LOG`/`$PIDFILE` absolute, so `serve.sh ds_recall_oracle` from ANY
   cwd lands the worker's oracle dir at `$EVID/.sglang_ds_oracle` (the driver default).
5. **Negative tests**: verify the driver `--reduce-only` exits 2 WITHOUT writing on a synthetic sink with a
   `no_active_trial` marker / a missing trial; verify `build_ledger.py` aborts on injected partial /
   failure-marker / missing-regime / wrong-source artifacts, then restore.
6. Fix the cheap nits I introduced: the duplicate `.sglang_ds_oracle/` `.gitignore` line and the stale
   `serve.sh` usage header.

## Concrete Success Criteria
1. `niah_recall_oracle.py` writes the canonical artifact ONLY via atomic rename after all checks pass; ANY
   failure marker (incl. `no_active_trial`) and any issued/recorded/recall/selected count mismatch is fatal;
   `--reduce-only`+`--expected-num` regenerates from the R18 sink. Verified: a synthetic bad sink → exit 2,
   no canonical write.
2. Committed `evidence/ac2_4_recall_oracle.json` regenerated from the verified sink: dense 1.0 / sparse
   0.4103, `oracle_records == recall_at_2048_records == selected_contains_needle_records` (2496 each),
   zero failure markers, `recall == selected_rate`, `source_oracle_dir_basename == ".sglang_ds_oracle"`.
3. `build_ledger.py` asserts the full invariant set before recording `run_meta.recall_oracle_corroboration`;
   verified it ABORTS on partial / failure-marker / missing-regime / wrong-source artifacts, then restored.
4. `serve.sh ds_recall_oracle` enforces `LAUNCH_CWD=$EVID` (subshell launch; LOG/PIDFILE absolute); existing
   modes unchanged (`bash -n` clean; LAUNCH_CWD default = caller PWD).
5. Tests pass; provenance consistent. Commit; round-19-summary with BitLesson Delta + Goal Tracker Update
   Request. No selection/adapter FIX. No exit by lying / editing loop state / cancel-rlcr-loop.
