# Round 19 Summary

Mainline: **repair the AC-2.4 recall-oracle fail-closed contract** (both R18-review blockers). The R18
recall numbers were correct, but the producer wrote the canonical JSON before validating, `no_active_trial`
markers weren't fatal, the ledger checks were too weak, and `serve.sh` didn't encode the oracle-dir/CWD
agreement the TP worker depends on. Diagnostic/evidence-integrity only; no selection/adapter fix.

## Work Completed
1. **Producer (`niah_recall_oracle.py`)** — factored reduce+validate into `_reduce_validate_write()`:
   builds the report IN MEMORY, runs the FULL contract, and writes the canonical artifact ONLY via atomic
   `.tmp` → `os.replace` AFTER every check passes — so a partial/failed run leaves the canonical artifact
   UNTOUCHED. Removed the `span_out_of_range`/`exception` whitelist: ANY non-zero failure marker (incl.
   `no_active_trial` and any future name) is fatal. Per-regime invariants: `trials_issued == num`,
   `trials_with_records == trials_issued`, `oracle_records > 0`, `recall_at_2048_records == oracle_records`,
   `selected_contains_needle_records == oracle_records`, `recall_at_2048 == selected_contains_needle_rate`,
   and every issued trial has a non-null server `prompt_tokens`.
2. **Serve harness (`serve.sh`)** — added `LAUNCH_CWD` (default caller `$PWD`, so existing modes are
   unchanged); `ds_recall_oracle` sets `LAUNCH_CWD=$EVID` and the server launches via `( cd "$LAUNCH_CWD" &&
   exec python3 -m sglang.launch_server … )` (LOG/PIDFILE absolute; `exec` keeps `$!` == the server PID for
   teardown). So the TP worker's CWD-default oracle dir (`cwd/.sglang_ds_oracle`) = `$EVID/.sglang_ds_oracle`
   = the driver's default `--oracle-dir`, from ANY caller CWD. Also refreshed the stale usage header.
3. **Consumer (`build_ledger.py`)** — `validate_recall_oracle_artifact()` now asserts the FULL success
   contract before recording `run_meta.recall_oracle_corroboration`: exact `{dense,sparse}` regimes,
   `index_topk==2048`, `source_oracle_dir_basename==".sglang_ds_oracle"`, ZERO failure markers, and per
   regime `trials_with_records==trials_issued>0`, `oracle_records==recall_at_2048_records==
   selected_contains_needle_records>0`, `recall==selected_rate`, non-null prompt-token sample.
4. **Regenerated the committed artifact** via a fresh GPU sweep with the hardened driver, invoked through
   the fixed `serve.sh` **from the repo root** (no manual cd / env export) — proving the LAUNCH_CWD fix:
   records landed in the driver's default oracle dir, driver rc=0, identical numbers (dense 1.0, sparse
   0.4103) now with the full contract fields. One TP=8 server, eager, torn down to 0 MiB.
5. **Nits** — removed the duplicate `.sglang_ds_oracle/` `.gitignore` line; refreshed the serve.sh usage
   header to list all current modes.

## Verification (the guards actually fire)
- **Producer (offline, synthetic sinks)**: clean run → writes (exit 0); a `no_active_trial` marker / a
  missing trial / a null prompt_tokens → **exit 2 and NO canonical write** (atomic-write-after-validate).
- **Consumer (ledger negatives)**: a partial (`trials_with_records < issued`), a failure-marker, a
  missing-regime, a wrong-source-dir, a `recall != selected_rate`, and a null prompt-token-sample artifact
  each make `build_ledger.py` **abort** (AssertionError, exit 1); restored → provenance consistent.
- **Harness**: serve.sh launched from the repo root → worker oracle dir correct → driver got records.

## Files Changed (committed `8a179067d`)
- `development/loop13/niah_recall_oracle.py` (atomic write + all-markers-fatal + count invariants),
  `development/loop13/serve.sh` (LAUNCH_CWD + usage header), `development/loop13/build_ledger.py`
  (strengthened validator), `development/loop13/evidence/ac2_4_recall_oracle.json` (regenerated, full
  contract), `evidence/evidence_table.md` + `evidence/meta/*` (regenerated), `.gitignore` (dedup).

## Validation
- CPU suite, explicit args: `ac4_garbage_counters` (production + ref_faithful + ref_cosine),
  `ac2_1_forced_all_assertions`, `ac6_bisection_matrix`, `ac6_corrob_ref_cosine_noinc`,
  `ac6_score_reduce_corrob`, `ac2_2_head_agg`, `ac4_sample_ids`, `verify_ac2_3 .sglang_ds_scorecap_sparse`
  (committed AC-2.3 artifact unchanged), `test_reference_selectors` (5/5) — **all exit 0**.
- `build_ledger.py` → provenance consistent; `bash -n serve.sh` clean. No sink/`.pt`/`.humanize` raw
  artifacts committed. One TP=8 server, eager, torn down to 0 MiB. No selection/adapter **fix**.

## Remaining Items (for AC-8 COMPLETE)
- **AC-3.1** captured decode-row materialized fp32 `K_label` selected-index equality (synthetic proof
  insufficient — needs latent/scales/query capture extension).
- **AC-4** serial cells (DSA-radix serial, production DS sparse serial, ref_faithful/ref_cosine serial) +
  selected-vs-total gaps.
- **AC-8** final root-cause writeup.

## BitLesson Delta
- Action: update
- Lesson ID(s): BL-20260621-niah-recall-oracle-fail-closed-span-self-verify
- Notes: A fail-closed PRODUCER is only fail-closed if it (1) writes the canonical artifact LAST, via atomic
  rename, ONLY after every check passes — writing-then-checking still leaves a canonical artifact on
  failure (the exact R15 forced-all-vs-scored class, now repeated for the oracle); (2) treats EVERY failure
  marker as fatal, not a hand-picked whitelist — the server records `no_active_trial` as a fail-closed
  marker too, so `sum(failure_markers.values()) == 0` is the right gate, not `span_out_of_range/exception`
  only; and (3) asserts COUNT invariants (issued==with_records, oracle==recall==selected counts,
  recall==selected_rate, non-null prompt-tokens) so a partial run can't pass. The CONSUMER (ledger) must be
  an INDEPENDENT gate asserting the same invariants — never trust a nearby JSON by presence alone. And a
  cross-process diagnostic whose worker resolves a dir from CWD must bake that CWD into the serve harness
  (LAUNCH_CWD via a `( cd … && exec … )` subshell, LOG/PIDFILE absolute), not merely document it, and prove
  it by launching from an arbitrary CWD. Watch the defaultdict footgun: `sum(d[k] for k in WHITELIST)`
  inserts those keys — sum over `d.values()` instead.

## Goal Tracker Update Request

### Requested Changes (already applied to the mutable section):
- Plan Version → 22 (Round 19); added the Round-19 evolution row.
- task4 → done (AC-2.3 R5; **AC-2.4 R18 measured + R19 fail-closed hardened**).
- Marked BOTH R18-review blockers **RESOLVED (R19)** with verification evidence (producer atomic
  write/all-markers-fatal/count invariants; ledger independent gate; serve.sh LAUNCH_CWD proven from repo
  root).

### Justification:
Codex flagged the same evidence-integrity class as R15 (a partial/failure-marker run could leave a canonical
JSON the ledger accepts) plus an unenforced oracle-dir/CWD requirement. The repair makes the producer never
publish on failure, treats every marker as fatal, makes the ledger an independent gate, and bakes the CWD
agreement into serve.sh (proven by a repo-root re-run). The committed artifact is unchanged in substance
(dense 1.0, sparse 0.4103) but now carries the full contract fields. Remaining close-out (AC-3.1, AC-4
serial/selected-vs-total, AC-8) is the active sequence toward COMPLETE — not deferrals.
