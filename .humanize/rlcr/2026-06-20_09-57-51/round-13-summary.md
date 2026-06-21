# Round 13 Summary

Mainline: **deliver AC-2.1 — the forced-all dense physical-slot assertions**, the plan's load-bearing
downstream-isolation control (the first GPU/instrumentation close-out item). Guarded diagnostic
instrumentation (no fix; production byte-identical when off) + one eager GPU run + a fail-closed reducer.

## Work Completed
- **Guarded instrumentation (default-off):**
  - `config.py`: new bool flag `forced_all_assert`, wired in all four places (`_ALLOWED_FIELDS`, the
    dataclass, validation, and the `parse_double_sparsity_config` explicit constructor).
  - `forced_all_assert_capture.py` (new): `maybe_dump_forced_all_assert()` dumps, per (rank, req, layer),
    the post-`logical_to_physical` **physical** slots, the forced **logical** positions, the request's
    `req_to_token[req, 0:seq_len]` slice, and the adapter `error_count`. Host-side copy only.
  - `deepseek_v2.py`: a guarded call right after `logical_to_physical`, **inside** the existing
    `not torch.cuda.is_current_stream_capturing()` guard (mirrors `_publish_ds_request_summary`). Fires
    only when the flag is on; the selected set is unchanged either way → off-path byte-identical (verified:
    reference tests pass, flag default `False`, `py_compile` clean, unknown-field guard intact).
- **Run + reducer:** `serve.sh ds_forced_all_assert` (= `ds_forced_all` + `forced_all_assert`, eager) —
  one TP=8 server, small dense drive, torn down to 0 MiB. `ac2_1_forced_all_assertions.py` →
  `evidence/forced_all_assertions.json` (fail-closed: nonzero exit on zero dense rows / missing field /
  any failing assertion; verified exit 2 on an empty dir).
- **Result (PASS) on 4368/4368 real dense rows** (median seq_len 793): forced logical sweep
  `[0..seq_len-1]` 4368/4368; physical == `req_to_token[req, 0:seq_len]` (element-wise) 4368/4368;
  **0** duplicate, **0** live-lane `-1`, **0** out-of-range, **0** adapter `error_count`. ⇒ When the dense
  selected set is forced to all tokens, the `logical_to_physical`→`transform_index_page_table_decode`
  adapter maps it to **exactly the request's own KV slots** (the same DSA feeds) with zero garbage. So the
  forced-all dense selection is a **provable no-op**, which confirms the dense regression is **downstream
  of selection** (the `_slot_written` current-slot exclusion, H3) — on live physical slots, not theory.
  The same counters are the **AC-4 garbage-rate** for the forced-all control (all zero); "unwritten" is
  subsumed by the physical==req_to_token equality. Wired into the ledger
  (`ds_forced_all.forced_all_assertions_artifact`) + the `findings.md` AC-2.1 section.

## Files Changed (committed `e62112335`)
- NEW (production, guarded diagnostic): `python/.../double_sparsity/forced_all_assert_capture.py`;
  `config.py` (flag); `deepseek_v2.py` (hook).
- NEW (loop13): `ac2_1_forced_all_assertions.py`; `evidence/forced_all_assertions.json`.
- MODIFIED: `serve.sh` (ds_forced_all_assert mode), `build_ledger.py` (artifact wiring + NOT_INSTRUMENTED
  update), `evidence/findings.md`, `evidence/evidence_table.md`, `evidence/meta/run_meta.json`,
  `evidence/meta/arms/*.json`, `.gitignore` (forcedall capture dir).

## Validation
- Full suite — `test_reference_selectors` (5/5), `verify_ac2_3`, `ac6_corrob_ref_cosine_noinc`,
  `ac6_score_reduce_corrob`, `ac2_2_head_agg`, `ac4_sample_ids`, `ac6_bisection_matrix`,
  `ac2_1_forced_all_assertions` — **all exit 0**.
- Off-path: `forced_all_assert` defaults `False`; config parses/validates; `py_compile` clean; reference
  selector tests unchanged → production byte-identical when off.
- `build_ledger.py` → provenance consistent (blob `80e818a7ff84`); reducer fail-closed on empty dir (exit 2).
- One TP=8 server at a time; torn down to 0 MiB. No `.pt`/`.humanize` committed. No selection/adapter **fix**
  (guarded instrumentation only).

## Remaining Items (for AC-8 COMPLETE)
- **AC-3.1** captured-row materialized fp32 `K_label` selected-index equality — extend `latent_capture` to
  store bounded latent/scales/query, then an offline analyzer at top-2048 on captured decode rows.
- **AC-2.4** recall-oracle@2048 — NIAH-only (`recall_oracle` flag + `.sglang_ds_oracle/trial.json`); GPU.
- **AC-4** garbage counters on the SCORED arms (enable the same instrumentation on production_ds/ref_*),
  remaining serial cells (DSA-radix serial, production DS sparse serial, ref_faithful/ref_cosine serial),
  and selected-vs-total gaps.
- **AC-8** final root-cause writeup — after the above.

## BitLesson Delta
- Action: add
- Lesson ID(s): BL-20260621-forced-all-downstream-isolation-control
- Notes: Added a lesson on the downstream-isolation control technique (force a stage's output to the
  trivial value and ASSERT the downstream mapping is exact ⇒ residual degradation is downstream) and the
  safe way to instrument a hot decode seam (config-borne default-off flag wired in all four config places;
  capture module mirroring score_capture; hook inside the existing not-CUDA-capturing guard; host-side copy
  only ⇒ byte-identical when off; eager run + fail-closed reducer).

## Goal Tracker Update Request

### Requested Changes:
- Mark **AC-2.1 (task2) DONE** — `forced_all_assertions.json`: 4368/4368 dense rows, physical==req_to_token,
  0 garbage, PASS ⇒ dense selection is a provable no-op (H3 confirmed downstream).
- Note **AC-4 garbage counters** instrumented for the forced-all control (all zero); enabling on the scored
  arms is the remaining AC-4 garbage-counter work.
- Plan Evolution Round-13 row added.

### Justification:
AC-2.1 is the plan's lower-bound downstream-isolation control; its PASS on live physical slots removes the
selected-index/adapter family from the dense cause and confirms the slot-validity (H3) verdict with
measured evidence, not theory. The instrumentation is a guarded, default-off, byte-identical-when-off
diagnostic (no fix), reusing the existing capture pattern. The remaining close-out items (AC-3.1, AC-2.4,
AC-4 scored-arm garbage/serial/selected-vs-total, AC-8) are the next sequence toward COMPLETE.
