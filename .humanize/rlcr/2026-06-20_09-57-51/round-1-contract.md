# Round 1 Contract

## Mainline Objective
Make the reference selector a **valid AC-5 gate input**: serve the cosine reference (AC-3.2), make the raw-dot ceiling **faithful** (H3-clean: include the current decode slot so it is not the production-validity-contaminated run) and **leak-free** (TF32 disabled, AC-3.4), measure dense+sparse for the faithful raw-dot and the cosine arms, then **recompute the AC-5 gate from best(faithful-raw-dot, cosine)** with the numeric gaps recorded.

This directly clears Codex's Round-0 blocking side issue #1 ("Reference path is not yet a faithful AC-3 ceiling") and its mainline gaps #1 (cosine/gate) and #2 (H3-contaminated, not-leak-free reference).

## Target ACs
- **AC-3.2** (served cosine reference) + **AC-3.3** (DS-active invariants; faithful dense reports `selected == seq_len`) + **AC-3.4** (leak-free fp32 / TF32 disabled) — the reference-ceiling completion.
- **AC-5** (recompute the gate from a valid best-of-raw/cosine ceiling).

## Blocking Side Issues In Scope (truly block the mainline)
- `reference_cosine` is accepted by config but raises `NotImplementedError` — must implement the materialized per-head signature cosine (normalize after mask-channel gather) since the gate requires best-of raw/cosine.
- TF32 not disabled in reference modes — add `torch.backends.cuda.matmul.allow_tf32 = False` (+ cuDNN) in the reference path, or label the arm TF32-risk.
- Raw-dot reference is H3-slot-validity-contaminated in dense (drops current slot) — add a faithful ceiling that includes the current slot (so dense `selected == seq_len`), keeping the existing H3-tainted run as a separate "production-validity scorer-isolation" control.

## Queued Side Issues (documented, OUT OF SCOPE this round)
- Full per-arm JSON evidence ledger for ALL Round-0 arms + length-cap garbage-rate columns + sample IDs/order (AC-4 completeness) — except I WILL attach proper per-arm metadata (config, selected-vs-total, sample set) for the NEW arms produced this round.
- AC-2 capture artifacts: `ds_capture` run, `.sglang_ds_*` dumps, `cheap_controls.json`, forced-all physical-slot-assertion JSON (AC-2.1/2.2/2.3) — task2/3/4.
- AC-7 no-mask ablation + one-knob accuracy sweep + per-head offline oracle — these run AFTER AC-5 is recomputed (they are gated on the gate outcome), so they are next-round, not this round's mainline.
- Committed CPU unit-test files for the diagnostic helpers.

## Round Success Criteria
1. `serve.sh` has a `ref_cosine` mode; `selector_impl="reference_cosine"` serves end-to-end (no NotImplementedError), DS genuinely active on sparse, and produces a real GSM8K dense+sparse pair.
2. A faithful raw-dot ceiling (current decode slot included; TF32 disabled) serves, reports `selected == seq_len` in dense, and produces GSM8K dense+sparse. The existing H3-contaminated raw-dot run is relabelled as a scorer-isolation control (not the ceiling).
3. AC-5 gate recomputed: naive-DS = best(faithful-raw-dot, cosine); record measured naive-DS dense/sparse, DSA dense/sparse, the computed gaps, and the GOOD/BAD outcome under the confirmed threshold (sparse within 5 pts + no collapse; dense within 3 pts).
4. New arms carry per-arm metadata (config JSON, selected-vs-total by regime, sample set), and the evidence table gains the cosine + faithful-raw-dot rows.
5. goal-tracker mutable section updated; round-1-summary written with BitLesson Delta; changes committed; tree clean; one TP=8 server at a time.
