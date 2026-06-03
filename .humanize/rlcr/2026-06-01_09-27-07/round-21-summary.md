# Round 21 Summary — Loop 7

## Mainline objective (round-21-contract.md)
**task20 (AC-2) — write the final strategic-gate supersession decision record that closes
Loop 7**, with a **blocking prerequisite**: repair the AC-6/task19 TTFT evidence provenance
(the R20-review gap) so task20 can cite a correct `m11`.

## Outcome: ACHIEVED — AC-2 + AC-6 MET; ALL 6/6 ACs MET. Loop 7 ready to close.

## Why this round (R20-review gaps)
The R20 review (ADVANCED) accepted the TTFT values but found two mainline gaps before close:
(1) the `ttft_*.json` artifacts lacked a `run_provenance` object and `m11` cited offset
commit SHAs (`f9f6ec056`/`68969deb0` as if they were the R19/R20 commits); (2) task20 (the
final decision record) was still unwritten. The reviewer directed: reconstruct provenance
(do **not** rerun) → then write task20.

## Work Completed (`coding`; no production-code change)

### Blocking prerequisite — AC-6/task19 provenance repair (reconstructed, no rerun)
- Verified the commit history: **`f9f6ec056`=R18, `68969deb0`=R19, `30173f08b`=R20**; R19
  and R20 commits touched **only `development/loop7/`**, so the DS/DSA production serving
  code is unchanged across R18→R19→R20. The R20 TTFT servers were launched from the R19
  tree `68969deb0` with the `--stream` probe uncommitted (committed as `30173f08b`).
- **Added `build_run_provenance(...)` to `perf_closed_batch.py`** (single schema source) and
  wired `--stream` to emit a `run_provenance` block (auto-detect git commit/dirty + GPU,
  plus pass-through `--launch-cmd`/`--op-point`/`--mem-per-gpu`/`--graph-evidence`) — future
  runs now self-document.
- **`ttft_add_provenance.py`** backfilled all **12 `ttft_*.json`** with the reconstructed
  `run_provenance` (server-code commit, tool commit, tree-dirty-during-run, GPU NVIDIA H200
  ×8/TP=8, exact launch cmd, effective config, mem fraction, gpu_mem/GPU, `mem_source`,
  graph flag + a representative `cuda graph: True` decode log line, radix/overlap flags,
  served count, op-point, artifact path), marked `reconstructed=true, reconstructed_in_round=21`
  with **metric values unchanged** (verified: p99 still 374.0 etc.).
- **Corrected `m11`'s commit story** to be exact + internally consistent, and **reconciled
  the 4K recall cell** (graph-N=50 default **80%**; eager N=20 **75%** — the prior `75%`
  conflated the two).

### Mainline — task20 final decision record (`m12_final_decision.md`)
The gate-supersession / loop-close artifact (plan M4, `refined_plan_v1.md:165-167`):
- **Supersedes** `runs/20260530_dsv32_loop6/ds_on_v32_decision.md`'s **Tier-2.A-primary**
  ordering with the **M0 regime attribution** (4K budget-limited / 16K budget-partial ~46%
  cap / 64K scorer-limited) → **Tier-2.B is the primary long-context lever; Tier-2.A is a
  bounded opt-in 4K lever**; states exactly what changed (the prior rationale was **sound
  when written**; the oracle data is what changed).
- Cites the full **per-AC evidence chain** (AC-1 oracle closure, AC-2 N=50 recall matrix +
  CIs, AC-3 hybrid-scorer non-regression, AC-4 production-ready lifted disposition, AC-5 64K
  servability, AC-6 perf+TTFT guardrails), including the **R8 stride/oracle provenance
  explicitly** (`oracle_stride_reference.json` `emitted_stride_value_counts {"1":14640}` +
  the `selection_kernel.py::_maybe_record_recall_oracle` stride=1 call site, raw sink noted
  gitignored), and confirms the **DEC-4 close-gate** (production-ready `m9` disposition
  exists → no dangling pursued-hardening).
- Records the **Ultimate-Goal outcome**: gap rigorously characterized (M0) + materially
  partially closed (16K 6%→38% via Tier-2.B decode-free; 4K 75%→95% via opt-in Tier-2.A);
  64K residual is a characterized scorer-limited negative result → the DEC-5 learned-selector
  follow-on; DSA default + Loop-6 Tier-1 op-point non-regressed.
- **Reviewed via `/humanize:ask-codex`**: "supersession logic supported; Loop-6 change +
  prior-soundness explicit; R8 stride provenance explicit; DEC-4 satisfied." Integrated its
  **2 high-signal factual fixes**: 4K is `recall@8192=100%` (not `@4096`); the AC-2
  graph-N=50 4K default is **80%** (not 75%). Also reconciled the same cell in `m11`.

## Files Changed
- `development/loop7/m12_final_decision.md` (NEW — the loop-close decision artifact).
- `development/loop7/ttft_add_provenance.py` (NEW — reconstruction backfill).
- `development/loop7/perf_closed_batch.py` (`build_run_provenance` + `--stream` self-docs).
- `development/loop7/m11_perf_consolidation.md` (exact commit story + 4K cell reconcile).
- `development/loop7/ttft_*.json` ×12 (added `run_provenance`; metrics unchanged).
- Commit `17782726f` (local — loop hook). **No production-code change.**

## Validation
- Provenance backfilled into all 12 artifacts; metric fields byte-unchanged (spot-checked).
- `m11`/`m12` commit story matches `git log`; `git diff --check` clean; both scripts
  `py_compile` clean.
- ask-codex factual review passed after integrating its 2 corrections.
- Full DS unit suite → **350 passed + 9 subtests** (no production-code regression).
- No GPU rerun (reviewer-directed reconstruction); no servers launched this round.

## AC status after R21
- **AC-2 → MET** (`m12_final_decision.md` is the final decision artifact); **AC-6 → MET**
  (provenance complete). With AC-1/3/4/5 (prior), **ALL 6/6 ACs MET**.
- **Loop 7 is ready to close** — all plan tasks (task1–task20) are complete; the M4
  close-gate (the production-ready Tier-2.A disposition exists, DEC-4) is satisfied.

## Remaining Items
- None blocking. Queued (explicitly out of scope, before any final merge — not loop-blocking):
  remove plan/workflow markers (`AC-*`, `task*`, `Tier-2`, `DEC-`) from production
  code/comments/tests (documentation-only, no runtime effect); the DEC-5 learned/distilled
  selector for the 16K/64K scorer-limited residual is the approved-but-deferred follow-on
  (its own loop).

## BitLesson Delta
- Action: none
- Lesson ID(s): NONE
- Notes: applied the selected lessons — BL-20260530-durable-tracked-acceptance-evidence
  (provenance embedded as tracked JSON, not referencing the gitignored `/tmp` boot logs;
  graph/mem evidence baked into each artifact), BL-20260527-shell-json-into-python-source
  (the backfill reads/writes JSON as data via `json.load`/`json.dump`),
  BL-20260529-gate-record-artifact-before-raise / -ds-radix-flip-config-bound-artifact
  (config-bound, self-describing artifact records). No NEW reusable cross-round pitfall
  surfaced: this round was evidence-provenance hygiene + a documentation-synthesis decision
  record reusing established patterns; the conclusions are project evidence in
  `m12_final_decision.md`, not a generalizable engineering lesson.

## Goal Tracker
Updated directly (Plan Version 31): R21 plan-evolution row; task19 + task20 → Completed and
Verified (Verified = pending R21 Review); Active Tasks = none; the R8 stride-provenance
queued item marked RESOLVED-R21 (cited in `m12`); **all 6/6 ACs MET**. No Goal Tracker
Update Request needed.
