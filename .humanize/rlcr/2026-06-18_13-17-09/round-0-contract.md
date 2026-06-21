# Round 0 Contract

## Mainline Objective
Port the minimal correct table-free Double Sparsity runtime from the dev clone onto a fresh branch
off latest `origin/main` in the v2 clone, prune all dev-only scaffolding while porting, and drive it
through the full closure proof: cheap static/import/unit gates first, then the GPU evidence
(calibrate → DS-active boot → abort test → perf parity), then push to the fork.

Per RLCR round semantics, this round spans the **entire plan (M1–M9)**; intermediate milestone
checks use manual `ask-codex`, not round boundaries. The round summary is written only when the whole
plan is believed complete.

## Target ACs (primary focus this round)
- **AC-1** (branch hygiene & diff scope) and **AC-3** (import & prune closure) are the gating ACs:
  nothing downstream is trustworthy until the branch is clean and `import sglang` is green with no
  dropped-module references.
- AC-2, AC-4, AC-5, AC-9, AC-10 are verified by the cheap closure gates (M4) before any GPU.
- AC-6, AC-7, AC-8 are the GPU evidence ACs, gated on M4 being green.

## Blocking Side Issues In Scope
- None known at round start. The known hard hand-port risks (CUDA-graph hunk retarget to the
  `runner/` + `runner_backend/` layout; entangled `_maybe_record_recall_oracle` prune in
  `selection_kernel.py`; `deepseek_v2.py` +740-line DS integration) are mainline work (task5/task6/
  task4), not side issues.

## Queued Side Issues Out Of Scope
- Any non-DS upstream drift cleanup, unrelated test failures on `main`, or cosmetic refactors of the
  ported DS code beyond what closure requires. Log and keep moving; do not let these become the
  round objective.

## Round Success Criteria
1. Branch exists in v2 clone off latest `origin/main` with `<BASE>` recorded (AC-1).
2. `python -c "import sglang"` clean; no shipped file imports a dropped module; precise exclusion
   sweeps pass; slim runtime + validator + calibrate unit tests pass (AC-2,3,5,9,10 cheap pass).
3. `validate_double_sparsity` ships and gates; radix fixture gate fully removed (AC-4).
4. Fresh mask calibrated and loader-accepted; DS-active boot proven via `meta_info["double_sparsity"]`
   (selected<total, dense_fallback==0); abort path same-step finish proven (AC-5,6,7).
5. Perf wrapper reproduces conc-64 within band: p50 decode TPS ≥ 24.2 AND P99 TTFT ≤ 30.1 s, evidence
   saved (AC-8).
6. Final dead-code sweep + exclusion re-check clean; branch pushed to `Jiminator/sglang` (AC-1,2,10).

## Hard Operational Constraints (carry forward — do not relitigate)
- NEVER set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments` for **serving** (calibration-only is OK,
  separate process).
- One TP=8 server at a time; tear down + wait for GPU idle between boot (M6) and perf (M8).
- No blanket `nvidia-smi` PID kills; no `pkill -f` matching the parent shell.
- All serving/perf in the v2 clone; loop machinery + plan stay in this dev clone. Shipping branch
  never receives `development/`, `.humanize/`, `.pensieve/`.
- Push only to the fork `Jiminator/sglang`; never the public upstream `sgl-project/sglang`.
