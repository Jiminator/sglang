# Goal Tracker

<!--
This file tracks the ultimate goal, acceptance criteria, and plan evolution.
It prevents goal drift by maintaining a persistent anchor across all rounds.

RULES:
- IMMUTABLE SECTION: Do not modify after initialization
- MUTABLE SECTION: Update each round, but document all changes
- Every task must be in one of: Active, Completed, or Deferred
- Deferred items require explicit justification
-->

## IMMUTABLE SECTION
<!-- Do not modify after initialization -->

### Ultimate Goal
Produce the verdict loop 11 never reached. loop 11 landed the whole structural build (M0–M3: deleted
the per-token TokenLabelTable, made table-free absorbed-latent scoring the served DS default,
authorized radix-on under owner DEC-12) but was terminated before its M4 measurement milestone. The
machine that held the live op-point (the calibrated GLM-5.1 channel mask under `/models/`, the live
servers) was then released. loop 11b: (1) re-establishes the serving op-point on a fresh 8×H200 node,
(2) runs the two pending loop-11 M4 measurement tasks — the per-step tax guard (loop-11 task8) and
the locked AC-11 sweep (loop-11 task9) — to HARD verdicts against native DSA, (3) delivers the
client-facing DS-vs-DSA end-to-end comparison per `development/SLOS.md`, and (4) makes the DS
production enablement path documented and free of stale defaults. A FAIL on the throughput SLO is an
acceptable, reportable outcome — the deliverable is the honest measured gap, not a DS win.

Scope: "per SLOS.md" = PRIMARY client workload only (GLM-5.1-FP8, 4096 ISL / 512 OSL, conc 16–64,
~55% prefix hit). The SLOS.md DEFERRED requirement (128k ISL / 1024 OSL) is OUT OF SCOPE.

### Acceptance Criteria
<!-- Each criterion independently verifiable. Full prose in development/loop11b/plan.md. -->

- **AC-0 — Op-point re-established on the fresh node.**
  - AC-0.1: GLM-5.1-FP8 channel mask regenerated with a `--dry-run-blocks 1` placement preflight and a
    committed `provenance.json` (exact command, corpus/token-block digest, model snapshot id, package
    + CUDA/driver versions, BOTH tensor `content_sha256` and full-file `sha256`, inherited-vs-re-earned
    statement). Calibrate flags: `--dtype bfloat16 --kv-cache-dtype fp8_e4m3 --tp 8 --label-dim 16
    --page-size 64 --num-samples 256 --block-size 512 --seed 42` + explicit corpus. The mask serves and
    clears AC-5 recall. Mask without provenance / passing dry-run / comparable recall is rejected.
  - AC-0.2: Radix-on authorized for THIS boot WITHOUT the dev override. DEC-1: validator pins mask by
    tensor `compute_content_sha256` + config fingerprint, REPLACING full-file `_sha256_file` + path
    pinning; legacy full-file/path and label-capture schemas stay fail-closed-rejected. Fixture minted
    once: boot under `SGLANG_DS_RADIX_OVERRIDE=1` → DEC-12 edge probes → `write_radix_fixture_state` →
    re-boot WITHOUT override authorizes. `/server_info` confirms the locked key set (model_path GLM
    snapshot, tp_size=8, page_size=64, kv_cache_dtype=fp8_e4m3, enable_double_sparsity=true,
    double_sparsity_config table-free scorer_norm=off, radix_fixture_artifact set,
    disable_radix_cache=false, disable_cuda_graph=false, disable_custom_all_reduce=false,
    mem_fraction_static=0.8, max_running_requests=64, cuda_graph_max_bs=64). Negative-control /
    mismatched / legacy fixtures refused; same-content mask at a different path/timestamp still authorizes.
  - AC-0.3: Capacity reproduces — `/server_info` `internal_states[*].memory_usage.token_capacity` (+
    `effective_max_running_requests_per_dp`) yields derived decode-bs cap ≥ 64 @ mem 0.8, CUDA-graph
    capture OK all buckets, no TokenLabelTable allocated, conc-64 running-req peak ≥ 61. (504640/410560
    are references, not hard gates.)
- **AC-2 — Tail TTFT (HARD @ locked sweep).** DS P99 TTFT < 22 s at conc 16/32/64, ABSOLUTE bar judged
  regardless of DSA (DEC-6); DS/DSA P99 ratio (≤1.10× comparator) REPORTED not gated; no admission cap
  below nominal conc ≤ 64. A 1-trial directional ladder is NOT the AC-2 verdict.
- **AC-3 — Throughput (HARD per-request).** DS decode-TPS p50 ≥ 30 (output tokens/(latency−TTFT)) at
  conc 16/32/64, ABSOLUTE regardless of DSA (DEC-6); DS/DSA p50 ratio (0.95×) + aggregate total-tokens/s
  REPORTED, not gated (DEC-3).
- **AC-4 — Per-step tax guard (HARD; loop-11 task8).** DS-vs-DSA same-batch one-batch decode-window
  ratio ≤ 1.10 @ bs64 (both mem 0.8, GRAPH mode); bs30 window ≤ 380k µs. Declares radix/shape/warmup/mem.
- **AC-5 — Quality + no-op refusal.** Recall@2048 ±0.5pp fail-closed vs frozen
  `loop9/runs/20260610_m0/recall_baseline.json` WITH comparability (matched length-set + per-length
  sample-count); else a served-fp8 baseline is defined+recorded. Cross-rank bit-identity HARD; radix
  value-equivalence re-verified on mint. Every published SLO trial: `dense_fallback_total == 0` AND a
  sparse-selection proof (`selected_tokens_mean < total_tokens_mean`); missing fields = input refusal.
- **AC-6 — DS concept intact.** Offline mask → absorbed-latent signatures → query·signature scoring →
  top-k → sparse MLA decode. No dense fallback; no DSA-indexer substitution.
- **AC-7 — DSA-native default un-regressed (strict).** Shared-surface changes (memory accounting, radix
  incl. DEC-1 validator change, graph runner, serve defaults) trigger the DSA regression SAME round;
  DSA `memory_usage.token_capacity` + behavior unchanged vs its own node reference; frozen Case-2 recipe matches.
- **AC-8 — Protocol / ledger / queue / evidence discipline.** `queue.md` current every round; evidence
  pre-flight before each handoff (artifact exists + tracked + claim matches POST-commit state);
  `results.md` rewrite-over-append; one-trial honesty until the locked sweep; frozen references intact;
  `git push` every round; close-out preserves raw evidence under stable naming (provenance.json,
  run-order ledger, per-trial reuse summaries, bench JSONLs + sidecars, per-boot server logs, fixture,
  mask hashes, /server_info snapshots, exact benchmark commands).
- **AC-9 — Measurement honesty / op-point fairness.** Locked sweep publishes TWO declared comparisons
  (DEC-2): (a) production-envelope DS 0.8 / DSA 0.85; (b) same-memory both 0.8. Both radix-ON, matched
  workload/seed-family. TWO trials/conc at the SAME per-conc seed = repeated run-to-run-stability
  measurements (NOT independent samples), report min/median/max; comparator `--ac11` trial-floor lowered
  to 2 (in-scope tweak). Enforceable one-server run order logged (alternate DSA/DS by trial; block-scheduled
  = LABELED unpaired + explicit acceptance). Per-trial prefix-reuse recorded. DSA miss does NOT invalidate the op-point.
- **AC-UX — Production enablement documented; stale refs fixed (no ABI change; DEC-5).**
  - AC-UX.1: runbook zero → serving DS server (calibrate+preflight → boot under override → DEC-12 probes
    → mint fixture → serve without override); fixture-fingerprint flow explained once; prod-vs-diagnostic
    knobs labeled; prefers `/server_info`.
  - AC-UX.2: Category-A fixes — repoint `serve_double_sparsity.sh`/`serve_native_nsa.sh`
    MODEL_PATH/CHANNEL_MASK_PATH to GLM-5.1-FP8; correct mem-0.6 / TokenLabelTable comments; reconcile
    the loop8 throughput warning to the loop11b verdict; de-DeepSeek `calibrate.py` docstring/argparse +
    `config.py` top_k comment; update stale `benchmark*.sh` / `benchmark_compare.py` comments + the
    "independent trials" wording + `CLIENT_SLOS.md`→`SLOS.md` references.
  - AC-UX.3: Category-B CLI help — `server_args.py:6090` no longer "for DeepSeek-V3.2 (FP8)"; `:6103`
    example path GLM-appropriate. NO flag rename / JSON-config schema change (DEC-5).

### Owner Decisions (RESOLVED; do not relitigate)
DEC-1 content-hash mask authorization · DEC-2 publish both production-envelope + same-memory · DEC-3
per-request decode-TPS gated / aggregate descriptive · DEC-4 two trials same seed + comparator floor→2
· DEC-5 Category A+B UX only, no ABI · DEC-6 DS judged on absolute SLO regardless of DSA.

---

## MUTABLE SECTION
<!-- Update each round with justification for changes -->

### Plan Version: 5 (Updated: Round 3 review)

#### Plan Evolution Log
<!-- Document any changes to the plan with justification -->
| Round | Change | Reason | Impact on AC |
|-------|--------|--------|--------------|
| 0 | Initial plan | - | - |
| 0 | Calibration recipe corrected to the loop8 DEC-3 GLM recipe: `--dtype fp8_e4m3 --label-dim 32` (plan/AC-0.1 carried the DeepSeek-V3.2 values `--dtype bfloat16 --label-dim 16`) | A label-dim-16/bf16 mask (content a4be98c4) served **−5.2pp** recall at L4096 vs the frozen baseline (58.0% → 52.8%) and −3.1pp at L16384. loop8 `task5_calibration_recipe.md` + DEC-3 set GLM `label_dim=32` (proportionate to `qk_nope_head_dim=192`, retains 2× channels) at `--dtype fp8_e4m3`; that recipe produced loop11's baseline-matching mask (R16: L4096 58.4%). Radix on-vs-off was equivalent (GATE A passed), so the gap is mask-quality from the wrong recipe, not a radix issue. | AC-0.1 recipe params corrected (the AC's binding criterion — mask serves + clears AC-5 recall — is unchanged). The label-dim-16 mask is superseded; re-calibrating + re-running the recall/mint gates. |
| 0 review | Completion claim rejected; M-B/AC-8 remain active | Review found the same-memory comparison deferred, official `benchmark_compare.py --ac11` refusing the published cross-side comparison due commit mismatch, AC-4 replaced by a sweep TPOT proxy, per-trial prefix-reuse/no-op/aggregate evidence missing, DS conc-64 achieved concurrency below nominal, raw JSONL/log/corpus artifacts ignored, no push, and `queue.md` stale. | AC-2/AC-3/AC-4/AC-5/AC-8/AC-9 remain incomplete for publication. AC-0/AC-6/AC-7 substantially advanced. |
| 1 review | Completion claim rejected again; R1 advanced comparator/same-memory/AC-4/admission, but publication remains blocked by AC-5 and AC-8 | R1 produced accepted comparator outputs for both op-points and a dedicated tax probe, but every committed DS `*.evidence.json` says `REFUSE` because `dense_fallback_total`, `selected_tokens_mean`, and `total_tokens_mean` are null; `ac5_no_op_evidence.md` explicitly defers wiring the GLM per-request DS summary. `queue.md` also remains contradictory/stale, raw JSONL/log artifacts are ignored rather than tracked, and push is still pending. | AC-2/AC-3/AC-4/AC-9 advanced; AC-5 and AC-8 remain incomplete, so task8/task9/task11 stay active for publication. |
| 2 review | Completion claim rejected; R2 advanced AC-5/AC-8 evidence transport, but AC-5 counters and AC-8 close-out remain incomplete | R2 wired GLM/`dsa_backend` DS summary and the committed `results_r2/` package reproduces both comparators from tracked `.gz` artifacts. However `bench_serving.py` derives `total_tokens_mean` as `selected_tokens / sparsity_rate` while the DS publishers set `sparsity_rate = 1 - selected/total`; the committed AC-5 total-token field is therefore not the sequence-length total Claude claims. `queue.md` is also still not current (`task2`/`task3` remain `pending`, close-out text still references R1 and marks push-pending work as DONE), and there is still no owner-approved push destination or written waiver. | AC-2/AC-3/AC-9 advanced; AC-5 and AC-8 remain active under task8/task9/task11 until total-token evidence is corrected, ledgers are current, and push or waiver is completed. |
| 3 review | Completion claim accepted; R3 closes the remaining AC-5 and AC-8 blockers | Review verified the explicit `total_tokens` contract in code, replayed `results_r3/` from a `git archive` of committed `.gz` artifacts only, verified all raw hashes, all 6 DS `trial_evidence.py` gates, both comparator replays (`production_envelope` and `same_memory`, rc=3), focused metric unit tests, clean worktree, and remote fork ref at `e0935e5a9`. Minor stale comments around `sparsity_rate` semantics remain queued because they do not affect the published AC-5/AC-8 verdict. | AC-5 and AC-8 are resolved; task8/task9/task11 moved to completed. |

#### Active Tasks
<!-- Mainline tasks only: each task must directly advance the round objective and carry routing metadata -->
| Task | Target AC | Status | Tag | Owner | Notes |
|------|-----------|--------|-----|-------|-------|
| — | — | none active | — | — | Round 3 review verified task8/task9/task11 completion; all plan tasks are now completed or explicitly out of scope. |

### Blocking Side Issues
<!-- Only issues that directly block current mainline progress belong here -->
| Issue | Discovered Round | Blocking AC | Resolution Path |
|-------|-----------------|-------------|-----------------|
| DS SLO trial no-op evidence is not yet publishable | 1 review / 2 review | AC-5, AC-9 | RESOLVED R3: explicit `total_tokens` field on both DS publishers (from host seq_len) + direct bench aggregation + consistency gate in trial_evidence; regenerated `results_r3/` → all 6 DS trials PASS, total ~4765 = true seq-len (`8df44a59c`/`c805b4be5`). |
| AC-8 close-out is not fully current or complete | 1 review / 2 review | AC-8 | RESOLVED R3: queue.md/results.md are ONE current state; raw evidence committed losslessly + preflight CLEAN (both comparator replays rc=3, 6/6 trial_evidence, hashes verified); PUSHED to owner fork Jiminator/sglang dev/double-sparsity-standalone (remote ref verified at `e0935e5a9`). |

### Queued Side Issues
<!-- Non-blocking issues stay queued and must NOT replace the round objective -->
| Issue | Discovered Round | Why Not Blocking | Revisit Trigger |
|-------|-----------------|------------------|-----------------|
| Broader cleanup of historical AC/DEC terminology outside the GLM production path | 0 review | Does not block the measurement verdict once production-facing text is clean | Next documentation/code-health pass |
| Plan terminology remains in implementation comments/help text (`benchmark_compare.py`, `batch_result_processor.py`, serve-script comments) | 1 review | Does not change the measured verdict, but violates the plan's implementation-note hygiene | Clean during the next doc/code-health pass after AC-5/AC-8 are closed |

### Completed and Verified
<!-- Only move tasks here after Codex verification -->
| AC | Task | Completed Round | Verified Round | Evidence |
|----|------|-----------------|----------------|----------|
| AC-8 | task1 kickoff queue created | 0 | 0 review | `development/loop11b/queue.md` exists, but final queue alignment remains active under task11. |
| AC-0.1 / AC-UX.2 | task2 serve-script defaults repointed | 0 | 0 review | `development/serve_double_sparsity.sh` and `development/serve_native_nsa.sh` default to the GLM-5.1-FP8 snapshot; DS mask path defaults to the GLM mask. |
| AC-3 / AC-5 / AC-9 | task3 pre-sweep methodology review | 0 | 0 review | `development/loop11b/runs/20260616_ma/task3_codex_methodology_review.md`; its comparator/reuse/no-op findings were carried into the later R1/R2 fixes. |
| AC-0.1 / AC-5 | task4 mask regen + provenance | 0 | 0 review | `development/loop11b/runs/20260616_ma/provenance.json`; recipe correction accepted because the original label-dim-16 recipe failed recall and label-dim-32 matched the frozen baseline. |
| AC-0.2 / AC-5 / AC-7 | task5 content-hash validator + radix mint | 0 | 0 review | `validator.py` content-hash fingerprint, fixture `channel_mask_content_sha256=35155ac4...`, GATE A/B/C artifacts; focused unit suite passed. |
| AC-0.3 / AC-7 | task6 capacity + DSA regression probe | 0 | 0 review | `server_info_ds.json` and `server_info_dsa.json` record token capacities 504640/410560 and graph-capable 64/64 op-point. Evidence doc has a stale superseded-hash line to fix under task11. |
| AC-4 | task7 dedicated per-step tax guard | 1 | 1 review | `results_v2/tax/log_ds_c64.txt`, `log_dsa_c64.txt`, `log_ds_c30.txt`, and `log_dsa_c30.txt` record graph-mode fixed-batch median ITL ratios bs64=1.056 and bs30=1.057, both ≤1.10; bs30 31.85ms is below 380k µs. |
| AC-UX | task10 production UX cleanup | 0/1 | 1 review | Operator-facing serve output was cleaned enough not to block the measurement close-out; remaining plan terminology is queued outside the production path. |
| AC-2 / AC-3 / AC-5 / AC-9 | task8 publishable locked sweep at both op-points | 3 | 3 review | Explicit `total_tokens` contract (`8df44a59c`); regenerated `results_r3/` from one run HEAD; archive replay verified raw hashes, all 6 DS trial_evidence PASS, and both comparators rc=3 with corrected `total_tokens_mean ~4765`. |
| AC-2 / AC-3 / AC-5 / AC-9 | task9 headline DS-vs-DSA report | 3 | 3 review | `development/loop11b/results.md` and `queue.md` point at `results_r3/`; verdict reproduces DS PASS@16/32 and FAIL@64 (honest SLO failure), DSA also FAIL@64. |
| AC-8 | task11 close-out evidence + queue/results + push | 3 | 3 review | `results_r3/` committed losslessly (`*.jsonl.gz`/`*.log.gz`, hashes, REPRODUCE); replay from `git archive` only succeeded; worktree clean; owner fork `jiminator/dev/double-sparsity-standalone` verified at `e0935e5a9`. |

### Explicitly Deferred
<!-- Items here require strong justification -->
| Task | Original AC | Deferred Since | Justification | When to Reconsider |
|------|-------------|----------------|---------------|-------------------|
| 128k ISL / 1024 OSL second op-point | (SLOS deferred) | Round 0 | OUT OF SCOPE per plan goal | a dedicated long-context loop |
