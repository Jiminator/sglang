# Round 4 Summary — Loop 7

## Mainline objective (round-4-contract.md)
Make the M0 recall oracle **fail-closed and binding** (AC-1/AC-2): config-borne
activation so it records on TP workers, strict span validation, explicit failure
artifacts, expected-record-count assertions in the sweep, and a re-run of
4K/16K/64K with no missing lengths — so task7's budget-vs-scorer attribution is
binding.

## Outcome: ACHIEVED — 64K oracle now MEASURED (not inferred), fail-closed, N=20.

## Work completed
1. **Config-borne activation (reaches TP workers).** Added `recall_oracle: bool`
   to `DoubleSparsityConfig` (`_ALLOWED_FIELDS` + dataclass + validation +
   `_coerce_bool` parse). Default off ⇒ byte-identical selection. The hook latches
   `oracle_artifact_sink.enable_via_config()` when the config flag is set, so the
   sink/trial paths resolve without env (env does not reach workers).
2. **Fail-closed hook** (`_maybe_record_recall_oracle`). No active trial,
   out-of-range needle span, or payload exception now emit explicit `failure`
   records keyed by `(request_id, trial_id, layer_id, decode_step)` instead of
   returning silently / swallowing. Out-of-range spans are **rejected, not
   filtered** (the old filter silently masked the absent 64K). Added a
   module-global sample counter so cross-process worker records don't all collide
   on `decode_step=0`.
3. **Rode the production long-context path.** Threaded `recall_oracle` into
   `retrieve_topk_graph_safe` (+ its line-1283 hook call + the deepseek_v2
   graph-safe call site). Reverted the initial `_force_eager_select |=
   recall_oracle` — the eager logical scorer does not scale to long-context int8
   tensors (error_containment silently dropped DS to dense, `ds`=None). Validator
   now requires `--disable-cuda-graph` when `recall_oracle` is on.
4. **Shared-FS cross-process files.** Default trial/sink under
   `./.sglang_ds_oracle/` (the repo bind-mount both driver and worker share);
   the old `/dev/shm` default is a per-sandbox tmpfs the worker can't see.
   Env-overridable via `SGLANG_DS_RECALL_ORACLE_DIR`. `os.makedirs` on write.
5. **Fail-closed sweep.** `niah_oracle_sweep.py` clears the sink, **forces decode
   steps** (`ignore_eos`, `decode_steps+1` tokens — DS selection is decode-only
   and NIAH prompts are immediate-EOS on raw `/generate`), then **asserts every
   issued trial produced records** and aborts on any missing length / hard
   failure. New `analyze_oracle.py` aggregates the sink into the budget-vs-scorer
   artifact with the uplift gate.
6. **serve script:** `RECALL_ORACLE=1` knob → `recall_oracle: true` in `DS_CONFIG`
   + auto-adds `--disable-cuda-graph`.

## Validation — binding GPU re-run (8×H200, DS int8 / mem 0.7 / TP=8, eager)
- **All three lengths recorded, fail-closed, N=20** (`[oracle-sweep] OK: all
  issued trials recorded; no hard failures`), ~4,880 (layer×decode-step) samples
  each. The previously silently-absent **64K is measured**.
- Budget-vs-scorer (`oracle_budget_vs_scorer_r4.json`, `m0_oracle_finding_r4.md`):
  - **4K budget-limited** — r@2048 44% → r@4096 86% → r@8192 100% (+56 pp).
  - **16K budget-partial** — 23% → 31% → 46% (+23 pp, caps ~46%): needs both a
    wider budget AND a better scorer.
  - **64K scorer-limited** — 15% → 20% → 24% (+9 pp): no feasible budget recovers.
- **342 DS unit tests pass** (fail-closed hook: no-trial/out-of-range/exception ⇒
  failure record; config-borne activation without env; validator recall_oracle
  guard; all prior scorer-variant/TP=8 tests).

## Decision impact
Confirms and sharpens the M0 A-vs-B decision: lead **Tier-2.B** (the only lever
for 64K and the binding lever for 16K); pursue **Tier-2.A** as a bounded win for
≤16K. Now binding at N=20 with 64K measured rather than inferred.

## Files changed
`config.py`, `oracle_artifact_sink.py`, `selection_kernel.py`, `selector.py`,
`validator.py`, `deepseek_v2.py`, `test_oracle_sink_and_force.py`,
`test_scorer_variants.py`, `niah_oracle_sweep.py`, `analyze_oracle.py` (new),
`serve_double_sparsity.sh`, `m0_oracle_finding{,_r4}.md`,
`oracle_budget_vs_scorer_r4.json`, `.gitignore`. Commit `bf2ce9b2b` (pushed).

## Remaining items (queued, justified)
- **AC-3 graph-safe Triton scorer port + full measurement matrix** (task #13):
  heavy kernel + GPU matrix (DSA same-node, N≥50 16K, MMLU at mem 0.7, dense-DS,
  within-budget parity, eager-vs-graph perf). Variants are correct + production-
  safe (R2/R3); the port + binding matrix is the next round's mainline.
- **Tier-2.A / AC-4** (task13–17), **M4 consolidation / AC-6** (task19–20):
  sequenced after AC-3 measurement.
- **Plan-marker code/comment cleanup**: pre-merge.

## BitLesson Delta
- Action: add
- Lesson ID(s): BL-20260602-ds-oracle-decode-only-and-shared-fs
- Notes: three stacked reasons the oracle recorded nothing despite config-borne +
  fail-closed (/dev/shm not shared across sandboxed processes; force-eager broke
  long-context DS via error_containment; prefill-only NIAH prompts do zero decode
  so decode-only DS selection never fires) — a genuinely reusable multi-round
  pitfall, captured with the verify-via-`double_sparsity`-meta method.

## Goal Tracker update request
- **Resolve Blocking Side Issue "oracle hook fail-open"** (flagged every review):
  fixed — fail-closed hook + config-borne activation + shared-FS + forced-decode
  sweep + expected-record assertions; 64K now measured.
- **task #12 (oracle fail-closed + 64K)** and **task #8 (oracle recording flaky)**
  → completed.
- **Keep Active**: task #13 (graph-safe Triton port + full AC-3 matrix) as the
  next round's mainline; AC-4 / M4 sequenced after.
