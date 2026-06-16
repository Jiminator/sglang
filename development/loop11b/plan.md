# Loop 11b — Finish loop 11: re-establish the op-point, settle the DS-vs-DSA SLO gap, productionize DS UX

## Goal Description
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

Scope note: "per SLOS.md" means the PRIMARY client workload only (GLM-5.1-FP8, 4096 ISL / 512 OSL,
conc 16–64, ~55% prefix hit). The SLOS.md DEFERRED requirement (128k ISL / 1024 OSL) is explicitly
OUT OF SCOPE for loop 11b.

## Acceptance Criteria

- AC-0: Op-point re-established on the fresh node.
  - AC-0.1: Channel mask regenerated for GLM-5.1-FP8 with a calibration preflight and a committed
    provenance record. The calibrate invocation MUST pass `--dtype bfloat16 --kv-cache-dtype
    fp8_e4m3` (`--dtype` is required by `calibrate.py`; `--kv-cache-dtype` is the metadata that must
    match serving), `--tp 8` (provenance-only; `device_map="auto"` does the sharding — no TP=8
    calibration semantics are implied), `--label-dim 16 --page-size 64 --num-samples 256
    --block-size 512 --seed 42`, and an explicit corpus choice (a committed local Pile-val corpus for
    determinism, or a recorded decision to stream `pile-val-backup`). Preflight: a `--dry-run-blocks
    1` placement/dtype check (fail-closed for the FP8 checkpoint) plus a free-GPU/disk check;
    calibration is treated as exclusive-node, long-running work with a recorded OOM/allocator policy.
    The provenance record commits: the exact command, corpus/token-block digest, model snapshot id,
    package + CUDA/driver versions, the tensor `content_sha256` AND the full-file `sha256`, and an
    explicit "inherited vs re-earned authorization" statement.
    - Positive: a `--dry-run-blocks 1` placement report passes; `calibrate.py` writes
      `glm51-fp8-channel-mask-*.safetensors`; a `provenance.json` records both hashes + the command +
      env; the regenerated mask serves and clears AC-5 recall.
    - Negative: a mask written with no provenance, without a passing dry-run placement check, or whose
      recall fails AC-5 comparability, is rejected (not used to serve).
  - AC-0.2: Radix-on is authorized for THIS boot's config WITHOUT the dev override. Per owner DEC-1,
    the validator's fixture authorization is changed to pin the mask by tensor-content hash
    (`channel_mask.compute_content_sha256`, `channel_mask.py:101-115`) plus the config fingerprint,
    REPLACING the full-file `_sha256_file` + path pinning (`validator.py:452-478`) — so a regenerated
    mask with identical tensor content authorizes without a re-mint regardless of file timestamp or
    path. (This is a shared-surface change → AC-7 DSA regression runs in the same round; the legacy
    full-file/path schema and the label-capture schema stay fail-closed-rejected.) The fixture is
    minted once for THIS op-point: boot radix-on under `SGLANG_DS_RADIX_OVERRIDE=1` (dev-only), re-run
    the DEC-12 table-free radix correctness probes (recall radix-on-vs-off ±0.5pp overall + per
    length, cross-rank selection identity, no dense fallback, clean production-reuse edge probe —
    boundary + partial@~2752 vs a length-matched radix-OFF control + eviction), then
    `validator.write_radix_fixture_state`, then boot AGAIN without the override to prove the artifact
    authorizes. `/server_info` (preferred; `/get_server_info` is the legacy alias) MUST confirm these
    exact keys: `model_path` (GLM-5.1-FP8 snapshot), `tp_size=8`, `page_size=64`,
    `kv_cache_dtype=fp8_e4m3`, `enable_double_sparsity=true`, `double_sparsity_config` (table-free,
    scorer_norm=off), `double_sparsity_radix_fixture_artifact` set, `disable_radix_cache=false`,
    `disable_cuda_graph=false`, `disable_custom_all_reduce=false`, `mem_fraction_static=0.8`,
    `max_running_requests=64`, `cuda_graph_max_bs=64`.
    - Positive: DS boots radix-on with the minted (content-hash) fixture and NO override; `/server_info`
      shows the keys above; the validator logs artifact authorization; a regenerated same-content mask
      at a different path/timestamp still authorizes.
    - Negative: a fixture whose content-hash or config fingerprint does not match the boot is refused;
      a tampered negative-control fixture is refused; radix-on without an authorizing artifact or
      override is refused; the legacy full-file/label-capture schemas are refused.
  - AC-0.3: Capacity reproduces on the fresh node — `/server_info`
    `internal_states[*].memory_usage.token_capacity` (with
    `internal_states[*].effective_max_running_requests_per_dp` captured) yields a derived decode-batch
    cap ≥ 64 at mem 0.8, CUDA graph capture succeeds on all buckets, no TokenLabelTable allocated,
    conc-64 running-req peak ≥ 61. (loop-11 references `token_capacity≈504640` / DSA `410560` are
    recorded as references, NOT exact hard gates.)
    - Positive: derived bs cap ≥ 64, capture OK, table not allocated.
    - Negative: bs cap < 64 at mem 0.8, any capture failure, or a table allocation is a hard stop.

- AC-2: Tail TTFT (HARD, judged at the locked sweep). DS pass/fail is the ABSOLUTE bar: DS P99 TTFT <
  22 s (SLOS) at conc 16/32/64, judged REGARDLESS of whether native DSA meets it (owner DEC-6). The
  DS/DSA P99 TTFT ratio per concurrency (comparator `AC11_TTFT_CEIL_RATIO` = 1.10×) is REPORTED as the
  competitive position, not the DS pass/fail gate. DS must not be admission-capped below nominal
  concurrency at conc ≤ 64.
  - Positive: DS P99 TTFT < 22 s at all three concurrencies; the report carries the DS/DSA ratio.
  - Negative: DS P99 TTFT ≥ 22 s at any conc fails (reported honestly); a 1-trial directional ladder
    presented as the AC-2 verdict is rejected (venue rule).

- AC-3: Throughput (HARD absolute per-request; ratios + aggregate reported). DS pass/fail is the
  ABSOLUTE bar: DS per-request decode-TPS p50 ≥ 30 (SLOS: output tokens/(latency−TTFT)) at conc
  16/32/64, judged REGARDLESS of DSA (DEC-6). REPORTED, not gated: the DS/DSA per-request p50 ratio
  (comparator `AC11_TPS_FLOOR_RATIO` = 0.95×) and aggregate total-tokens/s per conc (DS, DSA, ratio) —
  aggregate is descriptive, NOT a hard gate (owner DEC-3; the comparator does not gate aggregate).
  - Positive: DS decode-TPS p50 ≥ 30 at all three concurrencies; the report carries per-request ratio
    + aggregate.
  - Negative: DS decode-TPS p50 < 30 at any conc fails (reported honestly, not hidden).

- AC-4: Per-step tax guard (HARD; loop-11 task8). DS-vs-DSA same-batch one-batch decode window ratio ≤
  1.10 at bs64 (both mem 0.8, GRAPH mode), and the bs30 window ≤ 380k µs (loop-10 win intact). The
  guard declares radix state, sequence shape, warmup, and both mem fractions.
  - Positive: measured bs64 ratio ≤ 1.10; bs30 ≤ 380k µs; graph-mode numbers.
  - Negative: ratio > 1.10 at bs64, or bs30 > 380k µs, fails; an eager microbench presented as the
    guard is rejected.

- AC-5: Quality + no-op refusal. Recall@2048 within ±0.5pp fail-closed per landed change, measured
  against the frozen `loop9/runs/20260610_m0/recall_baseline.json` WITH comparability checks (matched
  length-set + per-length sample-count equality, per loop-11's `absorbed_recall_summary.py`; the
  frozen baseline is the owner-approved GLM-5.1 value-affecting quality reference — if the served fp8
  op-point is not comparable to it, a new served-fp8 baseline is defined and recorded rather than
  comparing across op-points). Cross-rank selection bit-identity HARD; radix value-equivalence
  re-verified when the fixture is minted. Plus an explicit DS no-op / dense-fallback refusal: every
  published SLO trial MUST carry `dense_fallback_total == 0` AND a positive sparse-selection proof
  (`selected_tokens_mean < total_tokens_mean`, or an equivalent counter); missing fields are an input
  refusal (the comparator only OBSERVES these, never refuses).
  - Positive: recall within bar on a comparable population; cross-rank identical;
    `dense_fallback_total == 0` and sparse-selection proven on all trials.
  - Negative: recall regression > 0.5pp (on a comparable population), cross-rank divergence, a nonzero
    or absent `dense_fallback_total`, or absent sparse-selection counters block publication.

- AC-6: DS concept intact. Offline channel mask → absorbed-latent signatures → query·signature scoring
  → top-k → sparse MLA decode. No dense fallback; no DSA-indexer substitution.
  - Positive: served path uses absorbed-latent table-free selection; mask consumed on the query side.
  - Negative: a dense-attention fallback or DSA-indexer reuse for DS selection fails.

- AC-7: DSA-native default un-regressed — strict. Shared-surface changes (memory accounting, radix
  plumbing incl. the DEC-1 validator content-hash change, graph runner, serve-script defaults) trigger
  the DSA regression in the same round; the shipped DSA default is unchanged in behavior and capacity.
  - Positive: DSA-native boots/captures; DSA `memory_usage.token_capacity` unchanged vs its own
    reference on this node; frozen Case-2 recipe matches.
  - Negative: any DSA-native capacity or behavior regression fails the round.

- AC-8: Protocol / ledger / queue / evidence discipline. `queue.md` current every round; evidence
  pre-flight before each handoff (artifact exists + tracked + claim matches POST-commit state);
  `results.md` rewritten-over-append; one-trial honesty until the locked sweep; frozen references
  intact; `git push` every round; AND the close-out preserves raw evidence under a stable naming
  convention — `provenance.json` (mask), a run-order ledger, per-trial reuse summaries, bench JSONLs +
  sidecars, server logs per boot, the fixture, mask hashes, `/server_info` snapshots, and the exact
  benchmark commands.
  - Positive: a reviewer reproduces the verdict from committed artifacts alone.
  - Negative: a claim without a committed/tracked artifact, or a layered (non-rewritten) ledger, fails
    pre-flight.

- AC-9: Measurement honesty / op-point fairness. The locked sweep publishes TWO declared comparisons
  (owner DEC-2): (a) the PRODUCTION-ENVELOPE comparison — DS mem 0.8 vs DSA mem 0.85, each at its own
  served operating point (the comparator legitimately allows the mem asymmetry but enforces
  radix-state match + the locked flag set); and (b) a SAME-MEMORY sensitivity comparison — DS and DSA
  both at mem 0.8 — to isolate the algorithm from the memory envelope. Both columns radix-ON, matched
  workload / seed-family. Trial structure is characterized truthfully: the sweep runs TWO trials per
  concurrency at the SAME per-conc seed (owner DEC-4) — these are repeated measurements for
  run-to-run stability, NOT independent samples; report min/median/max (NOTE: the comparator's
  `--ac11` mode currently hard-requires ≥3 trials, so its trial-count floor is lowered to 2 as an
  in-scope tooling tweak, with the repeated-measurement framing recorded). Fresh-node drift is
  controlled with an ENFORCEABLE one-server run order: per concurrency, alternate DSA/DS boots by
  trial with a logged boot order, retained boot/server logs, a `/server_info` snapshot per boot, and
  thermal/clock notes; if runs are block-scheduled by side (all DSA then all DS) they are LABELED
  unpaired and require explicit acceptance before a HARD verdict. Observed prefix-reuse / cached-token
  distribution is recorded per SLO trial (DEC-12 authorizes only production-representative reuse; GSP
  shape alone does not prove it). Native DSA is EXPECTED to possibly miss the absolute SLO on this
  node, and DS may miss it too; per owner DEC-6 the DS verdict is judged on the absolute SLO
  regardless of whether DSA passes (the op-point is NOT invalidated by a DSA miss).
  - Positive: comparator accepts each matched op-point; both comparisons published; per-trial reuse
    recorded; run order logged.
  - Negative: an unmatched op-point, an unlabeled block-scheduled order presented as paired, or a
    verdict published without per-trial reuse evidence is refused.

- AC-UX: Production enablement documented and stale references fixed (no ABI change; owner DEC-5).
  - AC-UX.1: A runbook takes a GLM-5.1-FP8 operator from zero to a serving DS server (calibrate +
    preflight → boot under override → DEC-12 probes → mint fixture → serve without override),
    explaining the fixture-fingerprint flow once and labeling which env knobs are production vs
    diagnostic. The runbook prefers `/server_info`.
  - AC-UX.2: Category-A fixes land across scripts AND benchmark/comparator text — repoint
    `serve_double_sparsity.sh`/`serve_native_nsa.sh` MODEL_PATH and CHANNEL_MASK_PATH to GLM-5.1-FP8;
    correct the mem-0.6 / TokenLabelTable comments (table deleted; mem 0.8); reconcile the loop8
    throughput warning to the loop11b verdict; de-DeepSeek the `calibrate.py` docstring/argparse and
    the `config.py` top_k comment; update stale `benchmark.sh` / `benchmark_baseline.sh` /
    `benchmark_compare.py` comments, the "independent trials" wording, and any `CLIENT_SLOS.md`
    references (now `SLOS.md`).
  - AC-UX.3: Category-B CLI help text updated — `server_args.py:6090` no longer says "for
    DeepSeek-V3.2 (FP8)"; the `:6103` example path is GLM-appropriate. NO flag rename or JSON-config
    schema change (Category C) is made (owner DEC-5: the inline `--double-sparsity-config` JSON is NOT
    collapsed into flags this loop).
    - Positive: `--help` and serve scripts describe the GLM-5.1 table-free reality; a fresh operator
      following the runbook reaches a radix-on DS server.
    - Negative: any stale DeepSeek/dsv32/TokenLabelTable/mem-0.6/CLIENT_SLOS reference remaining in the
      GLM production path fails; any CLI flag rename or JSON-config schema change fails.

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)
All of: op-point re-established with full provenance + a content-hash radix authorization (DEC-1);
the AC-4 tax guard and the locked 2×600s sweep run, producing AC-2/AC-3/AC-4 HARD verdicts at BOTH
the production-envelope and same-memory op-points (DEC-2); the headline DS-vs-DSA report with per-conc
decode-TPS, P99 TTFT, aggregate throughput, achieved concurrency, observed reuse, and DS/DSA ratios;
the production runbook plus all Category-A and Category-B UX fixes; close-out evidence preserved and
`results.md` regenerated. If AC-4 runs tight, ONE parked exact reducer (q4 fuse top-k emit with the
gather, or q5 bf16-primary scratch) may be pulled in, profile-justified.

### Lower Bound (Minimum Acceptable Scope)
Op-point re-established (mask regenerated with provenance, radix-on minted and authorized via the
content-hash fixture, bs cap ≥ 64, AC-7 clean); the locked sweep run once producing the honest
AC-2/AC-3 ABSOLUTE-bar verdict (pass or fail) with DSA measured radix-ON on the same node, at least
the production-envelope op-point (same-memory comparison may be deferred only if explicitly recorded);
AC-4 measured and reported (reducer optional); the stale serve-script model/mask defaults fixed and
the loop8 warning reconciled. A documented FAIL is a complete result. CRITICAL: radix-on authorization
is a PRECONDITION of the HARD AC-2/AC-3 DS-vs-DSA verdict (AC-9 requires both sides radix-ON). If
radix-on CANNOT be minted, the loop reports "no publishable radix-on verdict" and any radix-OFF numbers
are explicitly DIAGNOSTIC ONLY — not a substitute verdict.

### Allowed Choices
- Can use: the existing tooling (`calibrate.py` + `--dry-run-blocks` preflight,
  `validator.write_radix_fixture_state`, the DEC-12 edge-probe harness, `benchmark.sh` /
  `benchmark_baseline.sh` / `benchmark_compare.py --ac11`, the loop-11 capacity/tax drivers); mint the
  radix fixture; the owner-approved validator content-hash change (DEC-1); the owner-approved
  comparator trial-floor lowering to 2 (DEC-4); edit docs/comments/script-defaults/CLI-help text.
- Cannot use (without further owner approval): any CLI flag rename or `--double-sparsity-config`
  JSON-schema change (DEC-5); any change to the shipped DSA-native default; re-running frozen
  references; the deferred 128k-ISL SLO.

## Feasibility Hints and Suggestions

### Conceptual Approach
1. Repoint MODEL_PATH to `/cluster-storage/models/models--zai-org--GLM-5.1-FP8/snapshots/f396cf8…`.
2. Preflight + regenerate the mask: `python -m sglang.srt.layers.attention.double_sparsity.calibrate
   --model <GLM-5.1-FP8> --dtype bfloat16 --kv-cache-dtype fp8_e4m3 --tp 8 --label-dim 16 --page-size
   64 --num-samples 256 --block-size 512 --seed 42 --dataset <corpus> --dry-run-blocks 1 --output
   <mask>` (dry-run first, then the full run); commit `provenance.json` (both hashes + command + env).
3. Land the DEC-1 validator content-hash change (+ AC-7 regression). Boot DS radix-on under
   `SGLANG_DS_RADIX_OVERRIDE=1`, run the DEC-12 edge probes, `write_radix_fixture_state`, then boot
   again WITHOUT the override and confirm `/server_info`.
4. Capacity/AC-7 reconfirm; then the bs64 tax guard (loop-11 task8); then the locked sweep (loop-11
   task9 — 2 trials/conc, paired DS/DSA, BOTH the production-envelope and same-memory comparisons,
   reuse + fallback recorded); then the headline report; then the UX pass; close out.

### Relevant References
- `calibrate.py` (`--dtype` required; `--dry-run-blocks` fail-closed FP8 placement gate;
  `_enforce_dry_run_placement`), `channel_mask.py` (`save_channel_mask` embeds `created_at`;
  `compute_content_sha256` is the tensor-only hash adopted by DEC-1), `validator.py` (`_sha256_file`,
  `radix_fixture_config_fingerprint`, `apply_radix_fixture_artifact` — the content-hash change site).
- `development/serve_double_sparsity.sh`, `serve_native_nsa.sh`, `benchmark.sh`,
  `benchmark_baseline.sh`, `benchmark_compare.py` (`--ac11`: per-request TPS floor 0.95, TTFT ceil
  1.10, absolute 30/22, radix-state match enforced, mem asymmetry allowed, no aggregate/no-op gate,
  ≥3-trial floor to lower), `SLOS.md`, `server_args.py:6090/6103` (CLI help), `/server_info`
  (`http_server.py:631`), `scheduler.py:3337/3343` (`memory_usage.token_capacity`,
  `effective_max_running_requests_per_dp`).
- `development/loop11/{queue.md,results.md}` (DEC-1…DEC-12), frozen
  `loop9/runs/20260610_m0/recall_baseline.json`, loop-11 `absorbed_recall_summary.py` (comparability).

## Dependencies and Sequence

### Milestones
1. M-A Op-point re-establishment (AC-0, AC-7): model path → mask preflight + regen + provenance →
   DEC-1 validator content-hash change → radix mint (DEC-12 probes) → capacity + AC-7 reconfirm. Gates
   everything.
2. M-B M4 close (AC-4, AC-2, AC-3, AC-9): bs64 tax guard (loop-11 task8) → locked sweep (loop-11
   task9, 2 trials/conc, paired DS/DSA, production-envelope + same-memory, reuse recorded) → headline
   DS-vs-DSA report.
3. M-C Productionize (AC-UX): runbook + Category-A/B fixes; reconcile the throughput warning.
4. Close-out (AC-8): `results.md` regenerated, `queue.md` final, evidence preserved, pushed.

Dependencies: M-A precedes everything. The tax guard and the locked sweep both bind to the M-A
capacity config. The headline report depends on the sweep. The UX warning reconciliation depends on
the verdict. AC-7/AC-8/AC-9 discipline runs every round.

## Task Breakdown

| Task ID | Description | Target AC | Tag | Depends On |
|---------|-------------|-----------|-----|------------|
| task1 | Populate `queue.md` from this plan at kickoff (single source of truth) | AC-8 | coding | - |
| task2 | Repoint serve-script MODEL_PATH/CHANNEL_MASK_PATH to GLM-5.1-FP8; confirm model snapshot present | AC-0.1, AC-UX.2 | coding | task1 |
| task3 | Pre-sweep methodology review (analyze): comparator gaps (aggregate descriptive vs gated, no-op/dense-fallback refusal, mem-asymmetry fairness, ≥3-trial floor → 2), two-trial repeated-measurement framing, per-trial reuse capture, one-server run-order/drift controls, recall-baseline comparability, the same-memory comparison design | AC-3, AC-5, AC-9 | analyze | task1 |
| task4 | Regenerate GLM-5.1 channel mask: `--dry-run-blocks 1` preflight + full calibrate + commit `provenance.json` (both hashes, command, env, recall-comparability check vs frozen baseline) | AC-0.1 | coding | task2 |
| task5 | DEC-1 validator content-hash authorization change (+ AC-7 regression); then mint radix-on: DEC-12 edge probes under dev override → `write_radix_fixture_state` → prove no-override boot authorizes (`/server_info` keys) + same-content-different-path re-auth; negative-control + legacy-schema refused | AC-0.2, AC-5, AC-7 | coding | task4 |
| task6 | Capacity + AC-7 reconfirm on fresh node (derived bs cap ≥64 @0.8 via `memory_usage.token_capacity`, capture, no table; DSA-native un-regressed) | AC-0.3, AC-7 | coding | task5 |
| task7 | loop-11 task8 — per-step tax guard at bs64 (same-batch DS-vs-DSA window, graph mode); conditional q4/q5 reducer only on a profile-proven overage | AC-4 | coding | task6 |
| task8 | loop-11 task9 — locked sweep (2 trials/conc same seed, conc 16/32/64), paired DS/DSA radix-ON with logged run order, BOTH production-envelope (DS0.8/DSA0.85) and same-memory (both 0.8) comparisons, per-trial reuse + `dense_fallback_total` recorded; comparator trial-floor lowered to 2; produce AC-2/AC-3 absolute-bar verdicts | AC-2, AC-3, AC-5, AC-9 | coding | task6 |
| task9 | Headline DS-vs-DSA end-to-end report per SLOS.md primary workload (decode-TPS, P99 TTFT, aggregate, achieved conc, reuse, DS/DSA ratios at both op-points + honest absolute-bar verdict) | AC-2, AC-3 | coding | task8 |
| task10 | Production UX pass: runbook + Category-A fixes (scripts + benchmark/comparator text + CLIENT_SLOS) + Category-B CLI help text; reconcile loop8 warning to verdict; NO ABI change | AC-UX | coding | task9 |
| task11 | Close-out: regenerate `results.md`, finalize `queue.md`, evidence pre-flight (provenance, run-order ledger, reuse summaries), push | AC-8 | coding | task9, task10 |

## Claude-Codex Deliberation

### Agreements (Codex pass 1 + pass 2 + pass 3, verified in-repo)
- Mask full-file SHA cannot be reproduced (embedded `created_at`) → re-mint needed; owner adopted the
  tensor-content hash (DEC-1) so future masks are portable.
- The comparator gates per-request decode-TPS p50 + P99 TTFT + absolute 30/22, NOT aggregate
  throughput, and only OBSERVES dense-fallback → AC-3 aggregate is descriptive and AC-5 no-op needs an
  explicit refusal.
- `benchmark.sh` reuses the same per-conc seed across trials ⇒ repeated measurements; one TP=8 server
  at a time forces an explicit, logged run order (paired-by-trial or labeled-unpaired).
- mem_fraction asymmetry is comparator-sanctioned ⇒ the production-envelope comparison is legitimate
  (DEC-2 adds a same-memory sensitivity run alongside it).
- UX has zero required ABI changes; Category A safe, Category B help-text only (DEC-5).
- Capacity is read from `internal_states[*].memory_usage.token_capacity` +
  `effective_max_running_requests_per_dp` (`scheduler.py:3337/3343`).

### Resolved Disagreements
- Inherited-vs-re-earned authorization: resolved by code evidence (re-mint) + owner DEC-1 (switch to
  tensor-content hash so the mask is portable across runs/paths).
- Capacity exactness: softened from exact 504640/410560 to bs cap ≥64 + capture + no-DSA-regression.
- Lower-bound radix-OFF inconsistency (pass-2): fixed — radix-on authorization is a precondition of
  the HARD verdict; radix-OFF is diagnostic-only.
- Calibration command/preflight (pass-2): fixed — `--dtype` required, `--dry-run-blocks 1` preflight,
  corpus/env/version capture.
- Vague server-info / no-op checks (pass-2 + pass-3): fixed — exact `/server_info` keys (corrected to
  `memory_usage.token_capacity`) + a sparse-selection proof requirement.
- Recall comparability (pass-2): fixed — frozen baseline kept WITH comparability checks, else a
  served-fp8 baseline is defined.

### Convergence Status
- Final Status: **converged** (2 convergence rounds; the sole round-2 required change — a server-info
  field path — was code-verified and incorporated; no new risks; DEC set confirmed complete). All six
  DEC items are resolved by the owner below.

## Pending User Decisions

- DEC-1: Mask authorization mechanism. **DECIDED — switch the validator to pin the tensor-content hash
  (`compute_content_sha256`) + config fingerprint, replacing full-file-SHA + path pinning, so future
  masks are portable/reproducible.** (Still mint the fixture once for this op-point.) Decision Status:
  RESOLVED (owner).
- DEC-2: Headline op-point. **DECIDED — publish BOTH the production-envelope comparison (DS 0.8 / DSA
  0.85) AND a same-memory sensitivity comparison (both at mem 0.8).** Decision Status: RESOLVED (owner).
- DEC-3: The 30 TPS bar and aggregate. **DECIDED — judge on per-request median decode-TPS (the
  comparator's enforced metric, matching SLOS's output-tokens/(latency−TTFT)); aggregate total-tokens/s
  is REPORTED, not a hard gate.** Decision Status: RESOLVED (owner).
- DEC-4: Trial design. **DECIDED — run TWO trials per concurrency at the SAME per-conc seed, framed as
  repeated run-to-run-stability measurements (report min/median/max). Lower the comparator's ≥3-trial
  `--ac11` floor to 2 as an in-scope tooling tweak.** Decision Status: RESOLVED (owner).
- DEC-5: UX/ABI scope. **DECIDED — Category A (doc/comment/default/benchmark text) + Category B (CLI
  help text) ONLY; NO CLI flag / JSON-config schema change (the JSON blob is NOT collapsed into flags
  this loop).** Decision Status: RESOLVED (owner).
- DEC-6: DSA-miss handling. **DECIDED — judge DS on the absolute SLO (30 TPS / P99 TTFT < 22 s)
  regardless of whether native DSA passes; a DSA miss does NOT invalidate the op-point. The DS/DSA
  ratios are reported as the competitive position.** Decision Status: RESOLVED (owner).

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-workflow terminology ("AC-", "Milestone",
  "Phase", "Step", "task", "DEC-", "loop 11b"). Those live in this plan only.
- Serve-script/runbook/CLI-help edits use domain language (model name, mask path, mem fraction, radix
  fixture), not plan markers.

--- Original Design Draft Start ---

# Loop 11b Draft — Finish loop 11: re-establish the op-point on fresh hardware, settle the DS-vs-DSA SLO gap, clean up the DS production UX

> Written 2026-06-16, after **loop 11 was terminated mid-flight**. loop 11 landed its whole
> structural build (M0–M3, task0–task7): the per-rank `TokenLabelTable` is **deleted** (DEC-2),
> **table-free absorbed-latent scoring is the one served DS selection path**, radix-on is
> **authorized + re-enabled** under owner DEC-12 (production-representative-reuse edge contract),
> and the capacity payoff was proven live on the (now released) loop-11 node:
> `max_total_num_tokens` 504,640 / decode-batch cap **109** at mem 0.8. **What loop 11 never did**
> is its M4 *verdict* milestone — task8 (per-step tax guard at bs64) and task9 (the locked AC-11
> sweep). So the loop produced the fix but never produced the **measured answer**: does the
> capacity win actually close the serving SLO gap vs native DSA, or not. **Two things changed
> since:** (1) the loop ran out before M4, and (2) **the machine that held the op-point artifacts
> was released** — the GLM-5.1 channel mask under `/models/` is gone. This loop re-establishes the
> op-point on fresh hardware, runs the two pending measurement tasks to a HARD verdict, delivers the
> client-facing DS-vs-DSA end-to-end comparison per `development/SLOS.md`, and uses the occasion to
> make the DS production/user experience something a human can actually follow.
> Feed this through `gen-plan` once scope is confirmed.

---

## What this work is (and is NOT) — read first

**This is not a new optimization loop, and not frontier development.** The build is done; loop 11
did it. This is **finish + validate + productionize**. Same honesty posture as loops 10/11: Double
Sparsity is a **2-year-old paper** (arXiv:2408.07092) reproduced on an open-source codebase
(SGLang) against a model that already ships a *trained* sparse indexer (DSA). The algorithm has
**no theoretical hope of beating DSA** — the point is the engineering and the honest number.

This loop does exactly three things:

1. **Re-establish the serving op-point on the new 8×H200 node.** The released machine held the
   calibrated GLM-5.1 channel mask (`/models/…`) and the live servers. Before any number is
   trustworthy we must regenerate the mask, re-validate (or re-mint) the radix-on authorization
   fixture against this boot's config, and re-confirm capacity/AC-7 reproduce here.
2. **Close M4 — the two pending measurement tasks** (task8 AC-4 per-step tax guard, task9 the
   locked AC-11 sweep) → the loop's actual **AC-2 / AC-3 / AC-4 HARD verdicts**.
3. **Deliver the headline end-to-end DS-vs-DSA comparison** following `development/SLOS.md`, and
   **revisit the DS production/user experience** so enabling DS on GLM-5.1 is a short, documented,
   reproducible path instead of a wall of dev-only env knobs and stale defaults.

**Honest-verdict posture (owner, carried forward):** a **FAIL** on the throughput SLO is an
acceptable, expected, reportable outcome. The deliverable is the *measured gap*, not a DS win. The
loop is done when the number exists and is honestly characterized — pass or fail.

## Objective

Produce the verdict loop 11 never reached: **one locked, honest, apples-to-apples end-to-end
comparison of table-free DS vs native DSA (radix-ON both) on the GLM-5.1-FP8 client SLOs**, on a
*freshly re-established* op-point, plus a production-usable DS enablement path.

Concretely, against `development/SLOS.md` (GLM-5.1-FP8, **30 TPS decode floor**, **P99 TTFT < 22 s**,
4096 ISL / 512 OSL, concurrency 16–64, ~55% prefix hit):

1. **Re-establish the op-point** (channel mask + radix fixture + capacity) on the new node — the
   released-machine artifacts are gone, so this is a precondition, not a formality.
2. **Close M4**: AC-4 per-step tax guard at bs64 (task8) + the locked AC-11 sweep (task9) →
   AC-2/AC-3 HARD verdicts.
3. **The headline**: DS-vs-DSA end-to-end — does DS meet 30 TPS / P99 TTFT < 22 s, and what is the
   residual gap to DSA at the client op-point.
4. **Clean up the DS production UX**: stale `MODEL_PATH`/`CHANNEL_MASK_PATH` defaults, the radix
   fixture-fingerprint dance, the dev-only knob sprawl, and the now-contradicted throughput warning.

**Scope (per user):** this loop is deliberately **narrow** — it finishes and validates an existing
build and tidies its rough operational edges. Refactoring is welcome where it directly serves the
production-UX goal; it is **not** an invitation to reopen the scoring/selection pipeline (that
landed and is gated). New optimization ideas are queue-feeding candidates only if the headline
exposes a specific, cheap, attributable gap — never a reason to expand into a fresh build.

---

## Current state (HEAD `cd2d1e7c1`, branch `dev/double-sparsity-standalone`)

### Where loop 11 stopped

| milestone | tasks | status |
|---|---|---|
| M0 ground truth | task0–task2 | ✅ DONE |
| M1 capacity floor | task3, task4 | ✅ VERIFIED |
| M2 table-free | task5, task6 | ✅ VERIFIED (TokenLabelTable DELETED, DEC-2) |
| M3 radix-on | task7 | ✅ COMPLETE (DEC-12 authorized + reproducible) |
| **M4 verdict** | **task8 (tax guard), task9 (locked sweep)** | ❌ **never started — the loop was terminated here** |

The last four loop-11 rounds (R26–R29) were all task7 authorization repair + a handoff cleanup;
R29 (`cd2d1e7c1`) is a pure bookkeeping round. No M4 measurement exists.

### The validated config is pinned — and partly stranded by the machine change

`development/serve_double_sparsity_radix_fixture.json` (schema `ds_radix_fixture_state_tablefree_v2`)
records the exact config the loop-11 radix-on authorization was earned against:

```
model_path          = /cluster-storage/models/models--zai-org--GLM-5.1-FP8/snapshots/f396cf8…  (PRESENT on new node ✓)
tp_size=8  page_size=64  kv_cache_dtype=fp8_e4m3  selector_mode=table_free
channel_mask_path   = /models/glm51-fp8-channel-mask-s256.safetensors                          (GONE — /models does not exist ✗)
channel_mask_sha256 = 340b6c0bd40d6d7dcf4fef5ab2a9f096cf56d94bc844960500bcd72db65f08bd          (the fingerprint the validator checks)
```

**Survived the machine move:** the repo (code, committed loop-11 run artifacts, the fixture JSON,
the frozen recall baseline `development/loop9/runs/20260610_m0/recall_baseline.json`). The GLM-5.1
weights are present at the exact snapshot path the fixture names.

**Did NOT survive:** `/models/` (the channel mask file), the live DS/DSA servers, and the running
server behind loop-11's frozen serving ladder (its numbers are committed; the server is gone).

**New hardware:** 8×H200 (143,771 MiB/GPU), TP=8 — **same class** as loop 11. The physical node is
different; treat every loop-11 *live* number as needing re-confirmation, not as ground truth.

### The contradiction the headline must settle

- `serve_double_sparsity.sh` ends with a stern warning (citing `loop8/task9_gate_results.md`): DS
  decode-TPS **~23/17/17 tok/s ≪ 30** at conc 16/32/64 — *"does NOT meet the throughput SLO… enable
  DS only for long-context recall, not to raise standard-workload throughput."*
- loop 11's directional ladder (`development/loop11/draft.md`) showed DS decode-TPS p50
  **39.4 / 33.0 / 33.3** — *above* 30 — after the loop-10 per-step win and the loop-11 cap lift.

These disagree. loop 11 never ran the locked sweep that would adjudicate. **Until the AC-11 sweep
runs on the re-established op-point, the repo does not actually know whether table-free DS meets the
client SLO.** That is this loop's headline question, and the serve-script guidance is unshippable
until it is answered.

---

## The intermediate steps — why this is not just "run the benchmark"

The op-point must be rebuilt before a single measurement counts. This is the part the user
flagged ("regenerating the mask again"):

1. **Model path.** Confirmed present (snapshot `f396cf8…`). Both serve scripts default `MODEL_PATH`
   to a **stale DeepSeek-V3.2 path**; repoint to the GLM-5.1-FP8 snapshot. (UX item, but also a
   correctness precondition — the fixture's `model_path` must match the boot.)

2. **Channel mask regeneration.** Table-free DS **still consumes the offline channel mask**
   (`S_h`, `w_c`) on the *query* side — absorbed-latent removed the per-token *table*, not the
   *mask*. The mask file is gone. Regenerate with
   `python -m sglang.srt.layers.attention.double_sparsity.calibrate` (Method-1
   `mean(abs(Q_nope·K_nope))`, Pile-val 256×512, `--seed 42 --label-dim 16 --page-size 64
   --kv-cache-dtype fp8_e4m3`) against GLM-5.1-FP8, writing the path the fixture names.
   **Gate:** the calibrator is seeded + fixed-corpus → it *should* be byte-reproducible. If the
   regenerated mask's SHA-256 reproduces `340b6c0b…`, the committed radix fixture's fingerprint
   still holds and the radix-on authorization is **inherited**. If it does not (corpus
   unrecoverable, or any nondeterminism), the fingerprint mismatches and radix-on is **refused**
   (fail-closed by design) → go to step 3b.

3. **Radix fixture re-validation / re-mint.**
   - **3a (mask SHA matches):** boot DS radix-on with the committed fixture; confirm the validator
     accepts it for this boot's config. Authorization inherited; no re-probe needed.
   - **3b (mask SHA differs):** re-run the DEC-12 table-free radix correctness probes on the served
     workload — recall@2048 radix-on-vs-off within ±0.5pp overall and per length, cross-rank
     selection identity, no dense fallback, and a clean **production-reuse** edge probe (boundary
     page-aligned reuse + partial-page hit at the production upper bound cached ~2752 ≈ 63%, each
     within ±0.5pp of a length-matched radix-OFF control, + clean eviction/recompute; near-full
     reuse recorded out-of-contract, not a gate input) — then `validator.write_radix_fixture_state`
     to mint a fresh fixture. Recreate `/models/` (or repoint `CHANNEL_MASK_PATH`).

4. **Capacity + AC-7 re-confirm on fresh hardware.** Boot DS table-free radix-on: confirm loop-11's
   live capacity reproduces here (`max_total_num_tokens` ≈ 504k, bs cap ≈ 109 ≥ 64, CUDA-graph
   capture on all buckets, no table allocated, conc-64 running-req peak ≥ 61). Boot the DSA
   baseline. This both unblocks M4 and re-validates AC-1/AC-7 on the new node.

**Frozen-reference note (resolves "the machines were released"):** loop 11's task1 froze a DSA
radix-ON *directional* ladder on the released node. The closing verdict requires DS and DSA on the
**same machine in the same session** (`benchmark_compare.py` enforces a matched op-point), so the
locked AC-11 sweep **re-measures DSA here**. The loop-11 frozen serving ladder is demoted to a
directional reference only — it is not the verdict bar. The frozen *recall* baseline
(`recall_baseline.json`, a quality artifact) survives the move and stays the recall gate reference.

---

## Candidate menu (ranked — `gen-plan` sequences; each is one regenerate/implement → measure cycle)

0. **Op-point re-establishment (DO FIRST — gates everything).** Steps 1–4 above. **Verdict gate:**
   mask SHA reproduced (or fixture re-minted), validator permits radix-on for this boot, bs cap ≥ 64
   @ mem 0.8 reproduced, DSA-native default un-regressed (Case-2 / DSA@0.8 = 410560).
1. **task8 — per-step tax guard at bs64 (AC-4).** Same-batch DS-vs-DSA one-batch decode window at
   mem 0.8, capture-ladder extension to bs64, top-k/score-kernel scaling re-check. **Gate:** ratio
   ≤ ~1.10 at bs64; bs30 window ≤ ~380k µs (the loop-10 win is not traded for capacity). *Coding
   tail is conditional:* only if the guard runs tight, pull in a parked exact reducer (q4 fuse radix
   top-k emit with the logical→physical gather; q5 bf16-primary score scratch) — measure first,
   build only on a proven overage.
2. **task9 — locked AC-11 sweep + close-out (AC-2/AC-3).** `benchmark.sh` (DS) +
   `benchmark_baseline.sh` (DSA radix-ON) + `benchmark_compare.py --ac11`, 3 trials × 600 s, conc
   16/32/64, the `SLOS.md` workload. **Verdicts:** AC-2 P99 TTFT < 22 s and ≤ ~1.10× DSA per conc;
   AC-3 decode-TPS p50 ≥ 30 and aggregate ≥ ~0.95× DSA. Regenerate `results.md` as one coherent
   close-out (rewrite-over-append).
3. **The headline end-to-end report — the client-facing answer.** Extracted from task9: a single
   DS-vs-DSA table on the `SLOS.md` SLOs (decode TPS, P99 TTFT, aggregate, achieved concurrency per
   conc), the measured gap, and the honest verdict — meets / misses, by how much, and where the cost
   sits. This is the loop's reason to exist; it also retires or rewrites the stale serve-script
   throughput warning to match measured reality.
4. **Production / UX pass.** Reduce DS enablement to a short path: **choose model → one calibrate
   command → serve with sensible defaults.** Concretely: fix the stale `MODEL_PATH` /
   `CHANNEL_MASK_PATH` defaults (DeepSeek-V3.2 → GLM-5.1-FP8); write a canonical runbook (calibrate
   → boot → fixture → serve, with the fixture-fingerprint flow explained once); audit the dev-only
   env-knob sprawl (`RECALL_ORACLE`, `LIFTED_BUDGET`, `SCORER_NORM`, `ANCHOR_*`, `RADIX_FIXTURE_*`,
   `SGLANG_DS_RADIX_OVERRIDE`) and mark clearly which are production vs diagnostic; reconcile the
   throughput guidance with the loop11b verdict. **Surgical: doc + defaults + guidance.** Do NOT
   refactor the `--double-sparsity-config` JSON ABI into CLI flags unless gen-plan explicitly scopes
   it (that is a userspace change — see Pending decisions).
5. **WILDCARD — only if the headline exposes a specific, cheap, attributable gap.** A parked
   per-step reducer (q4/q5) or an admission/scheduling lever, justified by a measured trace. Not a
   license to start a new optimization program; loop11b is finish + validate.

Ideas found while working that beat these **replace them** under the same gates — but the menu is
short on purpose. The structural work is already banked.

---

## Open scope + the task queue (`development/loop11b/queue.md`)

Same discipline as loop 11:

- **Populating `queue.md` is the FIRST task of the loop** (after plan refinement, at kickoff) —
  seed from the refined plan plus any kickoff ideas. Do NOT populate during plan generation.
- Every task gets: id, description, targeted quantity (mask SHA / bs cap / tax ratio / TTFT-p99 /
  TPS), expected effect, lossiness posture, compatibility note vs already-landed loop-11 changes,
  status. Mid-loop ideas **append** with a one-line compatibility note; dropped/superseded tasks
  stay listed with the measured cause — no silent deletions.
- A task is completed only after its gates pass (quality teeth + the relevant measurement). The
  queue is committed every round.

`queue.md` is created empty at kickoff and is the single source of truth for what is planned, in
flight, done, or dropped.

---

## Subagent usage (context discipline — MANDATORY, carried from loops 9/10/11)

The main agent's context is the scarcest resource; this loop reads serve logs, calibration logs,
bench JSONL ladders, and fixture logs — none of which belong raw in the main context. **Delegate by
default; the main context holds decisions, verdicts, and the queue — not artifacts.**

- **Code reconnaissance** (where the mask is bound, the validator's fingerprint check, the config
  plumbing) → Explore subagents; only `file:line` + one-paragraph mechanism returns.
- **Artifact digestion** (calibration provenance check + mask SHA verify, serve-log capacity/
  headroom readout, bench JSONL ladder extraction, fixture-validation logs) → analysis subagents
  that return the numbers table, never the raw dump.
- **Well-scoped slices** (the calibration driver, a fixture re-mint runner, the one-batch tax
  driver) → implementation subagents with a tight contract.
- **Long-running measurement babysitting** (calibration forward pass; boot→sweep→teardown) → run
  detached; the main agent reads the terminal marker + summary file, not the stream.
- **Document drafting** (round summaries, the close-out, the UX runbook first draft) → subagent
  draft, main-agent reviewed before commit.

Two hard rules (unchanged): every subagent product is reviewed in full by the main agent before it
is trusted; nothing a subagent produced lands without passing the **same** verification gates as
main-agent work. Subagents save context, not review.

---

## The iterate→measure protocol (loop-9/10/11 heartbeat, reused)

For **each** queue task, exactly one cycle, run context-lean:

1. **Implement / regenerate** (op-point artifact, config default, or measurement driver). Shared
   surfaces (memory accounting, radix, graph runner, validator) trigger the stricter AC-7 DSA
   regression in the same round.
2. **Verify quality FIRST (the teeth):**
   a. **NIAH recall@2048** fail-closed ±0.5pp vs the frozen
      `loop9/runs/20260610_m0/recall_baseline.json` (after any mask regen / fixture re-mint).
   b. **Cross-rank selection bit-identity** (hard, every gate run).
   c. **Radix correctness** (value-equivalence per DEC-12) re-verified if the fixture is re-minted.
   d. **Mask provenance**: regenerated mask SHA recorded and compared to the fixture fingerprint;
      inherited vs re-earned authorization stated explicitly.
3. **Measure the loop's quantities (cheap → expensive):**
   a. **Capacity probe** — boot + `max_total_num_tokens` readout → derived bs cap (tightest signal).
   b. **One-batch kernel guard** — 10-step decode window, DS vs DSA same-batch, ratio ≤ ~1.10
      (protects the loop-10 win).
   c. **Targeted serving spot-check** — conc-64 (and 32 when TTFT-relevant) directional run, 1
      trial, against the freshly-measured DSA column.
   d. **Full 3-concurrency ladder** only at milestone gates. **Locked AC-11 sweep (3 trials ×
      600 s) once, at loop close**, as the publication artifact.
4. **Read the gap** against AC quantities (capacity, TTFT p99/conc, aggregate, tax ratio, decode
   TPS). A shifted bottleneck is a queue-feeding event, not scope creep.
5. **Keep or revert.** Ledger `development/loop11b/results.md` — **rewrite-over-append**, one
   authoritative current-state section.

Measurement discipline carried forward: **one TP=8 server at a time**; the frozen recall baseline is
never re-run; serving numbers are **1-trial directional until the closing AC-11 sweep** (no
SLO-pass claim beyond what the trial count supports); **graph-mode numbers are binding** (eager
microbenches lie); **`git push` at every round boundary** — doubly important here, the last node was
released mid-loop and this one can be too.

---

## Acceptance criteria (draft — `gen-plan` formalizes the numbers; inherits loop-11 numbering)

1. **AC-0 (NEW) Op-point re-established.** GLM-5.1-FP8 mask regenerated (SHA verified against the
   fixture fingerprint, or the fixture re-minted via the DEC-12 probes); validator permits radix-on
   for this boot's config; capacity reproduces (bs cap ≥ 64 @ mem 0.8); DSA-native default
   un-regressed (410560).
2. **AC-2 Tail TTFT.** At conc 16/32/64, DS **P99 TTFT < 22 s** (the `SLOS.md` hard bar) wherever
   DSA meets it, and DS P99 TTFT ≤ ~1.10× DSA radix-ON per concurrency. DS no longer
   admission-capped below nominal concurrency at conc ≤ 64. **Judged at the locked AC-11 sweep**,
   not the directional ladder.
3. **AC-3 Throughput.** DS per-request decode-TPS p50 **≥ 30** (the `SLOS.md` hard floor) at conc
   16/32/64, and DS aggregate ≥ ~0.95× DSA radix-ON at conc 64. Honest characterization if unmet —
   a miss is reportable, not a loop failure.
4. **AC-4 Per-step tax guard.** DS-vs-DSA same-batch one-batch decode window ratio ≤ ~1.10 at bs64
   (both mem 0.8); the bs30 window stays ≤ ~380k µs.
5. **AC-5 Quality.** Recall@2048 ±0.5pp fail-closed per landed change; cross-rank bit-identity hard;
   radix value-equivalence re-verified if the fixture is re-minted.
6. **AC-6 DS concept intact.** Offline mask → absorbed-latent signatures → query·signature scoring
   → top-k → sparse MLA decode. No dense fallback; no DSA-indexer substitution. (Already true; must
   not regress through the op-point rebuild.)
7. **AC-7 DSA-native default un-regressed — strict.** Shared-surface changes (memory accounting,
   radix plumbing, graph runner, validator, serve-script defaults) trigger the mandatory DSA
   regression in the same round. The shipped DSA default is untouched in behavior and performance.
8. **AC-8 Protocol/ledger/queue discipline.** Queue current every round; evidence pre-flight before
   each handoff (artifact exists + tracked + claim matches, describing the POST-commit state);
   `results.md` rewritten not layered; one-trial honesty; frozen references intact; push every round.
9. **AC-UX (NEW) Production enablement is documented and reproducible.** A runbook takes a GLM-5.1-FP8
   operator from zero to a serving DS server (calibrate → boot → fixture → serve) with stale defaults
   fixed, the diagnostic-vs-production knobs labeled, and the throughput guidance reconciled to the
   loop11b measured verdict.

---

## Files to read first

- **loop 11 state (the contract being finished):** `development/loop11/queue.md` (task statuses +
  the DEC-1…DEC-12 trail), `development/loop11/results.md` (the M0–M3 authoritative ledger),
  `development/loop11/draft.md` + `plan_v2.md` (original scope, AC definitions, owner decisions).
- **The op-point artifacts:** `development/serve_double_sparsity_radix_fixture.json` (pinned config +
  mask SHA + fixture schema); `python/sglang/srt/layers/attention/double_sparsity/calibrate.py` +
  `channel_mask.py` (mask regeneration + the production recipe in the docstring);
  `validator.py` (`write_radix_fixture_state`, the fail-closed fingerprint/schema check).
- **Serving + UX:** `development/serve_double_sparsity.sh` + `serve_native_nsa.sh` (the locked op-point
  flags, the stale `MODEL_PATH`/`CHANNEL_MASK_PATH` defaults, the dev-only knob sprawl, the
  contradictory loop8 warning); `development/benchmark.sh` + `benchmark_baseline.sh` +
  `benchmark_compare.py` (AC-11 mode + refusal rules); `development/SLOS.md` (the client bars; note:
  renamed from `CLIENT_SLOS.md`, which older scripts still cite).
- **The table-free DS selection path:** `double_sparsity/selector.py`, `selection_kernel.py`,
  `cuda_graph.py`, `config.py`; `python/sglang/srt/models/deepseek_v2.py` (absorbed `v_h` bind site,
  `kv_b_proj`/`W_UK`).
- **Gates:** `loop9/oracle_recall_summary.py`, `loop7/niah_oracle_sweep.py`, frozen
  `loop9/runs/20260610_m0/recall_baseline.json`.
- **Doctrine:** `CLAUDE.md`; `AGENTS.md`; `.pensieve/` maxims + decisions.

## Hardware / op-point

Single node **8×H200 (≈144 GB/GPU)**, TP=8, **GLM-5.1-FP8** (snapshot
`/cluster-storage/models/models--zai-org--GLM-5.1-FP8/snapshots/f396cf805182f4ca10fa675e1a99815b3ca384db`),
fp8_e4m3 KV, page 64, custom-all-reduce ON, `flashmla_kv` both phases, CUDA graph ON, mem 0.8 (DS
table-free) / 0.85 (DSA baseline), `max_running_requests=64`, `cuda_graph_max_bs=64`. **Never set
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments` for serving** (breaks custom-all-reduce IPC at GLM
TP=8 — and it is also what caused the calibration OOM lore). One TP=8 server at a time. Workload:
gsp 4096/512, ~55% prefix, seeds {16:213, 32:431, 64:31234}, server seed 20260607. **The physical
node changed since loop 11** — re-establish the op-point; do not inherit the released machine's live
numbers.

## Decisions already made by the owner (carried from loop 11 — do NOT relitigate in `gen-plan`)

1. **Lossiness bar:** recall-gated (±0.5pp fail-closed + cross-rank bit-identity) with declared
   value-affecting records; bitwise identity vs the deleted fp16-label path is not required.
2. **Comparison bar = radix-ON DSA** (production default). DS radix-on is **authorized under DEC-12**
   (production-representative-reuse edge contract; near-full reuse is out-of-contract
   value-affecting, characterized at +1.57pp, not a gate input).
3. **Table-free absorbed-latent is the one DS selection path** (TokenLabelTable deleted, DEC-2);
   `scorer_norm="off"` only.
4. **DS is an honest-verdict educational exercise** — a FAIL vs the client SLO is a reportable
   outcome, not a loop failure, and must be characterized, not hidden.

## Pending decisions (resolve in the `gen-plan` discussion)

- **Mask reproducibility ⇒ inherited vs re-earned authorization.** Is the GLM calibration corpus
  recoverable so the regenerated mask reproduces SHA `340b6c0b…` exactly (radix authorization
  *inherited*, committed fixture stands), or do we accept a freshly-calibrated mask and **re-mint**
  the fixture by re-running the DEC-12 edge probes (radix authorization *re-earned*)? This is the
  single biggest fork in the loop's size.
- **UX-pass depth.** Doc runbook + fix stale defaults + reconcile the warning (recommended,
  surgical) — *vs* a deeper config-ABI change (collapse the inline `--double-sparsity-config` JSON
  blob into real CLI flags, prune dev-only knobs). The latter is a **userspace change** and needs an
  explicit owner call before any work (Doctrine §8: do not break userspace casually).
- **task8 tax-guard tail.** If AC-4 runs tight at bs64, pull in a parked reducer (q4/q5) in-loop, or
  record the overage and ship the honest number? (Default: measure, then decide on the evidence.)
- **Sweep scope.** Confirm conc 16/32/64 × 3 trials × 600 s as the sole verdict venue. The `SLOS.md`
  deferred req #1 (128k ISL / 1024 OSL) is a *second* op-point — almost certainly out of scope for a
  finish loop, but worth an explicit "not now" so it is not silently dropped.
- **Verdict venue.** Confirm the closing AC-11 sweep is the sole AC-2/AC-3 venue and that the
  loop-11 frozen serving ladder is retired to "directional reference only" given the machine change.

--- Original Design Draft End ---
