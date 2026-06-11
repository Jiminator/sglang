# Loop 10 Task Queue

Single source of truth for what is planned, in flight, done, or dropped. Seeded at loop kickoff
(round 0) from `development/loop10/plan.md`'s task table plus kickoff ideas, per the draft's
protocol. Statuses: `queued` | `in-progress` | `done` | `dropped` | `conditional` | `candidate`.
A task is `done` only after its gates pass (losslessness teeth + profile). Drops keep their row
with the measured/reasoned cause — no silent deletions. New mid-loop ideas are APPENDED as
`candidate` entries with a one-line compatibility note, never absorbed into a running task.

Exactness regimes (binding, declared per entry before landing — see plan "Change classification"):
- **exact**: layout/shape/keying/bucketing with same reduce algorithm+dtype → zero selcap index
  diff (hard gate) + cross-rank bit-identity + recall. Custom-AR pinned two-shot for all compact
  variants.
- **value-affecting**: reduce algorithm/dtype/summation-order changes → cross-rank bit-identity
  HARD + recall@2048 ±0.5pp fail-closed + explicit declaration here and in the ledger; selcap diff
  recorded as evidence (expected nonzero), not pass/fail.

## Mainline queue (from plan task table)

| ID | Description | Targeted bucket | Expected effect | Regime | Compatibility vs landed | Status |
|----|-------------|-----------------|-----------------|--------|------------------------|--------|
| task0 | Populate this queue from the plan + kickoff ideas (loop's FIRST runtime task) | — (protocol) | AC-5 queue discipline established | n/a (docs) | n/a — first task | done (round 0) |
| task1 | Op-point selcap harness (bs-29 concurrent decode under graph replay; dumps/digests tagged graph key, selector width, raw bs, padded bs, max real seq_len) + promote selcap diff to HARD gate in cloned loop10 gate script + freeze pre-change bs-1 and op-point digests | — (gate tooling) | DONE — bs-1 freeze matched loop-9 R1 fingerprint bit-exactly (64 steps, 0 SHA mismatches: cross-boot reproducibility + tagging selection-neutrality proven); op-point freeze: raw_bs=29, padded_bs=32=graph_key, replay path, 2×12 steps deterministic; frozen digests in `runs/20260611_m0_freeze/`; measured selector width 202,756 | n/a (tooling; selection-neutrality PROVEN by the zero-diff freeze) | Builds on loop-9 `selection_capture_tool.py` / `run_r1_gates.sh`; no pipeline change | done (round 0) |
| task2 | Re-derive projection + transport model vs 480,989: two-shot at compact sizes, threshold-flip map across bs ladder (160 KB one-shot boundary, 8 ranks), cast-tax accounting, loop-9 spike-bench evidence | transport (model only) | DONE — artifact `reviews/task2_projection_transport_model.md`: real ladder [1,2,4,8,12,16,24,32..512] (verified vs serve.log); op point pads 29→32 (320 KiB, two-shot side); flip buckets bs≤16; PIN CAVEAT: `override_shot(2)` leaves one_shot_push_threshold intact — task5 must pin via `override_algo=TWO_SHOT_PULL`; projections: transport 49–58k (M1) / 35–45k (M2), logical-score 16–22k, top-k 24–32k (M1, bar at risk) → total ~387–410k (M1) | n/a (analysis) | None — pure analysis | done (round 0) |
| task3 | Width-bucketing design dossier: (bs,width) key contract end-to-end (graph dict, output buffers, replay lookup, capture order, DSA metadata lifetime, DSGraphState ownership), ladder-wide coverage + capture-memory budget vs M4 headroom, overflow/fallback semantics | all DS buckets (design) | DONE — artifact `reviews/task3_width_bucketing_dossier.md`: tuple key (bs,width) gated by `_use_ds_selector_width_keys` (DS-on decode only); config-borne `selector_width_buckets` (Patch 1 empty → full-width only); dispatch bs-bisect then width over `seq_lens_cpu[:raw_bs]`; DS-only `PinnedDSScoreReduceCA` wrapper with per-call `override_algo=TWO_SHOT_PULL`; compact set adds ~1.10 GiB vs ~14.2 GiB headroom; 104 captures (boot ~1.6–2.0×, measured in task5); per-patch expected identity declarations (P1: graph_key; P2: +selector_width); 11-entry risk register with gate observables | n/a (analysis) | Consumes task2 threshold-flip map; verified vs plan + frozen digests | done (round 0) |
| task4 | Keying/metadata-lifetime patch: (bs,width) identity through runner + DSA metadata + DSGraphState ownership, FULL-WIDTH variants only (zero behavior change); DS-off invariants (AC-4.1 tests) | — (structure) | Zero perf change BY DESIGN; isolates runner risk before compact buffers | exact (zero selcap diff required) | First pipeline-touching change; same-round AC-4.2 DSA regression on landing | queued (blocked: task1, task3) |
| task5 | Compact patch: real per-width DSGraphState buffers (W=5120 prefix window + guaranteed full-width fallback, whole bs ladder per DEC-2), real-row host dispatch (`seq_lens_cpu[:raw_bs].max()`), pinned two-shot AR, bucket-boundary tests (W / W+1 / 4096→4608 growth / padded rows), runtime contiguity assertion | transport + logical-score + top-k (all width-scaled) | transport 108–111k → ~35–55k; logical-score 36.9k → ~15–20k; top-k 36.3k → ~20–28k (task2 re-rates) | exact (pinned two-shot; zero selcap diff) | Requires task4 keying landed; capture-memory/boot-time measured vs M4 headroom | queued (blocked: task4) |
| task6 | M1 gate run: full AC-2 suite (selcap bs-1 + op-point, cross-rank, recall) + same-round AC-4.2 DSA regression + frozen-recipe Case-1 profile + per-bucket gap read; keep-or-revert | all | Banks M1 or reverts it; bottleneck-shift read feeds this queue | n/a (gate run) | Verdict on task4+task5 | queued (blocked: task5) |
| task7 | Cast elimination / bf16-authoritative top-k input | transport (cast tax ~15–18k) | Removes fp32↔bf16 cast pair; exact ONLY if top-k sees identical reduced bf16 values vs copy-back path (radix needs 2-round bf16); else value-affecting | exact-if-proven, else value-affecting (declared before landing) | On compact buffers from task5; touches radix top-k input dtype | queued (blocked: task6) |
| task8 | Per-bucket transport choice: measure pinned two-shot vs DECLARED one-shot override vs NCCL at compact sizes; log reduce dtype + actual algorithm per bucket; classify each candidate BEFORE landing; build the loser before verdicts | transport | Cheapest correct transport per bucket; loop-9 spike bench says compact bf16 custom-AR can LOSE to NCCL at [29,4608]-class shapes | value-affecting for any algorithm change; pinned two-shot stays exact | On compact buffers from task5; per-instance override isolated to DS score reduce only | queued (blocked: task6) |
| task9 | CONDITIONAL top-k redesign: Triton multi-block single-launch on compact rows (target measured 17.7 µs/call floor), deterministic tie-break contract bit-exact; AOT wheel install stays gated behind full AC-4 | top-k | 36.3k → toward ~17.7 µs/call × 780 ≈ 13.8k ceiling-case; only runs if top-k bucket > 28k after M1+M2 | exact (tie-break contract bit-exact) | On compact rows from task5; Triton-first per DEC-3 | conditional (blocked: task6, task8) |
| task10 | Close-out: final attribution review, results.md authoritative state, evidence pre-flight, queue reconciliation | all (review) | AC-1 per-bucket attribution + AC-5 evidence discipline verified | n/a (analysis) | Last task | queued (blocked: task6, task7, task8, task9) |

## Kickoff candidates (appended round 0; not in the plan task table)

| ID | Description | Targeted bucket | Expected effect | Regime | Compatibility vs landed | Status |
|----|-------------|-----------------|-----------------|--------|------------------------|--------|
| cand1 | Post-compaction skew re-read: nsys timeline after task5 lands to split the residual transport bucket into wire-time vs cross-rank arrival-skew absorption; if skew dominates, consider launch-order/stream-priority shaping (no data change) | transport | Attribution only at first; potential few-k µs if skew is shapeable | n/a (measurement), any follow-up exact | Needs compact buffers landed (task5); nsys kernel-name attribution (NVTX dead under replay) | candidate |
| cand2 | Radix launch-count reduction short of full redesign: collapse the 5-kernel/11-launch suite on compact rows (e.g. fuse count+scan passes) if task9's full redesign is not triggered but launch overhead is visible at W=5120 | top-k | Few-k µs launch overhead; cheaper than task9 | exact (tie-break contract unchanged) | On compact rows; mutually exclusive with task9 (one supersedes the other) | candidate |

## Drops

(none yet)
