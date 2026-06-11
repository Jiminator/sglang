# Loop 10 Ledger — DS-on Decode Dead-Width Tax

Rewrite-over-append: this file holds ONE authoritative current-state section, rewritten whenever
state changes. History lives in git. Plan: `development/loop10/plan.md`. Queue (single source of
truth for task state): `development/loop10/queue.md`.

## Current state (round 0)

### The number

| Reference | µs / 10-step decode window | ratio | status |
|---|---|---|---|
| Loop-9 final landed = loop-10 baseline (`loop9/runs/20260611_r1/`) | **480,989** | 1.403× | frozen starting point |
| Loop-10 current landed | 480,989 (no pipeline change landed yet) | 1.403× | — |
| Case-2 DSA floor (frozen, never re-run) | 342,857 | 1.0× | target reference |
| AC-1 hard bar | ≤420,000 | ~1.23× | open |
| AC-1 stretch | ≤395,000 | ~1.15× | open |

### Per-bucket residuals (R1 column = what loop 10 attacks)

| Bucket | R1 µs/window | hard bar | projected after M1 (task2) | projected after M2 (task2) |
|---|---|---|---|---|
| DS score-reduce transport (AR + casts) | ~108–111k | ≤60k | 49–58k | 35–45k |
| `_logical_score_kernel` | 36,908 | ≤20k | 16–22k | 15–20k |
| DS radix top-k | ≈36,300 | ≤28k | 24–32k (at risk) | 20–28k |
| shared non-DS topk/sort | 20,524 | n/a (not DS-attributable) | — | — |

### What is landed in loop 10 so far

- **Gate tooling (task1, this round)**: selection-capture bucket-identity tagging in the
  production dump path (`selection_capture.py` records raw_bs, padded_bs, selector_width,
  graph_key, replay_path, max_real_seq_len; `DSGraphState.last_replay_graph_key` stamped by the
  DSA backend's pre-replay metadata init — both replay variants). Zero selection-pipeline change;
  2 new CPU regressions (352 pass in the DS unit file).
- **Loop-10 capture tool** `development/loop10/selection_capture_tool.py`: bs-1 workload
  (loop-9-identical), op-point `run-op` (29 concurrent ~4k-token requests, deterministic
  staggered admission), identity-aware `verify` (hard requirements: raw_bs/replay/padded_bs),
  `diff --fail-on-diff` (hard row gate), `diff-digest` (hard SHA gate, identity fail-closed with
  explicit `--allow-identity-change` declarations).
- **Gate script** `development/loop10/run_gates.sh`: A1 bs-1 selcap → HARD digest gate; A2
  op-point selcap → HARD identity + digest gate; B recall oracle; C Case-1 frozen-recipe profile.
- **task2 analyze artifact** `reviews/task2_projection_transport_model.md`: verified capture
  ladder `[1,2,4,8,12,16,24,32,...,512]`; op point pads 29→32 (320 KiB compact → two-shot side);
  threshold-flip buckets bs≤16; **pin caveat: `override_shot(2)` leaves `one_shot_push_threshold`
  intact — task5 must pin via `override_algo=TWO_SHOT_PULL`**; NCCL 38.5 vs custom-AR 51.9
  µs/call at [29,4608] bf16 (task8 matters).

### Frozen gate baselines (this loop's diff targets)

- bs-1 digest: `runs/20260611_m0_freeze/bs1_digest.json` — must match the loop-9 R1 fingerprint
  (`loop9/runs/20260611_r1/selcap_digest.json`) bit-exactly at freeze time.
- op-point digest: `runs/20260611_m0_freeze/op_digest.json` — first frozen op-point baseline
  (raw_bs=29, padded_bs=32, graph replay proven).
- recall: `loop9/runs/20260610_m0/recall_baseline.json` (frozen, reused).

### Open items / next

- task3 width-bucketing design dossier (Codex, in flight).
- task4 keying/metadata-lifetime patch — blocked on task1 frozen digests + task3.
- Conditional/queued: task7–task10 per `queue.md`.
