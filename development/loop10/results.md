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

### Frozen gate baselines (this loop's diff targets) — FROZEN 2026-06-11 round 0

- bs-1 digest: `runs/20260611_m0_freeze/bs1_digest.json` — **matched the loop-9 R1 fingerprint
  bit-exactly** (`bs1_diff_vs_loop9r1.json`: 64 steps, 0 SHA mismatches). This proves both
  cross-boot selection reproducibility and that the bucket-identity tagging change is
  selection-neutral on hardware.
- op-point digest: `runs/20260611_m0_freeze/op_digest.json` — verdict PASS with hard
  requirements raw_bs=29 / replay path / padded_bs=32 on every step; graph_key=32,
  selector_width=202,756, identity uniform, 2 passes × 12 steps run-to-run deterministic,
  8 ranks bit-identical; request seq_lens 4,046–4,166 (op-point ISL-4096 class).
- Local forensic baselines (gitignored .pt snapshot dirs, ~3.7 GiB): `runs/20260611_m0_freeze/
  selcap_bs1/`, `selcap_op/` — used by `diff --fail-on-diff` for row-level attribution when a
  digest gate fails.
- recall: `loop9/runs/20260610_m0/recall_baseline.json` (frozen, reused).
- Measured fact: the static selector width on this build is **202,756** (`req_to_token.shape[1]`),
  not the 202,752 quoted in the plan — immaterial (width is read from code, never hardcoded).

### Round-1 evidence repair (Codex round-0 blockers, `runs/20260611_r0_repair/`)

- Recall gate PASS: overall recall@2048 **64.706%** vs frozen baseline 64.696% (+0.01pp,
  bar ±0.5pp) — `recall_gate.json`.
- Case-2 DS-off regression PASS: **340,621 µs vs frozen floor 342,857 (0.99×)**; per-bucket
  deltas within tens of µs except −2.4k all-reduce routing jitter — inside the loop-9 DEC-1
  noise band — `case2_cmp_vs_frozen_floor.txt`. The round-0 tagging change now carries all three
  losslessness teeth plus the DSA regression.

- **task3 analyze artifact** `reviews/task3_width_bucketing_dossier.md` (round 0): the binding
  M1 design — tuple graph key `(bs, width)` gated to DS-on decode by
  `_use_ds_selector_width_keys`; config-borne `selector_width_buckets` (Patch 1 plumbs with an
  empty compact list → full-width only, zero behavior change); width dispatch after the existing
  bs bisect over `forward_batch.seq_lens_cpu[:raw_bs]` (real rows only); DS-only
  `PinnedDSScoreReduceCA` wrapper pinning via per-call `override_algo=TWO_SHOT_PULL`; compact
  W=5120 set costs ~1.10 GiB against ~14.2 GiB M4 headroom; 104 whole-model captures (boot-time
  measured in task5); DS-off invariant test suite designed; 11-entry risk register tied to the
  frozen gate observables.

### task4 BANKED (round 1, KEEP) — M1 Patch 1

- Code (commit `6c92240b9`): `(bs, selector_width)` graph-variant keys for DS-on decode only
  (`use_ds_selector_width_keys` gate; PDMux/spec/dllm/encoder untouched), config-borne
  `selector_width_buckets` (empty → full width only), DSA decode metadata keyed by the variant
  via the `_ds_graph_variant_key` channel, `last_replay_graph_key` stamps the full key, real-row
  width dispatch helper. 19 new CPU tests; DS unit file 371 pass.
- Gates (evidence `runs/20260611_task4_gates/`, commit `65627ea74`): bs-1 selcap 0 SHA
  mismatches + 0/2,496 rows; op-point 0 SHA mismatches + 0/27,144 rows; the ONLY identity change
  is the declared `graph_key` int→tuple; recall 64.706% (+0.01pp); Case-1 481,253 µs vs R1
  480,989 (1.00×, all buckets flat — zero-behavior proven in perf); Case-2 DS-off 341,037 µs vs
  the frozen 342,857 floor (0.99×, within noise) with all 52 graphs captured through the
  re-keyed runner.
- **Gate-baseline advance**: task5's exact gates diff against `runs/20260611_task4_gates/`
  digests (bs-1 + op-point), with `selector_width` (202756→5120 at the op point) and `graph_key`
  as the declared identity changes; indices/lengths SHAs must remain bit-identical.

### DEC-L10-1 (round 2): task5 transport component re-classified value-affecting (plan-sanctioned path)

The task6 bs-1 hard gate FIRED on the first compact landing — working exactly as designed — and
the diagnosis produced two measured findings:

1. **Pre-existing alignment accident**: the bs-1 FULL-WIDTH score reduce was never custom-AR
   eligible (1 × 202,756 × 2 B = 405,512 B fails `should_custom_ar`'s 16-byte divisibility) —
   every frozen bs-1 fingerprint (loop-9 R1, m0_freeze, task4) embeds a silent NCCL reduce for
   bs-1 while the op point embeds custom-AR two-shot. The pre-change transport was already
   shape-dependent. Compact buffers (bs × 5120 × 2 B, always 16-aligned) made those buckets
   custom-AR-eligible → algorithm flip NCCL→pinned two-shot on exactly the compact bs-1 steps →
   summation-order change → boundary churn where selection is score-sensitive (seq > top_k):
   measured median ~107/2,048 positions (5.2%), mean 130, max 695 (`runs/20260611_task6_gates/`
   + forensic row diffs). Steps with seq ≤ top_k and all full-width steps are bit-identical.
2. **Correctness probe** (`ar_algo_probe.py` / `ar_algo_probe.json`, 8 ranks): two-shot,
   one-shot-pull, one-shot-push, size-based, and NCCL all produce EXACT sums on
   exactly-representable inputs at every probed shape incl. 10 KiB — the effect is summation
   ORDER only, not corruption. (Side finding: ONE_SHOT_PUSH hard-errors >160 KB — an unpinned
   flip could crash, reinforcing the pin.)

Per the plan's change classification ("revert, or re-classify with a user-visible ledger
entry"), the compact-bucket transport flip is DECLARED VALUE-AFFECTING under the loop-9 bf16
precedent: cross-rank bit-identity HARD (PASS), run-to-run determinism (PASS), recall@2048
±0.5pp fail-closed (gate rerun), selcap diff recorded as evidence (expected nonzero at bs-1).
Deliberately replicating the alignment accident (forcing NCCL on compact buckets) to preserve
digest equality was rejected as enshrining a bug; the op point keeps two-shot on both sides.
After this landing banks, the task6-state digests become the exact-diff baselines for task7+.

### Capture-memory amendment (round 2, measured): DSGraphState shared per width

Per-variant DSGraphState ownership OOM'd CUDA-graph capture at 41/104 captures (mem 0.7,
`runs/20260611_task6_gates/selcap_op_serve.log`): width-proportional scratch and selcap mirrors
multiplied by the 52-entry ladder (~15 GiB full-width scratch + ~6.7 GiB mirrors → ×2 with
compact variants). Amended (dossier §1 deviation, documented): ONE shared DSGraphState per
selector width, allocated at the global max capture bs; every same-width variant's DSAMetadata
references it. Aliasing is safe (one replay at a time; each replay fully rewrites the rows it
reads). This also REALIZES the loop-9 M4 audit's recoverable headroom (~14 GiB) instead of
merely respecting it. The dump's padded_bs now derives from the graph key.

### Open items / next

- task6 M1 gate rerun (in flight): SELCAP_DIFF_AS_EVIDENCE for the declared component, recall
  HARD, same-round Case-2, Case-1 profile + per-bucket read, capture budget artifact.
- Then task7/task8; task9 conditional; task10 close-out per `queue.md`.
