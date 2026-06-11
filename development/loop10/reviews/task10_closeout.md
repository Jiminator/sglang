## FINAL AC TALLY

| AC | Status | Binding artifact + number |
|---|---|---|
| AC-1 total | MET hard and stretch | `development/loop10/runs/20260611_task7/summary_torch.txt`: 361,824 µs, ≤420,000 hard and ≤395,000 stretch; 1.055× vs frozen 342,857 µs DSA floor. |
| AC-1.1 transport | MET hard and stretch | `runs/20260611_task7/cmp_vs_loop9r1.txt`: 17,623 µs = 14,137 AR + 2,206 `direct_copy` + 1,280 `bfloat16_copy`, ≤60,000 hard and ≤45,000 stretch; AR boot variance caveat applies. |
| AC-1.2 logical-score | ORIGINAL BAR NOT MET; terminal only via DEC-L10-3 + DEC-L10-4 | `runs/20260611_task7/cmp_vs_loop9r1.txt`: 22,887 µs vs ≤20,000 hard, improved from 36,908 µs (-38%). Terminality comes from owner re-scope DEC-L10-3 and explicit stop-condition change DEC-L10-4, not from meeting the bar. Frontier artifacts: `exact_floor_random.json`, `exact_floor_page64.json`, `headsplit_proto.json`, `layout_proto.json`. NO valid lower-bound proof exists and none is claimed: exact-floor artifacts report `valid_lower_bound_exhaustion=false`; the round-4 stripped/roofline proof is retracted; `roofline_probe.json` is notes/int8 evidence only. |
| AC-1.3 DS top-k | MET hard and stretch | `runs/20260611_task7/cmp_vs_loop9r1.txt`: approx. 23,271 µs = 18,501 hist + 1,338 block_count + 2,546 emit + 886 block_prefix, ≤28,000 hard and ≤24,000 stretch. |
| AC-2 no extra lossiness | MET under declared regimes | DEC-L10-1 is the sole declared value-affecting exception; task7/task11 exact gates are zero-diff. |
| AC-2.1 recall | MET | `runs/20260611_task7/recall_gate.json`: 64.706%, PASS within ±0.5pp vs frozen `loop9/runs/20260610_m0/recall_baseline.json`. |
| AC-2.2 cross-rank identity | MET | `runs/20260611_task7/selcap_bs1_verify.log`: 64 steps, 8 ranks bit-identical; `selcap_op_verify.log`: 24 steps, 8 ranks bit-identical. |
| AC-2.3 exact selcap diff | MET with DEC-L10-1 declared exception | Chain is `m0_freeze -> task4 -> task6r2 -> task11 -> task7`; exact hops are zero-diff, and the compact transport hop has op-point zero-diff with bs-1-class transport-order churn declared in DEC-L10-1. |
| AC-2.4 boundary/contiguity | MET | Queue/results record compact W=5120, W+1 full fallback, 4096->4608 growth, padded-row handling, and weak-contiguity refusal; `task8_matrix.json` confirms `[32,5120]` bf16 weak-contiguous custom-AR eligibility. |
| AC-3 DS concept intact | MET | Landed path remains offline mask -> signatures -> query·signature scoring -> top-k -> sparse MLA decode; no dense fallback or DSA-indexer substitution recorded. |
| AC-4.1 DS-off invariants | MET | task4 results/queue: DS-off keying invariants and CPU tests passed; width-key path gated to DS-on decode. |
| AC-4.2 Case-2 regressions | MET | Same-round regressions: 340,621 / 341,037 / 341,488 µs vs frozen 342,857 floor; frozen references were compared against, not replaced. |
| AC-5 protocol/ledger/queue | MET | `results.md` is the authoritative current ledger; queue entries and drops are retained; this regenerated close-out supersedes the layered prior artifact. |

## PER-BUCKET ATTRIBUTION

Final binding profile: `development/loop10/runs/20260611_task7/cmp_vs_loop9r1.txt`. Per-call values use the 780-call decode-window denominator used by the loop close-out.

| Bucket | R1 µs/window | Final µs/window | Per-call µs | Bar | Verdict |
|---|---:|---:|---:|---|---|
| DS transport: AR + cast/copy | approx. 108-111k | 17,623 | 22.59 | ≤60k hard / ≤45k stretch | MET stretch |
| `_logical_score_kernel` | 36,908 | 22,887 | 29.34 | ≤20k hard / ≤15k stretch | NOT MET original bar |
| DS radix top-k | approx. 36,300 | 23,271 | 29.83 | ≤28k hard / ≤24k stretch | MET stretch |
| TOTAL | 480,989 | 361,824 | n/a | ≤420k hard / ≤395k stretch | MET stretch |

Transport caveat: the AR kernel varies 14.1k to 35.4k across gate boots due to boot/skew absorption, but every observed boot remains within AC-1.1 bars.

AC-1.2 causal closure: all measured frontier levers are worse than the landed kernel: tb sweep, fewer workers, block-grid, stripped same-structure, bitwise-exact head-split, `[H,D,T]` layout transposition, and int8. The layout prototype is bitwise-different and 2.5-10.5× slower cold across all four variants on both slot layouts; landed-kernel cold-net replay is 27.77/29.48 µs/call, matching the binding 29.34 µs/call residual.

## DECISION RECORD

- DEC-L10-1, round 2, plan-sanctioned ledger declaration: the compact-bucket NCCL-to-pinned-two-shot transport flip was declared value-affecting for bs-1-class alignment buckets, while the op point remained zero-diff and the probe showed order effects rather than corruption.
- DEC-L10-2, rounds 4-5, owner plus review authority: the owner kept the AC-1.2 bar and exact-only ladder before int8, then the round-4 stripped lower-bound claim was retracted and replaced by a measured-frontier record.
- DEC-L10-3, round 6, owner AskUserQuestion authority: after the completed measured frontier, the owner re-scoped AC-1.2 to a characterized finding while preserving the numeric fact that 22,887 µs did not meet ≤20,000.
- DEC-L10-4, round 7, owner AskUserQuestion authority: the owner explicitly changed the loop stop condition to accept DEC-L10-3 as terminal despite the original AC-1.2 hard-bar miss.

## EVIDENCE PRE-FLIGHT

Tracked binding evidence verified in-repo: `plan.md`, `queue.md`, current `results.md`, task2/task3/task8 reviews, task7 summaries/digests/diffs/verify logs/recall, task8 `task8_matrix.json`, task11 `task11_bench.json`, `exact_floor_random.json`, `exact_floor_page64.json`, `headsplit_proto.json`, `layout_proto.json`, `roofline_probe.json`, Case-2 comparison summaries, and the loop9 frozen recall/baseline references.

Tracked source provenance verified: `task11_exact_floor_harness.py`, `task11_headsplit_prototype.py`, `task11_layout_prototype.py`, `task11_roofline_probe.py`, `task11_logical_score_bench.py`, and `task8_transport_matrix.py`.

Explicitly local forensic evidence: raw trace trees, raw serve/run logs, selcap pass directories, and oracle sink files; their durable claims are represented by tracked summaries, digests, comparisons, and JSON reports.

Current state (round-9 correction of a generation-time sentence): `results.md`, `queue.md`, and this regenerated close-out are all committed and current at HEAD; the prior layered close-out exists only in git history as the superseded artifact. No binding cited artifact is untracked.

## QUEUE/PROTOCOL COMPLIANCE

Task0-task7 are terminal: task0 established queue discipline; task1 built/froze gates; task2/task3 completed analysis; task4-task6 banked M1 width bucketing and compact buffers; task7 banked bf16-authoritative top-k/copy-back removal.

Task8 is terminal as dropped on same-shape measured evidence: captured-replay two-shot 14.85 µs/call beats one-shot-pull 20.42 and NCCL 69.73 at `[32,5120]`; one-shot-push is non-viable above ~160 KiB.

Task9 is terminal as condition-false: post-M2 top-k is approx. 23,271 µs, under both hard and stretch bars.

Task10 is terminal by this regenerated close-out. Task11 is terminal through DEC-L10-3 characterization plus DEC-L10-4 stop-condition change.

`cand1` and `cand2` remain dispositioned unexecuted candidates, not blockers: transport attribution was satisfied by task8 plus the boot-variance note, and top-k launch reduction was not triggered because AC-1.3 met stretch.

Frozen-reference discipline holds: binding Case-1 profiles are loop10 runs; Case-2 runs are AC-4.2 regressions only; frozen floor 342,857 and frozen recall baseline were never replaced. Baseline chain integrity holds: `m0_freeze -> task4 -> task6r2 -> task11 -> task7`, with every hop zero-diff or DEC-L10-1-declared.

## LOOP SUMMARY

Headline: Case-1 DS-on decode moved 480,989 -> 361,824 µs, from 1.403× to 1.055× vs the frozen DSA floor.

Landed: op-point selection-capture gates, `(bs,width)` graph keying, compact W=5120 variants with full-width fallback, shared per-width DSGraphState, pinned two-shot score reduce, logical-score tb-512 tuning, and bf16-authoritative top-k input with copy-back removal.

Declined with evidence: transport alternatives, top-k redesign, head-split, `[H,D,T]` layout transposition, stripped lower-bound framing, and int8 signatures. Int8 remains transaction-limited and projects above the original 20k bar; layout transposition is both bitwise-different and slower; head-split is bitwise-exact but slower.

Lessons produced: collective buffer resizing can silently flip transport algorithms, per-variant graph state multiplies capture memory, eager collective benchmarks can invert captured-replay rankings, and the final AC-1.2 residual is cold-cache scattered-gather bandwidth rather than an unmeasured exact kernel lever.

## TERMINAL VERDICT

Loop 10 is terminal because DEC-L10-4 explicitly changes the stop condition to accept DEC-L10-3 as terminal: AC-1 total, AC-1.1, AC-1.3, AC-2, AC-3, AC-4, and AC-5 are MET, while original AC-1.2 is plainly NOT MET at 22,887 µs vs ≤20,000.
