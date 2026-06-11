1. INCUMBENT READ: DS compact score reduce is already pinned to bf16 compact buffers `[padded_bs, 5120]` with per-call `override_algo=TWO_SHOT_PULL` via `PinnedDSScoreReduceCA`. Binding captured-replay evidence: `all_reduce_two_shot_kernel<bf16,8u>` is `24,420 us / 780 calls = 31.3 us/call` at `(32, 5120)`, 320 KiB. Cast/copy is `5,999 us`; transport bucket total is `30,419 us`, already below both AC-1.1 hard `<=60k` and stretch `<=45k`. Case-1 total is `375,892 us`, below hard `<=420k` and stretch `<=395k`.

2. CANDIDATES: Declared `ONE_SHOT_PULL`: viable for correctness per 8-rank probe, sum-exact over tested sizes, but no binding performance win is measured. `ONE_SHOT_PUSH`: not viable at the op-point; it hard-errors above roughly 160 KiB, so declaring it for 320 KiB would crash. `NCCL`: lower-confidence eager CUDA-event evidence shows `38.5 us/call` at `[29,4608]` bf16, 267,264 B, which is smaller than the incumbent 320 KiB op-point yet already slower than incumbent binding `31.3 us/call`. At `[29,202752]`, NCCL `105.9 us` is effectively tied/slower than custom-AR two-shot `104.1 us`. Coordinator/custom-AR alternatives from the loop-9 spike are also lower-confidence and slower at the small bucket: `51.9 us/call`.

3. VERDICT: DROP. Measured cause: the landed binding incumbent is already faster than the best available alternative evidence even when that alternative was measured eagerly at a smaller size: `31.3 us/call` incumbent at 320 KiB versus NCCL `38.5 us/call` at 267 KiB. No viable candidate has measured evidence for a material win, and `ONE_SHOT_PUSH` is disqualified by crash behavior at the target size. Landing any transport change would be value-affecting, require explicit declaration, digest re-freeze, and recall-blind transport-risk accounting, while the transport bucket is already about 2x under the stretch bar and Case-1 total is already under stretch. The benefit side is not measured; the churn and risk side is concrete.

4. CONSEQUENCES: Task9 should read transport as closed/drop, not as pending insurance. Close-out attribution should credit loop10 transport to the landed compact bf16 `TWO_SHOT_PULL` pin, with remaining condition pressure on AC-1.2 logical-score rather than DS score-reduce transport.

---

## Round-4 addendum: the same-shape measured matrix (Codex round-3 required action)

Direct 8-rank matrix at the REAL compact op-point shape [32, 5120] bf16 (327,680 B,
weak-contiguous, custom-AR eligible), `development/loop10/task8_transport_matrix.py`,
artifact `runs/20260611_task8/task8_matrix.json`. Same conditions for all three
non-crashing candidates; ONE_SHOT_PUSH crash evidence cited from `ar_algo_probe.json`.

| Candidate | eager µs/call | captured-replay µs/call (BINDING) | µs/window @780 |
|---|---|---|---|
| pinned TWO_SHOT_PULL (incumbent) | 37.31 | **14.85** | **11,583** |
| forced ONE_SHOT_PULL (declared) | 33.18 | 20.42 | 15,928 |
| NCCL bf16 | 33.73 | 69.73 | 54,389 |

VERDICT CONFIRMED: **DROP** — the incumbent wins the binding mode by 27% over the only
viable declared one-shot and by 4.7× over NCCL. The eager column INVERTS the ranking
(NCCL appears competitive eager), which retroactively explains the loop-9 spike bench's
"NCCL wins at compact sizes" as an eager-measurement artifact and re-validates the
captured-replay-is-binding discipline for collectives, not just kernels. No transport
change lands; the original task8 measurement contract is now satisfied with direct
same-shape evidence.
