# Loop 9 Plan — Close the DS-on Decode-Overhead Gap on GLM-5.1 (Double Sparsity Kernel Optimization)

## Goal Description

Reduce the Case-1 (DS-on) one-batch decode GPU-kernel time on GLM-5.1-FP8 (single node 8xH200,
TP=8, bs 29, mem-fraction 0.7, ISL 4096 / OSL 512) from the frozen baseline of 632,239 µs per
10-step decode window toward the frozen Case-2 DSA floor (342,857 µs at the same bs/mem), by
attacking the three DS-specific index/scoring kernel buckets directly:

| Bucket | Case-1 cost vs Case-2 | Source site |
|---|---|---|
| NCCL ring score all-reduce (`ncclDevKernel_AllReduce_Sum_f32_RING`, once per layer per step) | +124,873 µs | `all_reduce_token_scores` helper AND the direct reduce in the graph-safe selector path (`python/sglang/srt/layers/attention/double_sparsity/selection_kernel.py`) |
| Generic torch top-k/sort stack (mbtopk / radixSort / sbtopk / scan_by_key / searchsorted) | +138,602 µs | `select_topk_sequence_order` / `_topk_by_score_then_pos` |
| `_logical_score_kernel` (DS channel-score compute) | +63,107 µs | `_logical_score_kernel` / `_logical_score_triton` |

DSA's fused indexer does the equivalent work in ~17.2k µs — that is the cost scale to move toward.
Penalty B (batch-efficiency: DS is KV-pool-capped at bs 29 vs DSA's bs 64) is secondary and
audit-first; no signature coarsening. This loop is engineering reproduction of the Double Sparsity
paper (arXiv:2408.07092) on an MLA model for performance-engineering learning — not frontier
algorithm work; DS theoretically cannot beat the trained DSA indexer, and that is fine. The
DSA-native default path must remain byte-identical. SLO re-validation, recall R&D, multi-node, and
knob compatibility are out of scope this loop.

Lossiness bar (user decision DEC-3): a landed change may move token selection only within a
bounded recall budget — NIAH oracle recall@2048 on the fixed gated workload must move by at most
0.5 percentage points per landed change — while cross-rank selection agreement remains a hard,
bit-identical requirement.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.

- AC-1: The DS-specific decode kernel buckets shrink, measured per-bucket on re-profiled Case 1 against the frozen Case-2 reference. Per-bucket gates are PRIMARY; the run total is secondary because shared-kernel boot-to-boot variance is ~27k µs. Per DEC-1, each gate is a hard target to be attempted in earnest; if a gate proves unreachable, a documented, materially-attributed trend reduction in that bucket is acceptable, with the shortfall and reason recorded in the ledger.
  - AC-1.1: Score-reduce bucket — the NCCL ring score-reduce line is eliminated from Case-1 DS-attributed decode kernels and replaced by a named custom-all-reduce kernel at the DS reduce site.
    - Positive Tests (expected to PASS):
      - Post-change Case-1 profile shows zero `ncclDevKernel_AllReduce_Sum_f32_RING` calls attributed to the DS score reduce (via its NVTX range), with a named custom-AR kernel in its place.
      - The ledger records the actual reduce backend selected (`custom_ar_v1` / `custom_ar_v2` / `pynccl` / `torch_dist`).
      - CUDA graph capture and replay succeed with zero replay allocations.
    - Negative Tests (expected to FAIL):
      - A profile showing the reduce silently falling back to pynccl/torch-distributed fails the gate even if the old ring line is gone.
      - A re-route that breaks graph capture or introduces replay allocations fails.
  - AC-1.2: Top-k selection bucket — the torch top-k/sort kernel group in the Case-1 decode window drops to at most 80,000 µs (stretch: toward DSA's `topk_transform` ~7.7k µs scale).
    - Positive Tests (expected to PASS):
      - `compare_decode.py` kernel-group diff shows the mbtopk/radixSort/sbtopk/scan_by_key/searchsorted group at ≤80,000 µs with no compensating growth in other DS-attributed buckets.
    - Negative Tests (expected to FAIL):
      - A result where the torch stack shrinks but a new selection kernel adds back more time than was saved (net bucket regression) fails.
  - AC-1.3: Logical-score bucket — the `_logical_score_kernel` group in the Case-1 decode window drops to at most 40,000 µs.
    - Positive Tests (expected to PASS):
      - Per-kernel diff shows the logical-score group at ≤40,000 µs.
    - Negative Tests (expected to FAIL):
      - An "optimization" that relocates score work into another kernel without net DS-bucket reduction fails.
  - AC-1.4: Total (secondary, trend): final Case-1 decode GPU-kernel total at or below 560,000 µs (minimum trend marker) / 516,000 µs (strong marker), interpreted with the ~27k µs shared-kernel noise in mind; per-bucket evidence prevails on conflict.
    - Positive Tests (expected to PASS):
      - The ledger shows the final total alongside per-bucket attribution for every landed idea.
    - Negative Tests (expected to FAIL):
      - Claiming AC-1 success on a total reduction that is not attributable to the targeted DS buckets fails.
- AC-2: No lossiness beyond the user-accepted bound (DEC-3) — every landed change has a recorded gate result from the M0 harness.
  - Positive Tests (expected to PASS):
    - Per landed change: NIAH oracle recall@2048 on the fixed gated workload moves by ≤0.5 percentage points versus the frozen pre-loop baseline, and the result is recorded in the ledger.
    - Per landed change: `selected_indices` and `valid_lengths` are bit-identical across all attention-TP ranks within the same run (hard correctness gate, independent of the recall bound — rank disagreement on selected tokens is a correctness failure, not lossiness).
    - The harness's exact selected-index diff versus the frozen production oracle is recorded for attribution (diagnostic, not pass/fail).
    - Adversarial micro-fixtures (equal-score runs straddling the k-boundary, -inf/unwritten rows, seq_len < top_k, fully-padded rows) produce valid output contracts (ascending positions, -1 padding, consistent valid_lengths) on every candidate path.
  - Negative Tests (expected to FAIL):
    - A change with recall@2048 delta >0.5pp must be reverted or fixed — banking its speedup fails the loop.
    - Any cross-rank selection mismatch fails hard, regardless of recall.
    - A pre-existing tie-semantics defect exposed by fixtures must be raised to the user — silently changing selection semantics fails.
- AC-3: Core DS concept intact — offline channel-importance mask → per-token signatures → query·signature scoring → top-k token selection → sparse MLA decode over the selected tokens.
  - Positive Tests (expected to PASS):
    - Code review of every landed change confirms all pipeline stages still execute on the DS-on decode path; the M0 harness exercises signature scoring end-to-end.
  - Negative Tests (expected to FAIL):
    - Any landed path that skips signature scoring, falls back to dense attention, or consumes DSA-indexer outputs for DS selection fails.
- AC-4: DSA-native default un-regressed.
  - Positive Tests (expected to PASS):
    - DS-off serving behavior is byte-identical (no DS code executes when the flag is off).
    - If any shared sgl-kernel/DSA kernel code is modified, the mandatory DSA regression (DS-off smoke + Case-2 re-validation) is run and its result recorded. Reuse without modification does not trigger it.
  - Negative Tests (expected to FAIL):
    - Any DS-off behavior change fails.
    - Modifying shared kernel code without running the mandatory regression fails.
- AC-5: Protocol and ledger discipline.
  - Positive Tests (expected to PASS):
    - Each profiling run is one trial; only Case 1 is re-profiled, reusing the exact recipe in development/profiling/plan.md; frozen Case 2/3 artifacts under development/profiling/runs/20260609/ are reused as references.
    - development/loop9/results.md carries one column per landed idea with: per-bucket µs, total µs, aggregate decode tok/s, recall-gate result, and (for the score-reduce change) the reduce backend.
  - Negative Tests (expected to FAIL):
    - Re-running Cases 2/3 without the AC-4 trigger fails protocol.
    - Averaging multiple trials into a single reported number fails.
    - An idea kept without its recall-gate record fails.

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.

### Upper Bound (Maximum Acceptable Scope)
The implementation lands the M0 harness plus all three kernel milestones with every per-bucket
hard gate met; both M2 candidates (the adapted sgl-kernel radix top-k AND a new DS-specific AOT
kernel) fully implemented, benchmarked head-to-head on captured Case-1 tensors, and the winner
integrated; the M4 memory audit performed with a successful, measured admission re-tune; a
wildcard selection-path redesign evaluated only if M1–M3 plateau short of the bar; a complete
per-idea ledger; and recorded follow-on notes for extending the new fast paths to the non-default
graph-safe DS variants in a later loop.

### Lower Bound (Minimum Acceptable Scope)
The implementation lands the M0 harness with the frozen oracle baseline and a successful protocol
dry-run; makes a genuine, documented attempt at each of M1, M2 (both candidates), and M3; every
landed change passes the AC-2 recall gate and the cross-rank hard check; per-bucket gates that
prove unreachable are downgraded to documented trend reductions per DEC-1; un-kept ideas are
reverted with reasons in the ledger; and the M4 audit is performed (re-tune only if the audit
shows recoverable GBs).

### Allowed Choices
- Can use: re-routing the DS score reduce through the GroupCoordinator custom-all-reduce dispatch;
  NCCL-shaping fallback levers (pre-flattened/contiguous reduce); bf16/lower-precision scoring or
  reduction levers gated by the AC-2 recall bound; adapting the sgl-kernel `fast_topk_v2` family;
  new DS-specific AOT kernels in sgl-kernel (per DEC-4), with Triton prototypes permitted for
  benchmarking; Triton-level optimization of `_logical_score_kernel`; NVTX instrumentation;
  mem-fraction/admission re-tuning if the M4 audit pays.
- Cannot use: signature coarsening (the loop-8 decision stands — int8 was the floor; coarser
  signatures are added lossiness); cross-layer batching of the score reduce (infeasible —
  layer-sequential dependency); silent fallback to dense attention or to DSA's indexer on the DS
  path; `PYTORCH_CUDA_ALLOC_CONF=expandable_segments` for serving (breaks custom-all-reduce IPC at
  GLM TP=8 graph capture); selection-semantics changes beyond the AC-2 bound; re-running frozen
  Cases 2/3 except under the AC-4 mandatory-regression rule; any change to DS-off default behavior.

> **Note on Deterministic Designs**: The measurement protocol is fixed per the draft (Case-1
> recipe reused verbatim from development/profiling/plan.md, one trial per run, frozen Case-2/3
> references); the implementation approach within each milestone is open within the bounds above.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

All of the following was verified against the code during plan convergence:

1. **Score reduce (M1).** Today the DS bind site (`python/sglang/srt/models/deepseek_v2.py`) binds
   `get_attention_tp_group().device_group` — the raw ProcessGroup — so both reduce call sites (the
   `all_reduce_token_scores` helper used by the eager paths, and the direct in-place reduce on the
   scratch score view inside the graph-safe selector path) dispatch to NCCL ring. The model's
   hidden-state reduces ride the GroupCoordinator's custom-all-reduce dispatch instead. The
   re-route: bind the GroupCoordinator (or an explicit all-reduce callable), cover both call sites
   with one reducible abstraction, and handle custom-AR's out-of-place result with a captured
   copy-back into the scratch buffer (zero replay allocations). The score tensor at bs 29
   (~534 KB fp32) is far below the 8 MB custom-AR cap. fp32 summation order differs between ring
   and one-shot, which can move scores by ~1 ulp — admissible under the AC-2 recall bound. Note:
   the flashinfer allreduce fusion path only exposes an allreduce+residual+RMSNorm pattern and
   cannot host a standalone SUM; cross-layer reduce batching is infeasible because each layer's
   top-k consumes that layer's reduced scores before the next layer runs.
2. **Top-k (M2).** Two candidates, both built per DEC-4: (a) adapt `fast_topk_v2` (sgl-kernel;
   its fixed topk=2048 matches DS's budget; needs a wrapper emitting ascending logical positions,
   -1 padding, and valid_lengths, plus boundary handling for its unspecified radix tie order);
   (b) a new DS-specific AOT kernel encoding the (score descending, position ascending) selection
   directly and emitting the sequence-ordered output contract in one pass. Benchmark both on
   captured Case-1 score tensors; integrate the winner.
3. **Logical score (M3).** Measure first — the Triton kernel already early-exits token blocks past
   seq_len, so wins must come from measured bottlenecks: block sizing for the label dimension,
   load coalescing on the [tokens, heads, label_dim] signature layout, validity-mask fusion, or
   fusing score output into top-k preparation.
4. **Memory (M4).** Audit-first — the KV pool is sized before DS table binding, so buffers freed
   by M1–M3 do not automatically lift the bs-29 cap; admission changes require an explicit,
   measured re-tune.
5. **Wildcard (M5, draft Idea 5).** Only if M1–M3 plateau short of the DEC-1 bar: a larger
   selection-path redesign informed by development/past_implementations/ (Twilight's selection,
   the original DoubleSparse kernels) — still DS-concept, still within the AC-2 bound.

### Relevant References
- development/profiling/results.md — the measured gap (frozen Cases 1–3, kernel-group breakdowns)
- development/profiling/plan.md — the exact Case-1 benchmark recipe to reuse verbatim
- development/profiling/runs/20260609/ — frozen artifacts and parsers (summarize_torch.py, summarize_nsys.py, compare_decode.py, run_case.sh, _env.sh)
- python/sglang/srt/layers/attention/double_sparsity/selection_kernel.py — `all_reduce_token_scores`, `_logical_score_kernel`, `select_topk_sequence_order`, `_topk_by_score_then_pos`, graph-safe selector path
- python/sglang/srt/layers/attention/double_sparsity/selector.py, token_label_table.py, token_label_write.py, cuda_graph.py, config.py — bind/runtime plumbing, table layout, graph scratch, config variants
- python/sglang/srt/layers/attention/double_sparsity/selection_recall_oracle.py — the NIAH recall instrument for AC-2 (it is NOT an index-equivalence checker)
- test/registered/unit/layers/attention/ — existing DS unit tests; home for the M0 harness fixtures
- python/sglang/srt/models/deepseek_v2.py — DS bind site (binds the raw device group today)
- python/sglang/srt/distributed/ — GroupCoordinator, custom all-reduce dispatch, graph-capture coordination
- python/sglang/srt/layers/attention/dsa_backend.py — DSA's fused indexer (the cost target)
- sgl-kernel/python/sgl_kernel/top_k.py and sgl-kernel/csrc/elementwise/topk.cu — `fast_topk_v2` family (reusable radix top-k, topk=2048)
- development/past_implementations/ — paper, study/, DoubleSparse, Twilight, sglang-last-with-double-sparsity
- CLAUDE.md — doctrine (surgical changes, fix the data structure not the symptom, prove with numbers)

## Dependencies and Sequence

### Milestones

1. **M0 — Lossiness harness, attribution, protocol dry-run** (gates everything)
   - Phase A: Freeze the oracle — capture the current production graph-replay selector output
     (selected_indices + valid_lengths) and the NIAH recall@2048 baseline on a fixed gated
     workload, before any optimization lands.
   - Phase B: Harness mechanics — graph-replay tests mutate pre-captured static input buffers via
     copy_ before replay and compare outputs read after replay (CUDA graphs capture tensor
     addresses, not call arguments). No naive standalone 8-rank NCCL graph-capture unit test:
     either ride the production capture coordination (cuda_graph_runner / parallel_state graph
     capture) for the all-rank check, or use single-rank graph replay plus an eager micro all-rank
     reduce-equality check plus the production dry-run. Adversarial fixtures run micro-sized; the
     full served-shape all-rank comparison runs once on the fixed workload.
   - Phase C: NVTX ranges around DS score-reduce / top-k / logical-score (attribution independent
     of kernel-name matching); protocol dry-run — one end-to-end Case-1 torch re-profile
     confirming run_case.sh, the parsers, and the frozen Case-2 diff still work.
2. **M1 — Score-reduce custom-AR re-route** (kills the +124,873 µs bucket)
   - Phase A (spike): feasibility evidence — does the served config alias attention-TP to the TP
     GroupCoordinator; which custom-AR (v1/v2) is active; does the [29, ~4608] fp32 tensor pass
     the custom-AR eligibility check (one-shot vs two-shot at TP=8); does it capture in the
     coordinator's graph-capture/registration context with zero replay allocations; does a
     micro-capture show a NAMED custom-AR kernel replacing the NCCL ring line. Exit: GO or NO-GO
     (fallback levers: pre-flattened/contiguous NCCL reduce; recall-gated bf16 reduce).
   - Phase B (implement): one reducible abstraction over BOTH call sites; GroupCoordinator (or
     callable) bound at bind_runtime_data with the raw ProcessGroup kept only as fallback;
     out-of-place custom-AR handled by captured copy-back; gates per AC-1.1 and AC-2.
3. **M2 — Fused top-k, two candidates** (kills the +138,602 µs bucket; per DEC-4)
   - Phase A: candidate A — adapt sgl-kernel `fast_topk_v2` with the DS output-contract wrapper.
   - Phase B: candidate B — new DS-specific AOT kernel in sgl-kernel.
   - Phase C: head-to-head benchmark on captured Case-1 score tensors; integrate the winner behind
     the selector (graph-safe, static shapes); gates per AC-1.2 and AC-2; mandatory DSA regression
     if shared kernel code was modified (AC-4).
4. **M3 — Logical-score optimization, measure-first** (targets the +63,107 µs bucket)
   - Phase A: per-kernel-instance profile on captured Case-1 shapes (occupancy, DRAM throughput,
     block-size sensitivity) to find the actual bottleneck.
   - Phase B: targeted optimization; fp32 score math preserved unless a numerics move passes the
     AC-2 gate; gates per AC-1.3 and AC-2.
5. **M4 — Penalty-B memory audit** (conditional per DEC-2)
   - Phase A: account label-table GB/rank, DS scratch GB/rank (including buffers freed by M1–M3),
     KV-pool token capacity, admitted decode batch at mem 0.7.
   - Phase B (only if Phase A shows recoverable GBs): explicit admission re-tune and re-measured
     batch cap. No signature coarsening.
6. **M5 — Wildcard redesign** (only if M1–M3 plateau short of the DEC-1 bar; draft Idea 5)
   - Selection-path redesign proposal informed by the reference implementations; still DS-concept,
     still within the AC-2 bound; implementation is a user decision at that point.

**Per-idea heartbeat (M1/M2/M3, and M4 if its re-tune runs):** implement → AC-2 gate → re-profile
Case 1 (torch capture, one trial; nsys only if attribution is unclear) → compare_decode.py against
frozen Case 2 → ledger column in development/loop9/results.md → keep-or-revert. Kept ideas stack;
the running Case-1 number becomes the next idea's baseline.

**Dependency shape:** M0 precedes everything. M1 Phase A precedes Phase B. M2 candidates depend
only on M0 (their benchmark uses captured tensors), but M2 integration lands after M1 so the
running-baseline ordering stays clean. M3 follows M2. The M4 audit follows M3 (it accounts buffers
freed by earlier milestones). M5 triggers only on plateau. Measurement serializes on the single
8xH200 node (one TP=8 server at a time); implementation work may overlap.

## Task Breakdown

Each task must include exactly one routing tag:
- `coding`: implemented by Claude
- `analyze`: executed via Codex (`/humanize:ask-codex`)

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | M0 harness: frozen production oracle capture (selected_indices + valid_lengths), NIAH recall@2048 baseline on the fixed gated workload, cross-rank equality check, adversarial micro-fixtures with graph-replay copy_ semantics, NVTX ranges around the three DS buckets | AC-2 | coding | - |
| task2 | M0 protocol dry-run: one end-to-end Case-1 torch re-profile; verify run_case.sh, parsers, and the frozen Case-2 diff | AC-5 | coding | task1 |
| task3 | M1 spike: custom-AR feasibility evidence for the DS score reduce (group aliasing, custom-AR v1/v2, eligibility at [29, ~4608] fp32, coordinator capture context, micro-capture named-kernel proof) | AC-1.1 | coding | task2 |
| task4 | M1 spike review: independent GO/NO-GO assessment of the spike evidence and fallback-lever choice | AC-1.1 | analyze | task3 |
| task5 | M1 implement: one reducible abstraction over both reduce call sites, GroupCoordinator binding, captured copy-back; AC-2 gate; Case-1 re-profile + ledger column (incl. reduce backend) | AC-1.1, AC-2 | coding | task4 |
| task6 | M2 candidate A: adapt sgl-kernel fast_topk_v2 — wrapper emitting ascending logical positions, -1 padding, valid_lengths, with boundary handling for the radix tie order | AC-1.2 | coding | task2 |
| task7 | M2 candidate B: new DS-specific AOT top-k kernel in sgl-kernel encoding (score desc, pos asc) selection with the sequence-ordered output contract | AC-1.2 | coding | task2 |
| task8 | M2 benchmark-off review: head-to-head numbers on captured Case-1 tensors plus complexity assessment; recommend the winner | AC-1.2 | analyze | task6, task7 |
| task9 | M2 integrate winner behind the selector (graph-safe); AC-2 gate; mandatory DSA regression if shared kernel code was modified; Case-1 re-profile + ledger column | AC-1.2, AC-2, AC-4 | coding | task5, task8 |
| task10 | M3 measure: per-kernel-instance profile of _logical_score_kernel on captured Case-1 shapes (occupancy, DRAM throughput, block-size sensitivity) | AC-1.3 | coding | task2 |
| task11 | M3 bottleneck review: interpret the kernel profile and rank optimization candidates | AC-1.3 | analyze | task10 |
| task12 | M3 implement the targeted optimization; AC-2 gate; Case-1 re-profile + ledger column | AC-1.3, AC-2 | coding | task9, task11 |
| task13 | M4 audit: label-table and DS scratch GB/rank (incl. buffers freed by M1–M3), KV-pool token capacity, admitted batch at mem 0.7; recoverable-GB verdict | AC-1 | analyze | task12 |
| task14 | M4 conditional re-tune: admission/mem-fraction adjustment and re-measured decode batch cap (only if task13 shows recoverable GBs) | AC-1 | coding | task13 |
| task15 | M5 wildcard proposal (only if per-bucket gates remain unmet after M1–M3): selection-path redesign options from the reference implementations, with expected-cost analysis | AC-1 | analyze | task12 |
| task16 | Close-out: consolidate the development/loop9/results.md ledger, final gap statement vs frozen references, follow-on notes (graph-safe variant support, AOT promotion, wildcard disposition) | AC-5 | coding | task12, task13, task14 |

## Claude-Codex Deliberation

### Agreements
- The lossiness harness (M0) must land first and gate every subsequent change.
- Per-bucket gates are primary and the run total secondary, because shared-kernel boot-to-boot
  variance (~27k µs, observed on MoE kernels between Case-1/Case-2 boots) can hide or fake
  progress on the total.
- M1 (all-reduce) before M2 (top-k), with a feasibility spike before implementation; the remaining
  technical unknowns (custom-AR v1 vs v2, one-shot vs two-shot at ~534 KB, attention-TP group
  aliasing) belong in the spike, not in user decisions.
- selection_recall_oracle.py is a host-side recall diagnostic, not an implementation-equivalence
  checker — the draft's verification plan needed the new M0 harness.
- Penalty B re-scoped audit-first: the KV pool is sized before DS table binding, so freed scratch
  does not automatically lift the bs-29 cap (draft Idea 4(a) corrected).
- Mandatory DSA regression whenever shared sgl-kernel/DSA code is modified — the frozen-reference
  premise does not survive a changed binary (this resolved what was briefly DEC-6).
- M3 is measure-first: the Triton score kernel already early-exits blocks past seq_len, so the
  draft's "skip work past seq_len" lever was already implemented.

### Resolved Disagreements
- **Draft Idea 1(a) (fold the score reduce into flashinfer allreduce fusion):** code verification
  showed the fusion exposes only an allreduce+residual+RMSNorm pattern — infeasible for a
  standalone SUM. Resolution: re-route through the GroupCoordinator custom-all-reduce dispatch
  instead. Codex round 1 sharpened this further: it is not a pure function swap — DS binds the raw
  device group today, there are TWO reduce call sites, and custom-AR is out-of-place while DS
  assumes in-place mutation. All three constraints are now in M1.
- **Draft Idea 1(b) (batch reductions across layers):** both sides found it infeasible — each
  layer's top-k consumes that layer's reduced scores before the next layer runs. Dropped.
- **All-reduce as first implementation vs feasibility spike:** first-pass Codex wanted a spike
  before committing; Claude wanted M1 first for payoff-per-effort. Resolution: M1 split into a
  spike (Phase A) and implementation (Phase B), ordered before M2.
- **M0 harness implementability:** round-2 Codex flagged the naive all-rank NCCL graph-capture
  trap and CUDA-graph replay-argument semantics. Resolution: production capture coordination or
  single-rank replay plus eager micro all-rank equality; static-buffer copy_ semantics; the frozen
  production output is the oracle (tie fixtures verify semantics, never redefine them).
- **M1 acceptance bar:** "NCCL ring line gone" alone was judged insufficient (a silent
  pynccl/torch-distributed fallback also removes the line). Resolution: named custom-AR kernel
  evidence required; fallback fails the gate; the ledger records the actual backend.
- **fast_topk_v2 drop-in suitability:** its radix tie order is unspecified and its tests tolerate
  non-exactness, so Codex required a prototype gate before committing. Superseded by user DEC-4:
  both candidates are built and benchmarked head-to-head.

### Convergence Status
- Final Status: `converged` (3 rounds; round-3 verdict: ready, no remaining required changes; the
  round-1 first-pass analysis, three review rounds, and all dispositions are recorded in this
  section and the decisions below)

## Pending User Decisions

All decisions were resolved by the user during the gen-plan discussion. None remain PENDING.

- DEC-1: Numeric success bar for AC-1
  - Claude Position: per-bucket hard gates (ring line eliminated + named custom-AR evidence;
    top-k stack ≤80k µs; logical-score ≤40k µs) with the total as a secondary trend marker
    (560k minimum / 516k strong).
  - Codex Position: same structure; originally framed as recovering ≥25% (minimum) / ≥40% (strong)
    of the Penalty-A tax, with the same per-bucket values.
  - Tradeoff Summary: total-only gates are noise-prone (~27k µs boot variance); hard bucket gates
    risk blocking the loop on a genuinely unreachable target.
  - Decision Status: **DECIDED — per-bucket hard gates, attempted in earnest; if a gate proves
    impossible, a documented trend reduction in that bucket is acceptable.**
- DEC-2: Stopping rule and M4 scope
  - Claude Position: run all of M1–M3 regardless of early bar-hits (independent payoffs); M4
    re-tune only if the audit shows recoverable GBs.
  - Codex Position: agreed (M4 as conditional audit, not a main optimization milestone).
  - Tradeoff Summary: stopping at the bar saves effort but leaves attributed buckets unoptimized.
  - Decision Status: **DECIDED — all of M1–M3; M4 re-tune conditional on the audit.**
- DEC-3: Lossiness / equivalence definition
  - Claude Position: bit-identical selected_indices + valid_lengths vs the frozen oracle, with
    fp32 sum-order changes admissible iff they pass that empirical bar.
  - Codex Position: bit-identical, with the numerics-admissibility question explicitly for the
    user; bf16 kept out of default fallbacks.
  - Tradeoff Summary: bit-identical is the safest bar but bans ulp-level reorder wins; a bounded
    recall-delta admits bf16/fp8 levers at quality risk.
  - Decision Status: **DECIDED — bounded recall-delta: NIAH oracle recall@2048 on the fixed gated
    workload moves ≤0.5 percentage points per landed change. Cross-rank selection agreement stays
    hard (bit-identical across ranks). Exact index diffs vs the frozen oracle are still recorded
    for attribution.**
- DEC-4: M2 build-vs-adapt and AOT allowance
  - Claude Position: prototype both, prefer a Triton JIT exact kernel, defer AOT promotion.
  - Codex Position: prototype timing gate first; prefer a DS-specific exact kernel if the
    fast_topk_v2 boundary pass gets complex.
  - Tradeoff Summary: adaptation reuses a proven radix kernel but fights its tie-order contract;
    a new kernel has a higher ceiling and higher build/CI cost.
  - Decision Status: **DECIDED — implement BOTH candidates fully (fast_topk_v2 adaptation AND a
    new DS-specific AOT sgl-kernel kernel) as separate tasks, benchmark head-to-head, integrate
    the winner. AOT additions are allowed this loop.**
- DEC-5: Config-variant support scope
  - Claude Position: optimize the served-default path only (scorer_norm=off, head_agg=max, anchors
    off, fp16 signatures, locked budget); other variants stay on the current torch path unchanged.
  - Codex Position: agreed; flagged the variant list explicitly so nothing silently breaks.
  - Tradeoff Summary: smallest surface and fastest payoff vs deferred coverage of cosine/hybrid/
    mean/anchors/int8/lifted variants.
  - Decision Status: **DECIDED — served default only; graph-safe variants must be explicitly
    supported in a later loop (recorded as a follow-on; the non-default variants keep riding the
    existing torch path with no behavior change this loop).**
- DEC-6: Frozen-reference exception when shared kernels change
  - Decision Status: **RESOLVED during convergence (no user input needed)** — modifying shared
    sgl-kernel/DSA code makes the DSA regression (DS-off smoke + Case-2 re-validation) mandatory;
    reuse without modification does not trigger it.

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", or similar workflow markers
- These terms are for plan documentation only, not for the resulting codebase
- Use descriptive, domain-appropriate naming in code instead

### Hardware / Op-Point Constraints (from the draft, binding for every run)
- Single node 8xH200, TP=8, FP8 e4m3, page 64, fp8 KV cache, custom-all-reduce ON.
- Never set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments` for serving — it breaks
  custom-all-reduce IPC at GLM TP=8 graph capture.
- One TP=8 server at a time; profiling runs serialize on the node.
- Model path, channel-mask path, and the full Case-1 DS config JSON are in
  development/profiling/plan.md and must be reused verbatim for every Case-1 re-profile.

## Output File Convention

This template is used to produce the main output file (e.g., `plan.md`).

### Translated Language Variant

When `alternative_plan_language` resolves to a supported language name through merged config loading, a translated variant of the output file is also written after the main file. Humanize loads config from merged layers in this order: default config, optional user config, then optional project config; `alternative_plan_language` may be set at any of those layers. The variant filename is constructed by inserting `_<code>` (the ISO 639-1 code from the built-in mapping table) immediately before the file extension:

- `plan.md` becomes `plan_<code>.md` (e.g. `plan_zh.md` for Chinese, `plan_ko.md` for Korean)
- `docs/my-plan.md` becomes `docs/my-plan_<code>.md`
- `output` (no extension) becomes `output_<code>`

The translated variant file contains a full translation of the main plan file's current content in the configured language. All identifiers (`AC-*`, task IDs, file paths, API names, command flags) remain unchanged, as they are language-neutral.

When `alternative_plan_language` is empty, absent, set to `"English"`, or set to an unsupported language, no translated variant is written. Humanize does not auto-create `.humanize/config.json` when no project config file is present.

--- Original Design Draft Start ---

# Loop 9 Draft — Close the DS-on decode-overhead gap on GLM-5.1 (perf loop)

> Written 2026-06-10, after **Loop 8 closed** (disk `development/loop8/` = roadmap Loop 10, GLM-5.1
> DS bring-up; AC-4 closed R12, DS landed default-OFF). That loop produced a one-batch profiling
> characterization (`development/profiling/`) that **attributes** exactly where DS-on decode spends
> its extra time vs DSA-native. This loop's whole job is to **drive that measured gap down**.
> Feed this through `gen-plan` once scope is confirmed. **This is a draft of a draft — open for
> improvement.**

---

## What this work is (and is NOT) — read first

**This is not frontier LLM development. There is no novelty here.** We are reproducing the results
of a **2-year-old paper** (Double Sparsity, arXiv:2408.07092, `development/past_implementations/double_sparsity_paper_2408.07092.pdf`)
on top of a **fully open-source codebase** (SGLang), and checking whether it can be made to perform
on **MLA models that are well behind the frontier** (GLM-5.1). The contribution, if
any, is engineering: making an existing sparse-attention idea cheap enough to be worth its opt-in
slot on a model that already ships a *trained* sparse indexer (DSA). This is purely educational and the algorithm theoretically has no hope of beating the highly optimized DSA algorithmn.theres a reason why the algorithm was introduced 2 years ago and nobody has actually decided to use it in a frontier model, we just want to see how it would perform and learn about basic performance engineering skills when doing so.

## Objective

Take the **measured DS-on decode overhead** from `development/profiling/results.md` and **close the
gap** against the DSA-native baseline — by attacking the DS-specific kernels directly. The scope of
a single idea may be anything from a **small perf fix or debug** up to a **full rewrite / large
refactor / algorithm redesign** of a key component — *provided* it:

1. **Keeps the core concept of Double Sparsity** (offline channel-importance mask → per-token
   signatures → query·signature scoring → top-k token selection → sparse MLA decode over the
   selected tokens). Reference implementations: `development/past_implementations/` (DoubleSparse,
   Twilight, sglang-last-with-double-sparsity, the paper, and the as-built study `study/`).
2. **Introduces no additional theoretical lossiness.** The set of tokens DS selects (and therefore
   its recall/quality) must be **equivalent to today's** — pure speed/memory wins only. Numerics may
   move only where they provably do **not** change selection (verify, don't assume — §"Guardrails").

**We are not chasing the SLO in this loop.** The target is the **Case-1 (DS-on) one-batch
profiling benchmark** we just built — nothing more. SLO re-validation, recall R&D, multi-node, knob
compat, nvfp4 — all deferred (roadmap untouched except to mark this the active direction).

---

## The gap we are closing (from `development/profiling/results.md`)

One-batch profiling (GLM-5.1-FP8, 8×H200 TP=8, `bench_one_batch_server`, ISL 4096 / OSL 512) gave
three **frozen reference points** (DSA does not change as we optimize DS — so we re-profile **only
Case 1** going forward):

| Case | Config | decode GPU-kernel µs / 10-step window (torch TP-0) | aggregate decode tok/s | role |
|------|--------|------|------|------|
| **1 — DS** | DS ON, mem 0.7, bs 29 | **632,239** | 459 | the thing we are making faster |
| **2 — DSA@same** | DSA, mem 0.7, bs 29 | 342,857 | 876 | **apples-to-apples target** (same bs/mem; only DS flags differ) |
| **3 — DSA@best** | DSA, mem 0.8, bs 64 | 422,236 | 1555 | best-DSA aggregate target |

The deficit is **two separable, multiplicative penalties**, and DS pays both:

### Penalty A — DS index/scoring tax ≈ 1.9× (Case 1 − Case 2 = **+289,382 µs** at the SAME bs 29)
DS does **not** add an index onto a dense path — it **replaces DSA's tightly-fused fp8 indexer
(~17.2k µs/window)** with a much heavier stack. The DS-specific additions (DSA issues **zero** of
these — confirmed: absent from Case 2/3 nsys kern_sums):

| DS-specific kernel group | Δ µs (Case1−Case2) | source site |
|---|---|---|
| `ncclDevKernel_AllReduce_Sum_f32_RING` (DS-only per-layer cross-TP score reduce) | **+124,873** (780 calls = 1/layer/step) | `all_reduce_token_scores` — `double_sparsity/selection_kernel.py:561` (called :1058,:1071,:1447) |
| generic PyTorch top-k/sort stack (mbtopk / radixSort / sbtopk<long> / scan_by_key / searchsorted) | **+138,602** | `select_topk_sequence_order` / `_topk_by_score_then_pos` — `selection_kernel.py:606`/`:585` |
| `_logical_score_kernel` (DS channel-score compute) | **+63,107** | `_logical_score_kernel` / `_logical_score_triton` — `selection_kernel.py:58`/`:1230` |

What DSA spends **instead** (and DS does not): `sm90_fp8_paged_mqa_logits` (6.9k) +
`topk_transform_decode_kernel` (7.7k) + `fast_hadamard_transform` (2.6k) ≈ **17.2k µs** — a fused
fp8 indexer ~**19× cheaper** than DS's logical-score + torch-topk + extra-all-reduce path. Core
sparse-MLA decode attention is ~31–42k µs in **both** Case 1 and Case 2 (clean parity) — so the gap
is the **index/scoring + its synchronization**, not attention math. Every DS-specific kernel fires
once per layer per step and **serializes** on the single CUDA-graph decode node sequence (added
critical-path latency, not hidden work).

### Penalty B — batch-efficiency ≈ 1.78× (DS is locked to bs 29)
DSA at bs 64 does only 1.23× the work for 2.2× the tokens (≈1.8× more GPU-efficient per token). DS
**can't reach bs 64**: its per-rank `TokenLabelTable` shrinks the KV pool and caps the decode batch
at ~29 (mem 0.7). This penalty is **memory-driven**, and shrinking the table further was already
rejected once (DEC-4, int8 was the floor) because coarser signatures **are** added lossiness — so
**Penalty A is the primary, clearly-lossless target**; Penalty B is secondary and only pursued via
**lossless** means (see Idea 4).

**Net:** DS bs29 459 → DSA bs29 876 → DSA bs64 1555 tok/s = **3.4× best-vs-best** ≈ 1.9× (A) × 1.78× (B).

---

## Candidate ideas (a menu — `gen-plan` picks/sequences; each is one implement→measure cycle)

Ordered by expected payoff-per-risk. Each idea names the pain-point kernel it kills and its lossiness
posture. **These are starting points, not a fixed list** — a better idea found while reading the code
or the reference impls replaces a worse one here.

1. **Kill the DS-only f32 ring all-reduce (Penalty A, +124,873 µs — biggest single item).**
   DS computes per-rank *partial* token scores (signatures are head/TP-sharded) then SUM-reduces them
   across the attention TP group in **fp32, once per layer per step** — 780 serialized NCCL calls
   DSA never makes. Levers, cheapest-first: (a) **fold it into the existing fused all-reduce path**
   (`enable_flashinfer_allreduce_fusion` is already ON — the trtllm fusion all-reduce shows up shared
   in both cases) instead of a standalone f32 ring; (b) **batch the reduction** — reduce all layers'
   scores in fewer, larger collectives instead of 78 separate ones; (c) **lower the reduce dtype**
   (fp32→bf16 halves traffic) — *numerics-sensitive, gate behind selection-equivalence*. (a)/(b) are
   lossless reshaping; (c) needs the recall-oracle check. *Scope: medium refactor of the
   score-reduce path.*

2. **Replace the generic PyTorch top-k/sort stack with a fused top-k (Penalty A, +138,602 µs).**
   `_topk_by_score_then_pos` does two `argsort`s + gathers over the full KV width to pick 2048 and
   re-sort to sequence order; this explodes into mbtopk/radixSort/sbtopk/scan/searchsorted. DSA does
   the equivalent in one `topk_transform_decode_kernel` (~7.7k µs). Lever: a **fused selection kernel**
   (adapt DSA's `topk_transform`, or a Triton/CUDA top-k that emits sequence-ordered indices directly)
   producing **bit-identical** selected indices to today's deterministic (value-desc, then pos-asc)
   tie-break. *Lossless if output indices match — verify against `selection_recall_oracle.py`.
   Scope: new kernel / large refactor of `selection_kernel.py` selection path.*

3. **Fuse the logical-score compute (Penalty A, +63,107 µs).**
   `_logical_score_kernel` computes query·signature scores as a separate Triton pass over the full
   table. DSA fuses scoring into `sm90_fp8_paged_mqa_logits`. Lever: fuse scoring into the
   selection kernel (compute-and-select in one pass, killing the intermediate score scratch), and/or
   keep signatures fp8 end-to-end so scoring rides the fp8 path DSA uses. *Numerics-sensitive (int8
   dequant, head-agg) — gate behind selection-equivalence. Scope: kernel redesign.*
   > Ideas 1–3 together target the full ≈+326k µs DS index/scoring subtotal. Even partial wins move
   > Case 1 toward Case 2's 342,857 µs floor. The ceiling for A alone is ~1.84×→~1.0× on the index tax.

4. **Lossless batch-efficiency wins only (Penalty B — secondary).**
   Do **not** coarsen signatures (that's added lossiness, DEC-4). Instead: (a) any scratch/device-buffer
   that Ideas 1–3 eliminate **frees KV-pool memory** → admits a larger decode batch at the same mem
   fraction — couple the wins; (b) audit the `TokenLabelTable` for **layout/padding waste** that can be
   reclaimed without changing what's stored. Measure whether the freed memory lifts the bs-29 cap.
   *Lossless by construction (no signature change). Scope: memory-layout debug + re-measure admission.*

5. **Wildcard — redesign from the reference implementations.**
   If profiling after Ideas 1–3 still shows DS structurally heavier than DSA, consider a larger
   redesign of the selection path informed by `development/past_implementations/` (Twilight's selection,
   the original DoubleSparse kernels) — still DS-concept, still lossless. Last resort; only if the
   incremental kernel fusions plateau short of the goal.

---

## The iterate→measure protocol (the heartbeat of this loop)

For **each** idea, run exactly one cycle:

1. **Implement** the idea (DS path only; DSA-native default must stay byte-identical when DS is off).
2. **Verify losslessness FIRST** (before trusting any speed number): the DS selection output must
   match the pre-change path — use `double_sparsity/selection_recall_oracle.py` / the dense-DS and
   within-budget sanity checks. If selected tokens changed (beyond provably-irrelevant numerics),
   the idea **fails the no-added-lossiness bar** — revert or fix, do not bank the speedup.
3. **Re-profile Case 1 only**, reusing the **exact** setup in `development/profiling/plan.md` (Case 1
   server args: `$COMMON_ARGS --mem-fraction-static 0.7 --enable-double-sparsity --double-sparsity-config "$DS_CONFIG"`,
   bs 29, ISL 4096 / OSL 512). **One trial per run.** Cases 2 & 3 are **frozen references — do NOT
   re-run them** (DSA is unchanged; reuse `development/profiling/runs/20260609/case2_dsa07/` and
   `case3_dsa08/`). Both captures (torch per-stage decode + nsys `--cuda-graph-trace=node`) as in the
   plan, or torch-only if a quick read suffices and nsys adds nothing for that idea.
4. **Read the gap.** Compare the new Case-1 decode-GPU-kernel breakdown against the frozen Case 2
   (342,857 µs) and Case 3. **Success = the gap closed**, and specifically the **targeted DS-specific
   kernel group shrank** (e.g. after Idea 1 the `AllReduce_Sum_f32_RING` line should be gone/folded;
   after Idea 2 the mbtopk/radixSort stack should collapse toward DSA's `topk_transform` cost). Record
   per-idea: Case-1 total decode µs, the targeted-kernel Δ, aggregate decode tok/s, and the
   selection-equivalence result.
5. **Keep or revert.** Keep only ideas that (a) close the gap and (b) pass the lossless check. Stack
   kept ideas; carry the running Case-1 number forward as the new DS baseline for the next idea.

Reuse the committed parsers from the profiling run (`development/profiling/runs/20260609/`):
`summarize_torch.py`, `summarize_nsys.py`, `compare_decode.py` — and `development/profile_ds.sh`'s
kernel-aggregation block. Write each idea's result into a running ledger
(`development/loop9/results.md`, create it) with the same kernel-breakdown table shape as
`development/profiling/results.md`, one column added per idea.

---

## Acceptance criteria (draft — `gen-plan` formalizes)

1. **Gap closed and measured.** Case-1 DS-on decode GPU-kernel time is **materially lower** than the
   632,239 µs baseline and **measurably closer to Case 2's 342,857 µs** (DSA@same bs/mem), attributed
   to the specific DS kernel groups that shrank. Each landed idea has a before/after Case-1 profile.
   *(`gen-plan` sets the numeric bar — e.g. a target % of the +289,382 µs Penalty-A tax recovered;
   stretch = approach the Case-2 floor / reach into Penalty-B via Idea 4.)*
2. **No added theoretical lossiness.** Every landed change has a recorded selection-equivalence check
   (recall oracle / within-budget / dense-DS) showing DS selects the same tokens as before. Any
   numerics change is shown not to alter selection.
3. **Core DS concept intact.** Still offline channel mask → signatures → score → top-k → sparse decode.
   No silent fallback to dense or to DSA's indexer.
4. **DSA-native default un-regressed.** DS-off path byte-identical; the shipped model untouched.
5. **One trial per profiling run; Case 1 only re-profiled** (Cases 2/3 reused frozen). Ledger in
   `development/loop9/results.md`.

## Files to read first

- **The gap (the whole point):** `development/profiling/results.md` (+ artifacts under
  `development/profiling/runs/20260609/`, the `cmp_case1_vs_case2.txt` / `cmp_case2_vs_case3.txt` diffs).
- **The benchmark to reuse verbatim (Case 1 only):** `development/profiling/plan.md`.
- **The DS selection path to optimize:** `python/sglang/srt/layers/attention/double_sparsity/selection_kernel.py`
  (`all_reduce_token_scores`, `_logical_score_kernel`, `select_topk_sequence_order`), plus
  `token_label_table.py`, `token_label_write.py`, `selector.py`, `selection_recall_oracle.py`.
- **DSA's fused indexer (the cost target to emulate):** `python/sglang/srt/layers/attention/dsa_backend.py`
  (`sm90_fp8_paged_mqa_logits`, `topk_transform_decode`).
- **Lossless bar / DS concept:** `development/past_implementations/` (paper, study/, the three impls).
- **Doctrine:** `CLAUDE.md` (surgical changes, fix the data structure not the symptom, prove with numbers).

## Hardware / op-point

Single node 8×H200, TP=8, FP8 e4m3, page 64, fp8 KV, custom-all-reduce ON. **Never set
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments` for serving** (breaks custom-all-reduce-v2 IPC at GLM
TP=8 graph capture — BL-20260608). One TP=8 server at a time. Model + mask paths and the full Case-1
`DS_CONFIG` are in `development/profiling/plan.md`.

## Pending decisions (resolve in `gen-plan` discussion)

- **The numeric success bar for AC-1** — what fraction of Penalty A (the +289,382 µs / 1.9× tax) must
  be recovered to call the loop done? (Suggest: a concrete µs/ratio target on the Case-1 decode total
  and on each targeted kernel group, not a vibe.)
- **Ordering / stopping** — do all of Ideas 1–3, or stop once the bar is hit? Is Idea 4 (Penalty B)
  in scope this loop or its own follow-on?
- **Lossless-equivalence definition** — bit-identical selected indices, or a bounded recall-delta the
  user accepts as "no theoretical lossiness"? (Default: bit-identical selection; numerics free to move
  only where selection provably doesn't.)
- **Build vs adapt for Idea 2/3** — write a new fused DS kernel, or adapt DSA's `topk_transform` /
  `mqa_logits` to the DS signature layout? (DS signatures differ from DSA's indexer; check feasibility.)

--- Original Design Draft End ---
