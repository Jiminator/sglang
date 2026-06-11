# Loop 10 Plan — Close the DS-on Decode Dead-Width Tax

## Goal Description

Reduce the frozen Case-1 DS-on decode GPU-kernel time (GLM-5.1-FP8, 8×H200 TP=8, mem-fraction 0.7, batch size 29, ISL 4096 / OSL 512, measured as µs per 10-step decode window at torch TP-0, one trial per run, scoring method identical to Loop 9) from the loop-9 R1 baseline **480,989 µs (1.403×)** materially toward the frozen Case-2 DSA floor **342,857 µs (1.0×)** — by eliminating the dead-width tax: under CUDA graphs every DS selector tensor is shaped by the static `req_to_token` width (`[bs, 202752]`) while the served op point has ≤4,608 live tokens per row, so ~98% of reduced bytes are dead. The primary lever is width-bucketed DS selector graphs with compact per-bucket score buffers (the loop-9 M5 rank-1 proposal — this loop IS the user decision that proposal asked for), followed by transport/cast residuals, with conditional top-k and logical-score follow-ups. Zero added lossiness is a hard bar; the DSA-native default must stay un-regressed in behavior and performance. Full parity with the DSA floor is NOT the bar: the last ~0.10–0.15× is DS's intrinsic TP=8 obligation (heads are sharded, so DS pays a per-layer cross-rank score exchange DSA never pays).

Verified code facts the plan is built on (all confirmed in-repo during plan convergence):

1. Selector width is static at capture: `retrieve_topk_graph_safe` takes a Python-int `max_seq_len`; all `DSGraphState` scratch (`scratch_scores` fp32, `scratch_scores_bf16`, `scratch_pv_mask`, radix block scratch) is sized `[max_bs, max_seq_len]` from `req_to_token.shape[1]` in `python/sglang/srt/layers/attention/double_sparsity/cuda_graph.py` and `python/sglang/srt/layers/attention/dsa_backend.py`. One captured graph cannot change width by metadata swap — width variants require separate graph captures.
2. Selection outputs are LOGICAL positions (0..seq_len−1, row-relative), mapped to physical slots via `req_to_token` in `python/sglang/srt/layers/attention/double_sparsity/page_table_adapter.py`. A prefix-window compact buffer [0, W) preserves logical positions exactly — **no compact→logical inverse mapping is needed** (draft risk (c) collapses to "buckets must be prefix windows").
3. Custom-AR v2 on 8 ranks routes one-shot only ≤160 KB (`python/sglang/srt/distributed/device_communicators/custom_all_reduce_v2.py`). At padded graph bs 32, bf16 compact buffers are 320 KB (W=5120) / 512 KB (W=8192) → **two-shot by default**; at small bs buckets (e.g. bs≤16 × 5120) the same buffer drops under the threshold and would silently flip to one-shot — an algorithm flip changes floating-point summation order, so it must be pinned or declared (see exactness regimes). `should_custom_ar` requires weak-contiguity: a strided `[:bs, :bucket]` view FAILS it and silently falls back to NCCL — compact buffers must be real allocations.
4. The reduce today: fp32→bf16 cast kernel, two-shot custom-AR at 11.76 MB (~106–120 µs/call including cross-rank skew absorption; 780 calls per 10-step window = 93,480 µs, 13.3% of all GPU-kernel time on the nsys timeline), copy back to fp32 — the casts are the ~15–18k µs/window tax.
5. Graph keys are int bs (`CudaGraphRunner.graphs`, `output_buffers`; PDMux uses a `"{stream}_{bs}"` string — an existing key-extension precedent); the bs bucket is picked by bisect over `capture_bs` BEFORE backend metadata init, where `forward_batch.seq_lens_cpu` is host-visible → width dispatch with no GPU sync is feasible. DSA decode metadata is keyed `decode_cuda_graph_metadata[bs]` — same-bs multi-width captures would overwrite state unless metadata lifetime adopts the full graph key.
6. Capture is largest-bs-first over the ladder into a single shared global graph memory pool; each (bs, width) variant is a whole-model graph capture. Loop-9 M4 audit: capture-memory headroom exists (~14.2 GiB recoverable at a bs≤64 ladder; compact DS scratch is ~25–40× smaller than full width) — budget must still be measured, not assumed, since DEC-2 chose whole-ladder coverage.
7. `_logical_score_kernel` and the radix top-k kernels already bound work by per-row `seq_lens` (live-block loops, early exit) → width-bucketing gains there are grid/launch/scratch shrink only; the dominant win is transport bytes + casts. The shipped radix suite is 5 distinct kernels, ~11 launches per call.
8. Overflow hazard: a width-W graph silently truncates scoring at W (`n_live = min(seq_len, W)`) — bucket-dispatch correctness is a hard exactness requirement; overflow must route to a full-width graph.
9. Gate-tooling state inherited from loop 9: the R1 selcap digest is bs-1 sequential traffic only (the bs-29 op-point replay path is never exercised); the selcap `diff` mode is diagnostic-only, not a hard gate; the NIAH recall oracle runs EAGER (`--disable-cuda-graph`) and can never see graph-only changes — graph-path exactness rests entirely on selection-capture gates.

Change classification (exactness regimes — every queue task declares its regime before landing):

- **Exact-by-design** (layout/shape/keying/bucketing with the same reduce algorithm and dtype): MUST prove zero logical-index diff vs the pre-change state (selcap diff promoted to hard gate) plus cross-rank bit-identity plus recall. The width-bucketing milestone is in this regime, with the custom-AR algorithm pinned to two-shot for all compact variants.
- **Value-affecting** (reduce algorithm/dtype/summation-order changes — e.g. two-shot→one-shot override, NCCL swap, producer-side bf16, threshold-induced algorithm flips): permitted only under the loop-9 bf16 precedent — cross-rank bit-identity HARD + recall@2048 ±0.5pp fail-closed + explicit declaration in queue.md and the ledger; the selcap diff is recorded as evidence (expected nonzero), not as a pass/fail gate for that change.
- A planned-exact change that shows nonzero diffs FAILS its gate: revert, or re-classify with a user-visible ledger entry — never a silent downgrade.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.

- AC-1: Gap closed and attributed — Case-1 decode GPU-kernel time ≤420,000 µs (HARD, ~1.23×), stretch ≤395,000 µs (~1.15×, promoted to hard only if the transport bucket lands ≤45k), attributed per-bucket vs the R1 column and the Case-2 floor (per DEC-1 of loop 9: per-bucket attribution primary, totals secondary — boot-to-boot shared-kernel variance up to ~27k µs).
  - Positive Tests (expected to PASS):
    - Final Case-1 torch profile (frozen recipe verbatim via `development/profiling/runs/20260609/run_case.sh <out> case1 torch 29`, one trial) summarized with `summarize_torch.py`/`compare_decode.py` shows decode GPU-kernel total ≤420,000 µs.
    - The close-out ledger attributes the reduction per-bucket (transport, logical-score, top-k) with per-call µs recorded alongside window totals.
  - Negative Tests (expected to FAIL):
    - A close-out claiming AC-1 with total >420,000 µs, or with totals-only evidence and no per-bucket attribution, fails review.
    - A claim resting on an eager microbenchmark or a re-run frozen reference (instead of the captured-replay frozen recipe) fails — the CUDA-graph captured replay number is the binding one.
  - AC-1.1: Score-reduce transport bucket — whichever reduce kernel(s) serve DS score reduce (custom-AR or NCCL) PLUS associated fp32↔bf16 cast/copy kernels — ≤60,000 µs (HARD), stretch ≤45,000 µs.
    - Positive: bucket sum over the named reduce kernel(s) and cast/copy kernels ≤60k in the final profile, with per-bucket evidence logging the actual reduce algorithm, dtype, input shape, bytes, and contiguity.
    - Negative: a bucket claim that excludes the cast/copy kernels from the sum fails; an undeclared AR-algorithm flip (e.g. threshold-induced one-shot at small bs) fails the change-classification gate.
  - AC-1.2: Logical-score bucket (`_logical_score_kernel`) ≤20,000 µs (HARD), stretch ≤15,000 µs.
    - Positive: final profile bucket ≤20k.
    - Negative: regression of this bucket above its R1 value (36,908 µs) in any kept change fails the keep-or-revert step.
  - AC-1.3: DS top-k bucket (radix suite or successor) ≤28,000 µs (HARD), stretch ≤24,000 µs.
    - Positive: final profile bucket ≤28k with the deterministic tie-break contract intact.
    - Negative: a top-k replacement that misses the bucket bar or alters the tie-break contract fails.
- AC-2: No extra lossiness (HARD, per landed change).
  - AC-2.1: recall@2048 within ±0.5pp of the frozen `development/loop9/runs/20260610_m0/recall_baseline.json`, fail-closed, per landed change (eager oracle: `development/loop7/niah_oracle_sweep.py` + `development/loop9/oracle_recall_summary.py`). This is the algorithm-level check; graph-path exactness rests on AC-2.2/AC-2.3.
    - Positive: oracle sweep at frozen settings reports overall and per-length deltas within ±0.5pp.
    - Negative: any length bucket beyond ±0.5pp → the change is reverted (fail-closed); a landed change with no recall artifact fails evidence pre-flight.
  - AC-2.2: cross-rank selection bit-identity (selcap verify, graph mode, 8 ranks) on EVERY gate run.
    - Positive: `selection_capture_tool.py verify` passes cross-rank identity and contract validation.
    - Negative: any rank divergence is a hard fail regardless of recall outcome.
  - AC-2.3: exact-by-design changes prove bit-identical logical selection indices vs the pre-change state — selcap diff PROMOTED TO HARD GATE (zero differing indices), at BOTH the existing bs-1 digest (diff target: `development/loop9/runs/20260611_r1/selcap_digest.json`) AND a new op-point gate: a harness that drives bs-29 concurrent decode under graph replay and records selection captures tagged with bucket identity (graph key, selector width, raw bs, padded bs, max real seq_len). Building this harness is an explicit task (task1) with its own acceptance check: it must prove raw_bs=29, the padded graph bs, the selected width bucket, and the replay path.
    - Positive: zero-index-diff reports vs both pre-change digests for every exact-classified landed change; capture mirrors record logical indices.
    - Negative: nonzero diff on an exact-classified change fails its gate (revert or explicit re-classification); a diff tool run that passes structurally while indices differ must be treated as a gate failure (this is the hard-gate promotion).
  - AC-2.4: bucket-boundary correctness — tests cover seq_len == W (compact route), seq_len == W+1 (MUST route to the full-width graph AND prove bit-identical selection vs pre-change full-width behavior), the served 4096→4608 window growth across a decode, and padded-row behavior; plus a runtime contiguity assertion on the exact tensor handed to custom-AR.
    - Positive: all boundary tests pass; the contiguity assertion holds on the compact buffer actually passed to the reduce.
    - Negative: a constructed overflow (seq_len > largest compact W) that takes a compact graph (silent score truncation) must be caught by the test as a failure; a compact buffer that silently falls back to NCCL due to failed weak-contiguity fails the transport-evidence check.
- AC-3: DS concept intact — offline channel mask → per-token signatures → query·signature scoring → top-k selection → sparse MLA decode.
  - Positive: the landed pipeline still flows through these stages; config semantics (`python/sglang/srt/layers/attention/double_sparsity/config.py`) unchanged.
  - Negative: any dense fallback or DSA-indexer substitution inside the DS path fails review.
- AC-4: DSA-native default un-regressed — STRICTER THIS LOOP (shared CUDA-graph runner is touched).
  - AC-4.1: executable invariants, not claims: DS-off graph keys and code paths unchanged (keying unit test: DS-off graphs keyed exactly as today — plain int / existing PDMux string; width-key logic structurally unreachable when DS is off); PDMux, speculative-decode, and encoder paths untouched; no DS allocations when DS is off.
    - Positive: keying/invariant tests pass DS-off; boot memory check shows no DS scratch DS-off.
    - Negative: a tuple/width key observed in any DS-off path, or DS graph state allocated DS-off, fails.
  - AC-4.2: any round touching shared capture/replay code (`python/sglang/srt/model_executor/cuda_graph_runner.py`, shared `dsa_backend.py` surfaces) runs DS-off smoke + a FRESH Case-2-recipe regression run in the SAME round, landing in loop10 run dirs, compared per-bucket vs the frozen 342,857 µs floor within the loop-9 DEC-1 noise band (~27k µs on shared kernels). The frozen floor number itself is never replaced or re-baselined. This applies per independently-landed patch: if the keying patch (task4) lands separately from the compact patch (task5), each landing gets its own same-round regression.
    - Positive: same-round Case-2 regression artifact exists in loop10 dirs, per-bucket within noise vs the frozen floor; DS-off smoke passes.
    - Negative: a round that touched the shared runner with no same-round Case-2 artifact fails the round gate; any edit that replaces the frozen floor fails.
- AC-5: Protocol, ledger, and queue discipline.
  - Positive: one trial per run; only Case 1 re-profiled (Case-2 runs exist ONLY as AC-4.2 regression artifacts); frozen references reused, never re-run; `development/loop10/queue.md` populated as the loop's FIRST runtime task (seeded from this plan's task table plus kickoff ideas) and kept current every round — entries carry id, description, targeted bucket, expected effect, lossiness posture/regime, compatibility note vs landed changes, status; new mid-loop ideas appended with compatibility notes; drops recorded with measured/reasoned cause; queue committed with each round; ledger `development/loop10/results.md` rewrite-over-append with one authoritative current-state section; evidence pre-flight before each round handoff (artifact exists + is tracked + claim matches artifact; `git add -f` cited evidence past the `*.log` gitignore); deviations logged in the goal tracker's Plan Evolution Log.
  - Negative: silent queue deletions fail; a cited-but-untracked artifact fails pre-flight; queue population attempted during plan generation (rather than at loop kickoff) violates the draft's explicit instruction; re-running any frozen reference fails.

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.

### Upper Bound (Maximum Acceptable Scope)

The implementation may completely rewrite the DS selection pipeline and its CUDA-graph-runner integration (per the user's explicit scope statement: any scale of refactoring is welcome): a `(bs_bucket, width_bucket)` graph identity for DS-on decode carried end-to-end (graph dict, output buffers, replay lookup, capture order, DSA metadata lifetime, DSGraphState ownership) across the WHOLE bs ladder (per DEC-2); real compact per-width selector scratch; pinned-transport compact reduce with cast elimination; bf16-authoritative top-k input; a conditional multi-block top-k redesign; hardened gate tooling (op-point selcap harness, diff-as-hard-gate, bucket-identity-tagged digests); full bucket-boundary test coverage; measured capture-memory budget; and mid-loop queue-fed optimization ideas pursued under the same gates. New ideas that beat the menu may replace it.

### Lower Bound (Minimum Acceptable Scope)

The implementation includes gate hardening (op-point selcap harness + diff hard-gate promotion) and the width-bucketed selector graphs with one compact bucket plus the guaranteed full-width fallback across the bs ladder, with the custom-AR algorithm pinned two-shot, PLUS whatever transport/cast residual work is needed to bring the Case-1 total to the AC-1 hard bar (≤420k) — all under the full AC-2/AC-4/AC-5 gate suite. The conditional top-k redesign and the logical-score follow-up may be skipped if their buckets already meet their hard bars.

### Allowed Choices

- Can use: Triton kernels (default delivery form); restructuring/renaming/splitting any file under `python/sglang/srt/layers/attention/double_sparsity/`; runner changes in `cuda_graph_runner.py` gated to DS-on decode; per-instance custom-AR algorithm overrides (pin/override) isolated to the DS score reduce — never touching default model collectives; NCCL as a measured transport alternative (declared regime); compact bucket sizes other than 5120 if boundary evidence justifies (W must cover the served 4,608-token window, remain a prefix window, and satisfy custom-AR size/alignment constraints); nsys captures for timeline-shaped questions (kernel-name attribution — NVTX does not fire under graph replay).
- Cannot use: dense attention fallback or DSA-indexer substitution inside the DS path (AC-3); approximate top-k or any selection change without an exactness proof (the fused score+select wildcard stays out unless ideas 1–4 plateau AND an exactness proof exists; worst case it expands to full width); device-computed bucket dispatch (no GPU sync before replay — host-visible scheduler metadata only); strided views as custom-AR inputs (weak-contiguity failure → silent NCCL fallback); compaction of DSA attention metadata or page-table mapping (`max_seq_len_k`, `page_table_1`, page-table transform inputs stay full-context); force-installing a rebuilt sgl-kernel wheel over the frozen-reference binary without the AC-4 gate; `PYTORCH_CUDA_ALLOC_CONF=expandable_segments` for serving (breaks custom-all-reduce-v2 IPC); re-running frozen references (Cases 2/3 recipes run only as AC-4.2 regression artifacts; loop-9 run dirs are read-only).
- Fixed per draft: the scoring method, the frozen Case-1 recipe (`development/profiling/plan.md` + `runs/20260609/_env.sh`), the frozen baselines (480,989 / 342,857), the three losslessness teeth and their tools, one TP=8 server at a time, single node (multi-node out of scope), SLO re-validation and recall R&D and the bs-64 re-tuned op point out of scope.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

One possible path (matches the task breakdown):

1. Harden the exactness gates first. Extend the loop-9 selection-capture tooling with an op-point harness (bs-29 concurrent decode under graph replay; dumps and digests tagged with graph key, selector width, raw/padded bs, max real seq_len) and promote the diff step from diagnostic to hard-fail in the gate script (clone of `development/loop9/run_r1_gates.sh`). Freeze pre-change digests (bs-1 and op-point) before touching the pipeline.
2. Land width-bucketing as two reviewed patches.
   - Patch one (keying/lifetime, zero behavior change): introduce the `(bs_bucket, width_bucket)` identity through the runner and DSA metadata with ONLY full-width variants captured — the exact gate must show zero index diffs; DS-off invariants (AC-4.1) land here. "Owns" means stable keyed lifetime for the `DSAMetadata`/`DSGraphState` objects per captured variant — not duplication of full-context DSA backing tensors.
   - Patch two (compact allocation + dispatch): real compact per-width DSGraphState buffers (weak-contiguous by construction); width bucket chosen at the existing pre-replay bisect site as the smallest captured W ≥ `forward_batch.seq_lens_cpu[:raw_bs].max()` (real rows only — never padded metadata slices); overflow routes to the always-captured full-width graph; custom-AR algorithm pinned two-shot for all compact variants; bucket-boundary tests per AC-2.4. Capture compact variants ladder-wide (DEC-2), measure capture memory and boot time before/after, and budget against the M4-audit headroom.
3. Attack transport residuals on the now-compact buffers: eliminate the fp32↔bf16 cast pair by fusing into producer/consumer or making post-reduce bf16 the authoritative top-k input — classified exact ONLY if top-k sees exactly the same reduced bf16 values as today's copy-back fp32 path (bf16→fp32 is value-preserving, so ordering is isomorphic; the radix kernel needs a 2-round bf16 path) and the selcap diff is zero; otherwise it is value-affecting and gates under the recall regime. Measure per-bucket transport choices (pinned two-shot vs declared one-shot override vs NCCL) — loop-9 spike-bench evidence suggests compact bf16 custom-AR can lose to NCCL at `[29,4608]`-class shapes, so build the loser before issuing a verdict.
4. Only if the top-k bucket is still above its bar: multi-block single-launch Triton redesign on compact rows (several blocks per row + cross-block coordination, targeting the measured 17.7 µs/call floor), preserving the deterministic tie-break contract bit-exactly. The AOT path and its env-gated tests exist from loop 9 but a wheel install triggers the full AC-4 gate — stay Triton-first.
5. After every landed task, re-profile Case 1 with the frozen recipe and read where the bottleneck moved; a shifted bottleneck is a queue-feeding event.

Projected landing (informative, NOT a gate; task2 re-derives before implementation): transport 108–111k → ~35–55k; logical-score 36.9k → ~15–20k; top-k 36.3k → ~20–28k; total ~367–403k (1.07–1.18×). The ≤420k hard bar has margin; ≤395k is stretch unless transport lands ≤45k. These re-rate the loop-9 M5 projection (made against the M2-era 512.7k baseline, and assuming one-shot AR that the 160 KB threshold rules out at op-point compact sizes).

### Relevant References

- `development/loop9/results.md` — current state, per-bucket R1 residuals, follow-on definitions, structural headroom items, loop-9 DEC-1 noise band.
- `development/loop9/reviews/task15_m5_wildcard_proposal.md` — the design being executed (rank-1 "B + C-lite"), its risk list and projections.
- `development/loop9/runs/20260611_nsys/nsys_vs_baseline.md` — fresh timeline evidence (two-shot reduce 13.3% of GPU time, 106 µs/call incl. skew; NVTX-under-replay caveat).
- `python/sglang/srt/layers/attention/double_sparsity/selection_kernel.py` — `reduce_token_scores`, `_logical_score_kernel`, `retrieve_topk_graph_safe`.
- `python/sglang/srt/layers/attention/double_sparsity/topk_kernel.py` — radix suite and deterministic tie-break contract.
- `python/sglang/srt/layers/attention/double_sparsity/cuda_graph.py` — `DSGraphState`, scratch bundles, graph-safe checks.
- `python/sglang/srt/layers/attention/double_sparsity/selector.py`, `config.py`, `page_table_adapter.py` — selector surface, config knobs, logical→physical mapping.
- `python/sglang/srt/model_executor/cuda_graph_runner.py` — bs-bucket ladder, capture/replay dispatch, graph keying, global memory pool.
- `python/sglang/srt/layers/attention/dsa_backend.py` — `allocate_graph_state` call sites, `decode_cuda_graph_metadata`.
- `python/sglang/srt/models/deepseek_v2.py` — `_select_topk_indices` bind site.
- `python/sglang/srt/distributed/device_communicators/custom_all_reduce_v2.py` — one-shot/two-shot thresholds, `override_algo`, weak-contiguity check.
- `sgl-kernel/csrc/elementwise/ds_topk.cu`, `sgl-kernel/python/sgl_kernel/top_k.py` — loop-9 AOT op (conditional path only); build recipe in BL-20260611-sgl-kernel-build-cuda13-cccl (`.humanize/bitlesson.md`) + `development/loop9/runs/20260611_r1/sgl_kernel_build.log`.
- Gate runners to clone: `development/loop9/run_r1_gates.sh`, `development/loop9/selection_capture_tool.py`, `development/loop9/oracle_recall_summary.py`, `development/loop7/niah_oracle_sweep.py`.
- Frozen recipe: `development/profiling/plan.md`, `development/profiling/runs/20260609/run_case.sh`, `_env.sh`.
- Doctrine: `CLAUDE.md`, `.pensieve/` maxims; subagent usage discipline per the draft (use Explore/implementation/analysis subagents liberally for context discipline, but every subagent product is reviewed in full by the main agent and passes the same gates).

## Dependencies and Sequence

### Milestones

1. Milestone M0 — Gate hardening and kickoff
   - Phase A: populate `development/loop10/queue.md` from this plan's task table plus kickoff ideas (the loop's FIRST runtime task; never during plan generation).
   - Phase B: op-point selcap harness + diff hard-gate promotion; freeze pre-change bs-1 and op-point digests (task1).
   - Phase C: projection re-derivation and transport model vs 480,989, including the loop-9 spike-bench evidence (task2).
2. Milestone M1 — Width-bucketed DS selector graphs (PRIMARY; exact regime; pinned two-shot)
   - Phase A: design dossier — width-key contract instantiation, ladder-wide coverage plan, capture-memory budget, overflow/fallback semantics, capture order (task3).
   - Phase B: keying/metadata-lifetime patch, full-width variants only, zero behavior change proven (task4).
   - Phase C: compact allocation + dispatch + boundary tests (task5).
   - Phase D: M1 gate run — full AC-2 suite + same-round AC-4 DSA regression + profile + per-bucket gap read; keep-or-revert (task6).
3. Milestone M2 — Transport residuals on compact buffers
   - Phase A: cast elimination / bf16-authoritative top-k input, regime-classified (task7).
   - Phase B: per-bucket transport choice measured and classified (pinned two-shot vs declared one-shot vs NCCL), with reduce dtype + actual algorithm logged per bucket (task8).
4. Milestone M3 — CONDITIONAL top-k redesign (only if the top-k bucket remains above AC-1.3 after M1+M2): Triton-first multi-block single-launch on compact rows (task9).
5. Milestone M4 — Logical-score residual: expected to largely fall out of M1; measure, act only if above AC-1.2 (folded into task6/task10 gap reads; a dedicated queue task is appended only if needed).
6. Milestone M5 — Close-out: final attribution review, ledger, evidence pre-flight (task10).

Per-task heartbeat (every queue task, exactly one cycle): implement → losslessness teeth FIRST (selcap bs-1 + op-point, cross-rank, recall) → profile Case 1 with the frozen recipe (nsys added when the question is timeline-shaped) → per-bucket gap read vs the R1 column and the Case-2 floor → keep-or-revert (bank only what passes and shrinks its targeted bucket) → queue.md update → ledger rewrite. Measurement discipline carried from loop 9: eager microbenches measure host JIT dispatch — the captured-replay number is binding; run seq≈0 floor probes before believing a hypothesis; build the loser before issuing a comparison verdict.

## Task Breakdown

Each task must include exactly one routing tag:
- `coding`: implemented by Claude
- `analyze`: executed via Codex (`/humanize:ask-codex`)

`development/loop10/queue.md` is seeded from this table at loop kickoff (first runtime task) and remains the single source of truth thereafter; mid-loop ideas are appended there as queued entries with compatibility notes, not absorbed into running tasks.

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Op-point selcap harness (bs-29 graph replay, bucket-identity-tagged dumps/digests) + promote selcap diff to hard gate in the cloned gate script + freeze pre-change bs-1 and op-point digests | AC-2.3 | coding | - |
| task2 | Re-derive projection + transport model vs 480,989 (two-shot at compact sizes, threshold-flip map across the bs ladder, cast-tax accounting, spike-bench evidence) | AC-1 | analyze | - |
| task3 | Width-bucketing design dossier: (bs,width) key contract end-to-end, ladder-wide coverage + capture-memory budget, overflow/fallback semantics, capture order | AC-1, AC-2, AC-4 | analyze | task2 |
| task4 | Keying/metadata-lifetime patch: (bs,width) identity through runner + DSA metadata + DSGraphState ownership, full-width variants only (zero behavior change; exact gate = zero diffs); DS-off invariants | AC-2.3, AC-4.1 | coding | task1, task3 |
| task5 | Compact patch: real per-width DSGraphState buffers, real-row host dispatch, full-width overflow fallback, pinned two-shot AR, bucket-boundary tests, contiguity assertion | AC-1.1, AC-2 | coding | task4 |
| task6 | M1 gate run: full AC-2 suite + same-round DSA regression (AC-4.2) + frozen-recipe profile + per-bucket gap read; keep-or-revert | AC-1, AC-2, AC-4 | coding | task5 |
| task7 | Cast elimination / bf16-authoritative top-k input, regime-classified (exact only if selcap diff is zero) | AC-1.1, AC-2 | coding | task6 |
| task8 | Per-bucket transport choice: measure pinned two-shot vs declared one-shot override vs NCCL at compact sizes; log reduce dtype + actual algorithm per bucket; classify each candidate BEFORE landing | AC-1.1, AC-2 | analyze | task6 |
| task9 | CONDITIONAL top-k redesign (only if top-k bucket > AC-1.3 after M1+M2): Triton multi-block single-launch on compact rows, tie-break contract preserved bit-exactly | AC-1.3, AC-2 | coding | task6, task8 |
| task10 | Close-out: final attribution review, results.md authoritative state, evidence pre-flight, queue reconciliation | AC-1, AC-5 | analyze | task6, task7, task8, task9 |

## Claude-Codex Deliberation

Two Codex passes ran: a first-pass analysis before Claude plan synthesis, and an iterative convergence review (2 rounds; round 2 reported "plan has converged from my side" with zero disagreements and zero required changes).

### Agreements

- Width-bucketed DS selector graphs are the right primary lever; transport (reduce bytes + casts) is the dominant DS-specific residual; prefix-window bucketing needs no compact→logical inverse mapping.
- The re-derived projection (~367–403k) is plausible as a measured outcome, not a guaranteed landing; ≤420k hard / ≤395k stretch is the right bar structure.
- Promoting the selcap diff to a hard gate and adding an op-point-shaped gate are necessary; the existing bs-1 digest alone does not exercise the op-point replay path.
- The recall oracle (eager-only) cannot see graph-only changes — graph-path exactness rests on the selection-capture gates; recall remains the algorithm-level lossiness check.
- DSA-default protection requires executable invariants plus same-round fresh Case-2 regression, with the frozen floor never replaced.
- The M1 two-patch split (keying first, compact second) is the right isolation boundary; metadata lifetime must adopt the full graph key or same-bs multi-width captures overwrite state.

### Resolved Disagreements

- Selector width vs attention metadata width (Codex round-1, high-impact): Claude's v1 did not separate them; resolved by a hard compaction-scope rule — only DS selector scratch/scoring/top-k buffers compact; DSA attention metadata and page-table mapping stay full-context.
- "Op-point gate" realism (Codex round-1, high-impact): v1 under-specified it; resolved by making the bs-29 concurrent-decode harness an explicit task (task1) with its own acceptance check (proves raw_bs, padded bs, width bucket, replay path).
- bf16 ordering claim (Codex round-1, high-impact): v1's "bf16→fp32 ordering is isomorphic" was too broad; resolved by the two-regime change classification — bf16-authoritative top-k is exact only if top-k sees exactly the same reduced bf16 values as the copy-back path and the selcap diff is zero.
- One-shot AR assumption (Codex first-pass): the draft/M5 projection assumed one-shot at the 8k bucket; refuted in code (160 KB threshold on 8 ranks) — projections re-derived for two-shot, transport bar relaxed from the draft's ≤40k to ≤60k hard / ≤45k stretch, and threshold-induced algorithm flips classified value-affecting unless pinned.
- "Case 2 frozen, never rerun" vs AC-4 re-validation (Codex first-pass): resolved — fresh Case-2 regression runs land in loop10 dirs and compare against the frozen floor; the floor number itself is never replaced.
- Per-bucket gains for top-k/logical-score (Codex first-pass): kernels already bound work by per-row seq_lens, so the draft's M5-era expectations were re-rated down; the plan's bars reflect grid/launch/scratch-shrink-only gains.

### Convergence Status

- Final Status: `converged` (round 2 of maximum 3: zero DISAGREE, zero REQUIRED_CHANGES; round-2 optional improvements — per-bucket algorithm/dtype logging, per-patch AC-4 regression on independent landing, "owns = stable keyed lifetime" clarification — are incorporated above).

## Pending User Decisions

All decisions were resolved in the gen-plan discussion. None remain PENDING.

- DEC-1: AC-1 numeric bar structure
  - Claude Position: ≤420k hard minimum; ≤395k stretch (promoted to hard only if transport lands ≤45k); per-bucket transport ≤60k/≤45k, logical-score ≤20k/≤15k, top-k ≤28k/≤24k.
  - Codex Position: same (first pass: "strong ≤395k only after measured bucket evidence"; convergence round confirmed).
  - Tradeoff Summary: the draft's ≤40k transport bar assumed one-shot AR, ruled out by the 160 KB threshold at op-point compact sizes; hard-committing ≤395k risks a late re-scope.
  - Decision Status: ≤420k HARD, ≤395k stretch, per-bucket bars as listed (user, this discussion).
- DEC-2: Bucket ladder shape and bs coverage
  - Claude Position: W=5120 + full-width fallback, compact variants for op-point bs buckets only; add 8192 later on evidence.
  - Codex Position: same (prefer 5120 + full for first measurement).
  - Tradeoff Summary: op-point-only coverage is cheapest; whole-ladder coverage is production-general from day one but multiplies whole-model graph captures/boot time and makes the small-bs AR threshold-flip hazard real (pinning rule added in response).
  - Decision Status: WHOLE bs-LADDER coverage with one compact bucket (W=5120) + guaranteed full-width fallback; capture-memory/boot-time budget measured in task3/task5; AR pinned two-shot for all compact variants in the exact regime (user, this discussion — user chose broader coverage than both reviewers' recommendation; recommendation recorded for context).
- DEC-3: Idea-2 (top-k redesign) delivery form
  - Claude Position: conditional defer; Triton-first; AOT wheel gated behind the AC-4 DSA regression.
  - Codex Position: same (defer AOT until after compact-width profiling).
  - Tradeoff Summary: compact rows may already put top-k under its bar; AOT has the lowest floor but carries wheel-install risk.
  - Decision Status: conditional defer, Triton-first, AOT wheel gated (user, this discussion).
- DEC-4: DS-off regression depth for AC-4
  - Claude Position (v1): byte-identity "preferred" language from the draft.
  - Codex Position: literal byte identity is not provable after editing shared runner code; use executable keying/runtime invariants + DS-off smoke + fresh same-round Case-2 regression.
  - Tradeoff Summary: invariants are testable and enforceable; byte-identity is a claim.
  - Decision Status: resolved by convergence (Claude adopted Codex's position; reflected in AC-4.1/AC-4.2).
- DEC-5: Loop budget
  - Claude Position: full menu — M1 primary, M2/M4 cheap once compact buffers exist, M3 conditional.
  - Codex Position: no objection; conditional structure endorsed.
  - Tradeoff Summary: Idea-1-only is cleanest but likely leaves ~15–25k of transport/cast residual unclaimed against the hard bar.
  - Decision Status: full menu (user, this discussion).
- DEC-6: Case-1 specialization vs production generality
  - Claude Position: specialize the capture LADDER to the op point; keep the MECHANISM general and always-correct via full-width fallback.
  - Codex Position: same.
  - Tradeoff Summary: superseded by DEC-2's whole-ladder choice — both mechanism AND capture coverage are now production-general; correctness for arbitrary contexts is guaranteed by the full-width fallback either way.
  - Decision Status: resolved (subsumed by DEC-2; user's whole-ladder choice makes the loop production-general).

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", or similar workflow markers
- These terms are for plan documentation only, not for the resulting codebase
- Use descriptive, domain-appropriate naming in code instead

### Loop-Specific Notes
- Hardware/op-point: single node 8×H200, TP=8, FP8 e4m3, page 64, fp8 KV, custom-all-reduce ON; one TP=8 server at a time; never set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments` for serving.
- Subagent discipline (per draft): use subagents liberally for reconnaissance, well-scoped implementation slices, and trace/CSV digestion — but every subagent product is reviewed in full by the main agent (diffs read, claims checked against artifacts) and lands only through the same verification gates. Subagents save context, not review.
- `git push` at every round boundary so cluster pre-emptions cannot destroy round state.
- This is reproduction/education work on a two-year-old paper (Double Sparsity, arXiv:2408.07092, PDF under `development/past_implementations/`) against a trained sparse indexer it is not expected to beat; the value is the performance-engineering discipline, and the gates exist to keep that discipline honest.

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

# Loop 10 Draft — Close the remaining DS-on decode gap (the dead-width tax)

> Written 2026-06-11, after **Loop 9 closed** (all ACs met: Case-1 decode 632,239 → **480,989 µs**,
> 1.84× → **1.403×** vs the frozen DSA floor; ring reduce eliminated, deterministic radix top-k
> shipped, persistent-worker logical-score landed; every change recall-gated and cross-rank
> bit-identical). This loop continues the **same goal with the same scoring method**: drive the
> frozen Case-1 one-batch decode number further toward the frozen Case-2 DSA floor (342,857 µs).
> Feed this through `gen-plan` once scope is confirmed.

---

## What this work is (and is NOT) — read first

**This is not frontier LLM development. There is no novelty here.** We are reproducing the results
of a **2-year-old paper** (Double Sparsity, arXiv:2408.07092, `development/past_implementations/double_sparsity_paper_2408.07092.pdf`)
on top of a **fully open-source codebase** (SGLang), and checking whether it can be made to perform
on **MLA models that are well behind the frontier** (GLM-5.1). The contribution, if
any, is engineering: making an existing sparse-attention idea cheap enough to be worth its opt-in
slot on a model that already ships a *trained* sparse indexer (DSA). This is purely educational and the algorithm theoretically has no hope of beating the highly optimized DSA algorithmn.theres a reason why the algorithm was introduced 2 years ago and nobody has actually decided to use it in a frontier model, we just want to see how it would perform and learn about basic performance engineering skills when doing so.

## Objective

Close the remaining **138,132 µs gap** (480,989 vs the 342,857 µs floor, 1.403×) on the **frozen
Case-1 benchmark** — GLM-5.1-FP8, 8×H200 TP=8, mem 0.7, bs 29, ISL 4096 / OSL 512, decode
GPU-kernel µs per 10-step window at torch TP-0, one trial per run. Scoring method identical to
Loop 9.

**Scope (per user): any scale of refactoring — from complete rewrites to small code changes — is
welcome**, provided it:

1. **Keeps the core DS concept** (offline channel mask → per-token signatures → query·signature
   scoring → top-k selection → sparse MLA decode).
2. **Introduces no extra lossiness in quality.** Same bar as Loop 9: recall@2048 within ±0.5pp
   (fail-closed) per landed change, cross-rank selection bit-identity HARD; and any change that is
   *supposed* to be exact (layout/transport/shape refactors) must prove **bit-identical logical
   selection indices** vs the pre-change state, not just recall parity.

Not in scope: SLO re-validation, recall R&D, the bs-64 re-tuned op point (loop-9 follow-on 4 —
separate characterization loop), multi-node.

---

## The gap we are closing (current state, from `development/loop9/results.md`)

| Reference | µs / 10-step window | ratio | status |
|---|---|---|---|
| Loop-9 frozen baseline (20260609) | 632,239 | 1.84× | historical |
| **Loop-9 final landed = THIS LOOP'S BASELINE** (`loop9/runs/20260611_r1/`) | **480,989** | **1.403×** | starting point |
| Case-2 DSA floor (frozen, never re-run) | 342,857 | 1.0× | target |

Per-bucket residuals at R1 (torch TP-0), i.e. what is left to attack:

| Bucket | µs / window | note |
|---|---|---|
| `all_reduce_two_shot_kernel<bf16,8u>` (DS score reduce) | **93,480** (+ ~15–18k bf16 cast/copy-back) | **the dominant residual** |
| new radix top-k suite (5 Triton launches/layer/step) | ≈36,300 | loop-9 gate was ≤80k; measured floor ~17.7 µs/call |
| `_logical_score_kernel` (persistent-worker grid) | 36,908 | loop-9 gate was ≤40k |
| shared non-DS topk/sort | 20,524 | present identically in Case 2 — NOT DS-attributable |

DSA spends ~17.2k µs on its entire fused indexer. DS currently spends ~167k (+casts). The
structural cause of nearly all of it is **the dead-width tax**: under CUDA graphs every DS selector
tensor is shaped by the static `req_to_token` width — `[bs, 202752]` — while the served op point
has only **≤4,608 live tokens** per row. The score reduce moves ~11.76 MB bf16 per call when live
data is ~0.27 MB: **~98% of reduced bytes are dead.**

Fresh evidence (post-loop-9 nsys capture, `loop9/runs/20260611_nsys/nsys_vs_baseline.md`): the
bf16 two-shot reduce is now the **single largest DS kernel on the timeline** — 13.3% of all
GPU-kernel time, 106 µs/call average *including cross-rank arrival skew absorbed as in-kernel
waiting* (so shrinking bytes also shrinks the skew-absorption window). Loop-9's kernel wins are
independently confirmed (top-k stack −76%, logical-score −41%, control buckets bit-for-bit clean).

**Why we believe the gap can close further:** the loop-9 M5 wildcard analysis
(`loop9/reviews/task15_m5_wildcard_proposal.md`, Codex analyze artifact) ranked the candidates and
projected that width-bucketed selector graphs with compact score buffers land the total at
**~1.10–1.15× the floor** (projection made against the M2-era 512.7k baseline; re-derive against
480,989 at plan time — R1 already banked part of its logical-score line). The last ~0.10–0.15× is
DS's intrinsic TP=8 obligation (heads are sharded → a per-layer cross-rank score exchange DSA
never pays) — full parity is NOT the bar.

---

## Candidate ideas (a menu — `gen-plan` picks/sequences; each is one implement→measure cycle)

1. **PRIMARY — width-bucketed DS selector graphs + compact per-bucket score buffers**
   (M5 proposal rank 1, "B + C-lite"; its disposition was needs-user-decision — this loop IS that
   decision). Capture the DS selection stages over a small ladder of *live-width* buckets (e.g. an
   8k bucket covers the served ≤4,608-token window) so the expensive selector shapes become
   `[bs, 8192]` instead of `[bs, 202752]`. Expected from the proposal: reduce ~110k → **~31–35k**
   (0.46 MB at the 8k bucket goes one-shot custom-AR, ~40 µs/call floor), logical-score
   36.9k → **~10–15k**, top-k 36.3k → **~16–24k**. Known risks (from the proposal — design for
   them, don't discover them): (a) the runner is bs-bucketed today; width-bucketing is a **real
   cuda-graph-runner integration change**; (b) bucket dispatch must come from **host-visible
   scheduler metadata**, not a device-computed max (no sync before replay); (c) dead logical
   positions must stay equivalent to −inf, and selected outputs must preserve **logical** positions
   exactly — the compact→logical inverse mapping becomes part of the correctness contract; (d) more
   captured variants = more capture memory and replay bookkeeping (the loop-9 M4 audit freed
   16.8 GB of over-captured ladder, so headroom exists — budget it explicitly); (e) custom-AR is
   out-of-place — the bucketed reduce needs a captured copy-back or consumers reading the compact
   buffer directly; (f) overflow beyond the largest bucket takes a graph-safe full-width fallback
   that is bit-identical. *This is an exact layout/transport change: gate on bit-identical logical
   indices, not just recall. Scope: large refactor (runner + DS cuda_graph.py + selection
   pipeline).*

2. **Fused multi-block top-k redesign** (loop-9 follow-on 2). The shipped Triton radix suite is 5
   launches/layer/step; the loop-9 AOT op proved a true single launch at 43.2 µs op-point but lost
   at long contexts (one block per row). Redesign with several blocks per row + cross-block
   coordination targeting the measured **17.7 µs/call floor across ALL widths** — ideally expressed
   directly on the compact bucketed buffers from Idea 1 (smaller rows make single-launch easier).
   The AOT build pipeline, env-gated tests, and non-finite contract fixtures all exist from loop 9.
   If a rebuilt wheel is to be *installed*, that is gated: mandatory DSA regression (DS-off smoke +
   Case-2 re-validation) per loop-9 follow-on 3 — or stay Triton-only and skip the wheel question.
   *The deterministic tie-break contract (score desc, pos asc; −inf/NaN never selected, +inf
   maximal) must be preserved bit-exactly.*

3. **Reduce-path residuals after bucketing**: pick the cheapest correct transport per bucket
   (one-shot vs two-shot custom-AR by size), eliminate the separate fp32↔bf16 cast kernels by
   fusing the cast into producer/consumer (the ~15–18k cast tax), and re-measure skew absorption
   (nsys timeline) once bytes shrink. *Lossless transport changes; selection bit-identity gates
   each.*

4. **Logical-score on compact width**: the persistent-worker grid already loops live blocks only;
   on compact buffers the dead-grid floor disappears entirely (M5 projection ~10–15k). Likely
   falls out of Idea 1 nearly for free — measure, don't assume.

5. **WILDCARD (only if 1–4 plateau) — fused score+select with exact two-round reduce** (M5 rank 4,
   explicitly low-confidence): exact global top-k over TP-summed scores from bounded local
   candidate unions is NOT guaranteed; an exactness round can fix it but worst-case expands to
   full width. Touch only with an exactness proof; anything approximate fails the
   no-added-lossiness bar and the cross-rank bit-identity hard gate.

Ideas found while reading the code that beat these replace them — the menu is a starting point,
not a contract. Complete rewrites of the selection pipeline are acceptable per the scope
statement, under the same gates.

---

## Open scope + the task queue (`development/loop10/queue.md`)

**The scope of this loop is deliberately NOT fixed to the menu above.** The agent is expected —
and incentivized — to invent additional optimization ideas while working (reading code, reading
profiles, watching where the bottleneck moves after each landed change) and to pursue them **in
the same loop**, as long as each new idea is compatible with the optimizations already completed
and accepted.

`queue.md` (already created, deliberately empty) is the loop's **self-contained task queue and
checklist** — the single source of truth for what is planned, in flight, done, or dropped:

- **Populating the queue is the FIRST task of the loop**, once plan refinement has completed and
  the loop kicks off — seed it from the final plan's tasks plus any further ideas generated at
  kickoff. Do NOT populate it during plan generation.
- Every task gets a queue entry: id, description, targeted bucket, expected effect, lossiness
  posture, compatibility note vs already-landed changes, status. New ideas discovered mid-loop are
  **appended as queued entries** (with that one-line compatibility check) rather than expanding
  the current task's scope.
- A task is marked completed only after its gates pass (losslessness teeth + profile). Dropped or
  superseded tasks stay listed with the measured/reasoned cause — no silent deletions.
- The queue is committed with each round so reviews see the same checklist the agent works from.

## Subagent usage (context discipline)

Use subagents **liberally** to keep the main context from bloating across a long multi-task loop:
Explore subagents for code/call-chain reconnaissance, implementation subagents for well-scoped
task slices, analysis subagents for trace/CSV digestion. Two hard rules:

1. **Every subagent's work is carefully reviewed by the main agent before it is trusted** — diffs
   read in full, claims checked against the actual artifact, never relayed unverified.
2. Nothing a subagent produced lands without passing the **same verification gates**
   (losslessness teeth + profiling) as main-agent work. Subagents save context, not review.

---

## The iterate→measure protocol (unchanged heartbeat, loop-9 tooling reused)

For **each** queue task, exactly one cycle:

1. **Implement** (DS path; shared surfaces touched → see AC-4, stricter than loop 9).
2. **Verify losslessness FIRST** — three teeth, all existing loop-9 tools:
   a. **Selection-capture bit-identity** (`loop9/selection_capture_tool.py` run/verify/diff, graph
      mode, 8 ranks): logical selected indices identical to the pre-change served state for exact
      changes. NOTE for Idea 1: capture mirrors must record **logical** indices after the
      compact→logical mapping; the diff target is the loop-9 R1 digest
      (`loop9/runs/20260611_r1/selcap_digest.json`).
   b. **Cross-rank bit-identity** (hard, every gate run).
   c. **NIAH recall@2048** (`loop7/niah_oracle_sweep.py` + `loop9/oracle_recall_summary.py`,
      fail-closed ±0.5pp vs the frozen `loop9/runs/20260610_m0/recall_baseline.json`).
3. **Profile after EVERY task implementation** (not just at round gates): re-profile Case 1 with
   the frozen recipe verbatim via
   `development/profiling/runs/20260609/run_case.sh <out> case1 torch 29` — torch profiler is the
   default; add an nsys capture when the question is timeline-shaped (skew, overlap, launch gaps).
   The purpose is twofold: confirm the optimization did its job in its targeted bucket, and see
   **whether the bottleneck has shifted**. **One trial per run**; Cases 2/3 and all loop-9 run
   dirs are frozen references, never re-run. (nsys note: NVTX ranges do NOT fire under graph
   replay — attribute by kernel name; see `loop9/runs/20260611_nsys/nsys_vs_baseline.md`.)
4. **Read the gap** per-bucket vs the R1 column and the Case-2 floor (`summarize_torch.py`,
   `compare_decode.py`). Per-bucket attribution primary, totals secondary (boot-to-boot
   shared-kernel variance up to ~27k µs — established in loop 9, DEC-1). A shifted bottleneck is
   a queue-feeding event: append the newly exposed target to `queue.md` as a candidate task.
5. **Keep or revert.** Bank only what passes step 2 and shrinks its targeted bucket. Carry the
   running Case-1 number forward. Ledger: `development/loop10/results.md` (rewrite-over-append
   when state changes; one authoritative current-state section — loop-9 lesson).

Measurement discipline carried from loop 9 (BitLesson-backed): eager microbenches measure host JIT
dispatch — the **CUDA-graph captured replay number is the binding one**; run seq≈0 floor probes
before believing a hypothesis; build the loser before issuing a comparison verdict.

---

## Acceptance criteria (draft — `gen-plan` formalizes the numbers)

1. **Gap closed and attributed.** Case-1 decode GPU-kernel time materially below 480,989 µs and
   measurably closer to 342,857 µs, attributed per-bucket. Suggested bars for discussion
   (`gen-plan` sets final): minimum **≤420k** (~1.23×); strong **≤395k** (~1.15×, the M5
   projection band). Per-bucket suggestions: score-reduce bucket (named AR kernel + casts)
   **≤40k**; logical-score **≤15k**; DS top-k **≤25k**.
2. **No extra lossiness** (the user's hard bar): recall@2048 Δ ≤0.5pp fail-closed per landed
   change; cross-rank bit-identity HARD; exact-by-design changes (layout/transport/bucketing)
   additionally prove bit-identical logical selection vs the pre-change state.
3. **DS concept intact** — no dense fallback, no DSA-indexer substitution.
4. **DSA-native default un-regressed — STRICTER THIS LOOP.** Idea 1 touches the shared CUDA-graph
   runner, not just DS files. Any change to shared capture/replay code triggers the mandatory DSA
   regression (DS-off smoke + frozen Case-2 recipe re-validation) **in the same round**, plus
   DS-off byte-identity where feasible. The shipped DSA default must stay untouched in behavior
   and performance.
5. **Protocol/ledger discipline**: one trial per run; Case 1 only re-profiled; frozen references
   reused; deviations logged in the goal tracker's Plan Evolution Log; evidence pre-flight before
   each round handoff (artifact exists + is tracked + claim matches artifact — loop-9 methodology
   lesson; watch the repo's `*.log` gitignore, `git add -f` cited evidence). `queue.md` kept
   current every round: statuses accurate, new ideas appended with compatibility notes, drops
   recorded with their cause; queue population itself is the loop's first task.

## Files to read first

- **Current state + follow-on definitions:** `development/loop9/results.md` (esp. "Structural
  headroom" items 1–3), `development/loop9/reviews/task15_m5_wildcard_proposal.md` (the design
  being executed), `development/loop9/runs/20260611_nsys/nsys_vs_baseline.md` (fresh timeline
  evidence).
- **The selection pipeline as loop 9 left it:**
  `python/sglang/srt/layers/attention/double_sparsity/selection_kernel.py` (`reduce_token_scores`,
  `_logical_score_kernel`, `retrieve_topk_graph_safe`), `topk_kernel.py` (radix suite),
  `cuda_graph.py` (DSGraphState, scratch bundles, capture parity), `selector.py`, `config.py`.
- **The graph-runner integration surface for width-bucketing:**
  `python/sglang/srt/model_executor/cuda_graph_runner.py` (bs-bucket ladder, capture/replay
  dispatch), the `allocate_graph_state` call sites in
  `python/sglang/srt/layers/attention/dsa_backend.py`, and the bind site in
  `python/sglang/srt/models/deepseek_v2.py` (`_select_topk_indices`).
- **AOT op (Idea 2):** `sgl-kernel/csrc/elementwise/ds_topk.cu`,
  `sgl-kernel/python/sgl_kernel/top_k.py`, build recipe in
  BL-20260611-sgl-kernel-build-cuda13-cccl (`.humanize/bitlesson.md`) +
  `loop9/runs/20260611_r1/sgl_kernel_build.log`. **Never force-install a rebuilt wheel over the
  frozen-reference binary** without the AC-4 gate.
- **Gate runners to clone:** `loop9/run_r1_gates.sh` (selcap → recall → torch re-profile pattern),
  `loop9/selection_capture_tool.py`, `loop9/oracle_recall_summary.py`.
- **Doctrine:** `CLAUDE.md`; `.pensieve/` maxims.

## Hardware / op-point

Single node 8×H200, TP=8, FP8 e4m3, page 64, fp8 KV, custom-all-reduce ON. **Never set
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments` for serving.** One TP=8 server at a time. The frozen
Case-1 recipe (server args, DS_CONFIG, bench command) is `development/profiling/plan.md` +
`runs/20260609/_env.sh` — scoring method unchanged from loop 9.

## Pending decisions (resolve in `gen-plan` discussion)

- **Numeric bar for AC-1** — accept the suggested ≤420k min / ≤395k strong, or re-derive from the
  M5 projection against the 480,989 baseline? (The proposal's ranges predate R1's logical-score
  landing; the strong bar should not double-count that win.)
- **Bucket ladder** — single 8k live-width bucket + full-width fallback, or a small ladder
  (e.g. 8k/32k/full)? Dispatch from which host-visible metadata (max seq_len in batch at schedule
  time)? Capture-memory budget (width buckets × bs buckets)?
- **Idea-2 delivery form** — Triton multi-block redesign (no wheel question) vs AOT CUDA (faster
  floor, but wheel install gated on the AC-4 DSA regression)? Or defer Idea 2 entirely if Idea 1's
  compact rows already put top-k under the bar?
- **DS-off regression depth for AC-4** — is DS-off byte-identity provable (preferred), or is the
  frozen Case-2 recipe re-validation + smoke the practical gate when the shared runner changes?
- **Loop budget** — single-idea loop (Idea 1 only, done right) vs the full menu? Idea 1 alone is
  projected to clear the strong bar; Ideas 2–4 are insurance/cleanup.

--- Original Design Draft End ---
