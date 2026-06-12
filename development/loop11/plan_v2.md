# Loop 11 Implementation Plan — Lift the DS Batch Cap: Eliminate the TokenLabelTable Tax, Fix the TTFT Tail, Serve Radix-On

## Goal Description

Close the remaining **serving-level** DS-vs-DSA gap at concurrency 16–64 on the canonical client workload (GLM-5.1-FP8, 8×H200 TP=8, gsp 4096-ISL / 512-OSL / ~55% prefix hit, per `development/SLOS.md` — note: the draft's `CLIENT_SLOS.md` reference is stale; `development/SLOS.md` is the document that exists).

Loop 10 closed the per-step kernel gap (Case-1 one-batch decode 480,989 → 361,824 µs; 1.055× vs the frozen DSA floor), and the 2026-06-12 serving re-measurement (`development/profiling/runs/20260612/`, `development/profiling/results.md`) proved the remaining gap is serving behavior, not kernel time:

- At matched bs30, DS and DSA decode-TPS are within ~4% — the selector is no longer the problem.
- The TTFT tail is the batch cap, not DS: DSA capped to bs30 has a *worse* p99 tail than DS. Whoever runs batch 30 at conc ≥32 queues.
- DSA's remaining advantage is purely the batch it can admit: at conc 64 it wins aggregate (676 vs 577 tok/s) and tail (28.1 vs 45.2 s p99) because it admits 64 while DS admits ~30.

The causal chain (measured): the TokenLabelTable (5.29 GB/rank fp16) + DS capture state are allocated from runtime headroom **after** the static pool → DS cannot raise `mem_fraction_static` above 0.7 (DSA serves at 0.8) → 142k-token pool → bs30 cap → admission queueing at conc ≥32 → p99 TTFT tail + aggregate loss. **Memory is the root; everything else is symptom.** Codebase exploration during plan generation added a second, draft-unknown lever: under DS, the pool still allocates the **DSA indexer index-k sidecar** (~10.3 KB/token across 78 layers ≈ 17% of the measured 61.5 KB/token pool cost) that DS never reads.

The loop therefore:

1. **Lifts the DS decode-batch cap (bs30 → ≥64)** by removing or radically shrinking the per-rank memory overheads: the TokenLabelTable (primary: absorbed-latent scoring derives signatures from the resident fp8 KV latent, making the table, the prefill write hook, and all 5.29 GB disappear exactly) and the unused DSA indexer sidecar (exact gating + corrected pool accounting), with int8 signatures + table-aware pool sizing as the owner-approved stepping stone.
2. **Fixes the tail TTFT** (admission queueing at conc 32/64) that the cap causes.
3. **Enables and validates radix-on serving for DS** — fixture gate per served config plus end-to-end correctness — so the comparison vs the production radix-ON DSA default is honest and DS benefits from the same ~55% prefix reuse.

While: keeping the per-step decode tax bounded (the loop-10 win must not regress) and keeping the DS concept intact (offline channel mask → per-token signatures, materialized **or implicitly derived** → query·signature scoring → top-k → sparse MLA decode; no dense fallback, no DSA-indexer substitution).

**Nature of the work (from the draft, binding):** this is not frontier LLM development — it reproduces a 2-year-old paper (Double Sparsity, arXiv:2408.07092) on an open-source codebase against a model that ships a trained sparse indexer (DSA). The contribution is engineering and education: see how far the idea can go. **Scope (owner): any scale of refactoring — complete rewrites to small changes — is welcome** under the concept and quality constraints above.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification. Hard/trend status per DEC-1 (resolved): **bs cap ≥64, TTFT ≤1.10×, aggregate ≥0.95×, and tax ≤1.10× are HARD, judged at the closing locked AC-11 sweep; ≥390k tokens @ mem 0.8 is a TREND target.** The directional ladder is an iteration signal only, never an AC verdict venue.

- AC-1: Capacity — the root cause.
  - AC-1.1 (HARD): derived decode-batch cap ≥ 64 at the 4608-token workload shape (any landed lever may produce it; the binding requirement is the cap, not a specific memory fraction).
    - Positive Tests (expected to PASS):
      - Boot at the served config: `max_total_num_tokens` readout yields floor(capacity/4608) ≥ 64.
      - CUDA graph capture completes for all width buckets at that config (capacity without capture success does not count).
      - Conc-64 serve smoke: achieved concurrency ≈ nominal (DS no longer admission-capped below nominal at conc ≤ 64).
    - Negative Tests (expected to FAIL):
      - Serving the unchanged fp16-table config at mem 0.8 fails to boot or OOMs (demonstrates the landed lever is causal, not ambient slack).
      - A capacity claim based on boot readout alone, without graph-capture success and the serve smoke, is rejected as AC-1.1 evidence.
  - AC-1.2 (TREND): DS serves at the DSA memory op-point — mem_fraction_static 0.8 with KV capacity ≥ ~390k tokens (within ~5% of DSA's at the same fraction). Stretch: TokenLabelTable bytes = 0 (absorbed-latent landed and table deleted per DEC-2).
    - Positive: boot at 0.8 with capacity readout ≥ ~390k tokens; task0's componentized memory table shows where every freed GB went.
    - Negative: a token-capacity comparison that ignores the per-token cell-size change from indexer-sidecar gating (i.e. counts tokens against the wrong bytes/token) is rejected.
- AC-2: Tail TTFT at the SLO range — the symptom (HARD, AC-11 venue).
  - Against the frozen radix-ON DSA @0.8 baseline (task1): DS p99 TTFT ≤ 1.10× DSA per concurrency at 16/32/64, and DS meets the absolute bar (p99 < 22 s) wherever DSA does.
  - Positive Tests:
    - The closing locked AC-11 sweep (3 trials × 600s, `benchmark_compare.py --ac11`) passes its TTFT gate per concurrency (the comparator's own constant is the same 1.10×).
    - Achieved ≈ nominal concurrency at conc ≤ 64 in the sweep.
  - Negative Tests:
    - A 1-trial directional ladder result presented as an AC-2 verdict is rejected (venue rule).
    - Any concurrency where DSA meets p99 < 22 s and DS does not fails the criterion.
- AC-3: Throughput (HARD, AC-11 venue).
  - DS per-request decode-TPS p50 ≥ 30 maintained at conc 16/32/64 (already met — must not regress), and DS aggregate ≥ 0.95× DSA radix-ON at conc 64.
  - Positive Tests: AC-11 sweep TPS gate passes (comparator constant is the same 0.95×); per-request p50 ≥ 30 at all three concurrencies.
  - Negative Tests: any concurrency with decode-TPS p50 < 30 fails; runs flagged by the comparator's no-op detection (`selected_tokens == total_tokens` or `dense_fallback_total != 0`) are rejected as evidence.
- AC-4: Per-step tax guard (HARD).
  - DS-vs-DSA same-batch one-batch decode window ratio ≤ ~1.10 at the new common batch (bs64, both mem 0.8), and the bs30 window stays ≤ ~380k µs (loop-10 close: 361,824 µs) — the loop-10 win is not traded away for capacity.
  - Positive Tests: 10-step torch decode window per the `runs/20260612` recipe at bs64 shows ratio ≤ ~1.10; bs30 re-measurement stays ≤ ~380k µs.
  - Negative Tests: eager-mode microbenchmarks submitted in place of graph-mode windows are rejected (captured-replay numbers are binding); ratio > 1.10 at bs64 fails.
- AC-5: Quality — owner-decided bar, fixed; do not relitigate.
  - Recall@2048 within ±0.5pp fail-closed vs the frozen `development/loop9/runs/20260610_m0/recall_baseline.json`, per landed change; cross-rank selection bit-identity HARD on every gate run; exact-by-design changes additionally prove bit-identical selection via the selection-capture tool vs the pre-change digest; value-affecting changes (fp8-latent-sourced signatures, int8 tables) are admissible and recorded as declared value-affecting decisions with gate evidence.
  - Radix-on correctness:
    - Fixtures pass on the GLM-5.1 config **per served signature config** — the fixture artifact fingerprints `signature_dtype`, so fp16, int8, and table-free modes each require their own artifact; artifacts are re-run/finalized after the final served selector mode is chosen (task7 is final-mode gated).
    - Table-free mode uses a NEW fixture kind + artifact schema (latent/selection cold-warm equivalence across a cache hit) with its own fail-closed validator path — the current label-capture requirement is **replaced, not waived**, and mismatched-config artifacts still refuse radix-on.
    - Cold-vs-warm selection equivalence demonstrated on the served workload (selection-capture diff across a cache hit); recall@2048 under radix-on within the same ±0.5pp bar; eviction/partial-hit edge probe clean (page-boundary hits at page 64).
  - Positive Tests: all gate artifacts exist, are tracked, and match their claims at round handoff (evidence pre-flight).
  - Negative Tests: recall delta > 0.5pp fails closed; any cross-rank selection divergence fails; radix-on serving without a matching per-config artifact is refused by the validator; a fixture pass from a different `signature_dtype` does not authorize the served config.
- AC-6: DS concept intact.
  - Offline mask → signatures (materialized or absorbed) → query·signature scoring → top-k → sparse MLA decode. No dense fallback; no DSA-indexer substitution for selection. Gating the *allocation* of the unused DSA indexer sidecar in DS mode does not violate this — DS never reads it.
  - Positive: pipeline review of the landed code confirms each stage present and live.
  - Negative: any code path where selection falls back to dense attention or reads DSA indexer outputs fails.
- AC-7: DSA-native default un-regressed — strict.
  - Changes to shared surfaces (memory accounting/pool sizing/cell-size math, indexer-cache allocation gating, radix plumbing, graph runner) trigger the mandatory DSA regression **in the same round**: DS-off smoke + frozen Case-2 recipe re-validation + radix-ON DSA serving smoke. The shipped DSA default stays untouched in behavior and performance.
  - Positive: same-round regression evidence accompanies every shared-surface change.
  - Negative: a shared-surface change landing without the same-round DSA regression fails the round; any measured DSA behavior/performance delta fails.
- AC-8: Protocol/ledger/queue discipline.
  - `queue.md` current every round (statuses, appended ideas with one-line compatibility notes, recorded drops — no silent deletions); evidence pre-flight before each round handoff (artifact exists + tracked + claim matches artifact; pre-flight sentences describe the POST-commit state); `development/loop11/results.md` rewritten-over-appended with one authoritative current-state section; 1-trial honesty until the AC-11 sweep (no lower-bound/SLO-pass claims beyond what trial count supports); frozen references never re-run; `git push` at every round boundary.
  - Positive: each round's commit shows the updated queue and pre-flight-consistent ledger.
  - Negative: a claim whose artifact is missing/untracked/mismatched fails pre-flight; re-running a frozen reference is a protocol violation.

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.

### Upper Bound (Maximum Acceptable Scope)
Absorbed-latent scoring fully replaces the TokenLabelTable — `token_label_table.py` and `token_label_write.py` deleted along with the table ABI (DEC-2) — with the DS-mode indexer sidecar gated off under corrected pool accounting, table-aware memory accounting landed, radix-on validated end-to-end including the new table-free fixture kind, the bs64 capture ladder and same-batch tax window re-validated, and the locked AC-11 sweep passing at mem 0.8 against the frozen radix-ON DSA baseline. Mid-loop ideas that beat planned tasks replace them (queue-mediated, same gates).

### Lower Bound (Minimum Acceptable Scope)
M0 ground truth complete (componentized memory table, probe matrix, frozen radix-ON DSA baseline, DS-Offload rejection memo); M1 landed in full per DEC-3 (indexer-cache gate + int8 served config + table-aware pool sizing) with the directional ladder re-run at the lifted cap and AC-1.1 met; radix-on enablement + correctness validated for the served config; absorbed-latent carried at least through the task5 prototype gate with documented findings (live-path equivalence + recall evidence, even if integration does not land); the locked AC-11 sweep run once at close against the frozen baseline with honest characterization of any unmet bars.

### Allowed Choices
- Can use: any scale of refactor of the DS signature/scoring pipeline (owner scope: complete rewrites acceptable); new Triton/CUDA kernels; pool-sizing/cell-accounting changes; new fixture kinds and artifact schemas for table-free mode; config-validation additions; the loop-10 shared-DSGraphState capture pattern; subagents for reconnaissance/digestion/measurement/drafting (mandatory per draft).
- Cannot use: dense fallback for selection; DSA-indexer outputs as DS selection input; behavior/performance changes to the shipped DSA default; `PYTORCH_CUDA_ALLOC_CONF=expandable_segments` for serving (breaks custom-all-reduce IPC at GLM TP=8); relitigation of the owner's lossiness bar or the three fixed owner decisions; re-running frozen references.
- Fixed by decision: absorbed-latent v1 is `scorer_norm="off"` only — config validation must reject cosine/hybrid combined with the absorbed path (their token-side norms require per-token `W_UK·c_kv` norms, defeating the absorption); the scoring kernel is paged (walks `req_to_token` logical positions; output stays `[bs, max_seq_len]` fp32 so the loop-10 pinned bf16 two-shot cross-rank reduce is untouched); the indexer-cache gate is a designed pool capability (capability flag or DS pool variant, guarded accessors that fail loudly, cell-size/configurator accounting update, offload/disagg/radix state-path audit), not a skipped allocation.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

**Absorbed-latent scoring (the MLA escape hatch, primary).** Since `k_nope[h] = W_UK[h] · c_kv`:

```
score[t,h] = Σ_{c∈S_h} w_c · q_c · k_nope[t,h,c]
           = ( Σ_{c∈S_h} w_c · q_c · W_UK[h][c,:] ) · c_kv[t]   =   v_h · c_kv[t]
```

with `v_h ∈ R^512` computed once per step per head (32×512 MACs — trivial). The signature IS the latent; the table, the prefill label-write hook, and 5.29 GB/rank disappear exactly, while the offline channel mask (`S_h`, `w_c`) keeps doing its job on the query side.

- **v_h build contract:** per-rank W_UK rows live in `kv_b_proj` as block-quantized fp8 with `weight_scale_inv`, dequantized inline during attention forward. Build `v_h` from a one-time bind-time dequantization of the needed W_UK slices to fp32 (matching attention's inline-dequant semantics), a few MB per rank; score with fp32 accumulation (matches current `_logical_score` semantics) — this resolves the draft's "absorbed projection precision" pending item.
- **Kernel shape:** paged, like today's logical-score kernel — walks per-request logical positions through `req_to_token`; reads the cached MLA fp8 nope latent (512 B/token) plus its per-128-channel-block fp32 scales (16 B/token) ≈ **528 B/token** — close to, not exactly, the fp16-label read (512 B/token) — and dequantizes in-kernel. One read serves all 8 heads (vs per-head label rows), and the op is tensor-core-shaped. Treat "lands at-or-under the current logical-score budget (23.5k µs/window)" as a measured risk, not an assumption.
- **Kernel-cost de-risk (block-scale reassociation, part of task5's design space):** compute `score[t] = Σ_blocks scale_b(t) · (v_h[block] · latent_fp8[t, block])` — apply the 4 per-128-block fp32 scales to per-block partial dots instead of dequantizing 512 elements per token. The real-arithmetic product is identical; only fp32 rounding reassociates, which the declared value-affecting record already covers. A further variant quantizes v_h itself per-block to fp8 and runs fp8 tensor-core dots — additionally value-affecting; gate separately if tried.
- **Validity semantics:** the table's written-slot mask exists to avoid scoring stale/unwritten label slots. The latent path must establish the equivalent invariant: scored slots are exactly the slots the KV cache has written — analyze and document ordering vs `out_cache_loc` allocation in decode and radix-reuse paths (seq_lens masking) as part of the prototype task.
- **Equivalence contract (prototype gate):** compare latent-derived scores/selections against the LIVE label path on the same step with the same inputs (selection-overlap metric + oracle recall) — not only against the hand-derived formula — before any selector-ABI change or table deletion. This is a declared value-affecting change (fp8-sourced signatures; self-consistency argument: scoring now sees exactly what attention sees).
- **Table-free selector ABI (integration step):** the selector bind currently requires a non-None table, and the deepseek_v2 call site preconditions on it. Replace with a latent-scoring binding (KV-pool latent buffer references + dequantized W_UK slices + channel mask); per-step `v_h` scratch lives in the shared DS graph state; the labels-gather stage drops out of capture while the score buffer shape `[max_bs, max_seq_len]` is unchanged; rope dims stay excluded by construction (assert it).

**DS-mode indexer-cache gate (exact, cheap, large).** Under DS, the pool allocates the DSA indexer index-k sidecar that DS never reads (~10.3 KB/token across 78 layers). Gate its allocation behind a designed pool capability and update the cell-size/configurator math so freed bytes become tokens — without the accounting update the gate frees nothing. Audit clear/copy/state/offload/disagg paths that assume the buffer exists; accessors fail loudly if called. Optionally probe the (smaller) native indexer object/weight-init overhead in DS mode.

**Capacity arithmetic (verify via task0 probes; derive constants from runtime config/log fields, not hard-coded):** pool cost today ~61.5 KB/token (~51.2 KB MLA fp8 KV + ~10.3 KB indexer sidecar; matches the measured 8.14 GB / 142,208 tokens). At DSA's 0.8 budget (~23.5 GB): table-free + indexer-gated ≈ 459k tokens; int8-table-inside-budget + indexer-gated ≈ 319k (bs ~69); int8 + indexer kept ≈ 280k (bs ~60, borderline). int8 table bytes are 0.5625× fp16 (int8 signatures + per-slot/head fp16 scales).

**Table-aware pool sizing:** today KV sizing happens before the DS bind allocates the table from leftover headroom. Deduct the pre-computed table bytes from available memory before pool sizing so the pool + table fit the budget deliberately — this is what lets int8 serve at a higher fraction stably.

**Served-envelope right-sizing + bounded selector-width capture (two further EXACT headroom levers; numbers are estimates, verify via task0):** (1) right-size the served runtime envelope for the SLO workload — cap `max_running_requests` near 64 and `cuda_graph_max_bs` at 64 (no captured batch above the workload cap). Serve logs imply defaults near 2048 requests / bs512 capture; ReqToTokenPool alone drops ~1.55 GiB → ~50 MiB at context 202752, and the measured ~4.68 GB DS graph memory should fall materially. The frozen DSA baseline keeps its own production defaults — the client workload is identical, so the comparison stays honest — but the envelope is recorded as part of the served DS config. (2) Bounded DS selector-width graph mode — stop auto-capturing the full-context-width DS graph variant (today capture adds the full width alongside the workload buckets; a full-width fp32 score scratch at large capture bs is hundreds of MiB before masks and graph objects), failing closed if a live sequence exceeds the declared width cap. Both are task0 probe rows and M1 candidate levers (seeded into the queue at kickoff, promoted on task0 evidence); they compose with each other, with the indexer gate, and with table elimination.

**DS-Offload (rejected — document, don't build):** the gather is top-2048 × 78 layers × 576 B ≈ 92 MB per decode step per request; at bs30 ≈ 2.8 GB/step over ~50 GB/s PCIe ≈ 55 ms/step — alone double the 33 ms/step budget. The labels (our big tensor) are read densely every step and cannot be offloaded at all. The memo records this with one measured PCIe number. The lesson inverts on us: their densely-read tensor was small; ours is the big one.

**Also considered and rejected during planning (recorded so the loop does not rediscover them):** (a) sparse score transport (per-rank local top-k union replacing the dense cross-rank reduce) — exactness fails because the top-k of SUMMED TP scores need not appear in any rank's local top-k, and the exact candidate union is 8×2048 = 16384 > the ~4608 live width, saving nothing; (b) page-level two-stage prefilter — with 4608/64 = 72 pages and k = 2048, even a perfect prefilter keeps ≥ 32 pages (≤ 2.25× bound, looser in practice) for new metadata and write complexity; only revisit at ≥16k contexts; (c) offload re-checked at bs64 — ~5.9 GB/step ≈ ~118 ms at 50 GB/s PCIe, worse than the bs30 rejection; (d) compressing the written-slot bitmap — ~11 MB/rank, immaterial next to table and graph memory; (e) lower/adaptive top-k — changes the recall@2048 contract and the apples-to-apples DSA comparison (quality bar, not a tuning knob).

**Radix synergy:** with signatures derived from cached KV, cold-vs-warm signature equality holds largely by construction — but correctness still depends on identical cached fp8 latent bytes/scales, `req_to_token` mapping, sequence masks, and partial-prefix behavior, so the table-free fixture kind and the served-workload equivalence checks remain mandatory.

### Relevant References

- `python/sglang/srt/layers/attention/double_sparsity/token_label_table.py` — the 5.29 GB table (deleted at upper bound)
- `python/sglang/srt/layers/attention/double_sparsity/token_label_write.py` — prefill label-write hook (deleted at upper bound)
- `python/sglang/srt/layers/attention/double_sparsity/selection_kernel.py` — logical-score kernel, query projection, cross-rank score reduce, graph-safe top-k retrieval
- `python/sglang/srt/layers/attention/double_sparsity/selector.py` — runtime bind (currently requires the table)
- `python/sglang/srt/layers/attention/double_sparsity/cuda_graph.py` — DS graph state, capture parity, width buckets
- `python/sglang/srt/layers/attention/double_sparsity/config.py` — scorer modes (`scorer_norm` off/cosine/hybrid), signature dtype
- `python/sglang/srt/layers/attention/double_sparsity/validator.py` — radix fixture gate, artifact schema, refusal path
- `python/sglang/srt/layers/attention/double_sparsity/page_table_adapter.py` — logical→physical slot mapping (capture-safe)
- `python/sglang/srt/layers/attention/double_sparsity/radix_fixture_capture.py` — label capture/fingerprinting for fixtures
- `python/sglang/srt/models/deepseek_v2.py` — `kv_b_proj` (W_UK, block-fp8), selection bind site, DS finalize/bind
- `python/sglang/srt/mem_cache/memory_pool.py` — MLA fp8 pool layout (512 nope fp8 + 16 scale + 128 rope bytes/token/layer), DSA pool with indexer sidecar
- `python/sglang/srt/layers/attention/dsa/quant_k_cache.py` — fp8 quantization kernel, per-128-block scale computation
- `python/sglang/srt/model_executor/model_runner.py` and `model_runner_kv_cache_mixin.py` — mem-fraction → pool sizing → headroom (where table-aware accounting goes)
- `test/manual/test_dsv32_radix_label_capture_fixture.py`, `test_dsv32_fp8_scale_stability.py`, `test_dsv32_radix_cache_fixture.py` — the radix fixtures
- `serve_double_sparsity.sh` — served defaults (mem-fraction history, signature_dtype, radix gating env vars)
- `development/profiling/runs/20260612/` — `_env.sh` + stage drivers (the serving recipe of record), `breakdown.md` (per-kernel ground truth), `serving/SUMMARY.txt`
- `development/profiling/results.md` — the 2026-06-12 re-measurement (the loop's motivation)
- `development/benchmark.sh`, `benchmark_baseline.sh`, `development/benchmark_compare.py` — AC-11 mode (3-trial, 600s window, TPS ≥0.95× / TTFT ≤1.10× gates, no-op detection, refusal rules)
- `development/SLOS.md` — the SLO definitions (corrected reference)
- `development/loop9/selection_capture_tool.py`, `development/loop9/oracle_recall_summary.py`, `development/loop7/niah_oracle_sweep.py`, frozen `development/loop9/runs/20260610_m0/recall_baseline.json` — quality gates
- `development/loop10/reviews/task10_closeout.md`, `development/loop10/results.md` — what landed in loop 10 + decision record
- `development/past_implementations/study/` (`00-survey.md`, `08-current-system-architecture.md`), `DoubleSparse/offloading/`, `sglang-last-with-double-sparsity` — prior-art evidence for the rejection memo
- `CLAUDE.md`, `.pensieve/` maxims, `.humanize/bitlesson.md` — doctrine

## Dependencies and Sequence

### Milestones

1. M0 — Ground truth and frozen bars
   - task0: componentized per-rank memory accounting + max-fraction/capacity probe matrix (each probe = boot + capacity readout + graph-capture check + short serve smoke)
   - task1: freeze the radix-ON DSA @0.8 directional baseline ladder (one 3-concurrency run, frozen as the loop's comparison column; today's 20260612 radix-off ladder stays the radix-off reference)
   - task2: DS-Offload rejection memo (analyze; one measured PCIe number)
2. M1 — Cheap capacity levers (completes in full before M2 starts, per DEC-3)
   - task3: DS-mode indexer-cache gate (designed pool capability + accounting update; AC-7 regression same round)
   - task4: int8 served config + table-aware pool sizing at the task0-measured best fraction
   - Candidate levers (queued at kickoff, promoted on task0 evidence; not part of the M1 completion gate): served-envelope right-sizing and bounded selector-width capture — exact headroom levers that compose with task3/task4
   - Milestone gate: directional ladder re-run at the lifted cap
3. M2 — Absorbed-latent scoring
   - task5: prototype kernel (score-only diagnostic; live-path equivalence + recall gates)
   - task6: integration — table-free selector ABI, capture, **delete table + write hook (DEC-2)**, capacity payoff at mem 0.8
   - Milestone gate: directional ladder re-run
4. M3 — Radix-on for DS
   - task7: enablement (fixtures per served config; new fixture kind if table-free landed; artifacts finalized against the FINAL served selector mode + signature_dtype) + correctness validation (cold/warm selection equivalence on the served workload, recall under radix-on, eviction/partial-hit/page-boundary probes)
5. M4 — Close
   - task8: per-step tax re-validation at bs64 (capture-ladder extension, same-batch window, both mem 0.8)
   - task9: locked AC-11 sweep (once) + close-out (results.md regenerated as one coherent document)

Dependency notes: task0 feeds every capacity decision; task1 must land early because every AC-2/AC-3 judgment references it; task3 and task4 are independent of each other but both precede task5 (DEC-3: full M1 first); task6 supersedes task4's table configuration (the int8 table path is deleted with the table per DEC-2 when task6 lands); task7 depends on task1 and is eased by task6 (cold/warm equality partially by construction) but not blocked by it; task8 binds to whichever capacity configuration is final; task9 is last. A shifted bottleneck after any landed change is a queue-feeding event, not a scope expansion of the current task.

## Task Breakdown

Each task includes exactly one routing tag: `coding` (implemented by Claude) or `analyze` (executed via Codex, `/humanize:ask-codex`).

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task0 | Componentized per-rank memory accounting (MLA KV, indexer sidecar, table signatures+scales, graph scratch, allocator slack — capturing runtime fields: pool configurator cell size, `max_total_num_tokens`, `memory_usage.kvcache`, table bytes, graph memory) + max-stable-fraction/capacity probes per config {fp16, int8, table-free mock} × {indexer on/off} × {default vs right-sized envelope: `max_running_requests`≈64, `cuda_graph_max_bs`=64, bounded selector width}, each incl. graph-capture success; output the measured (config → max fraction → KV tokens → bs cap) table | AC-1 | coding | - |
| task1 | Freeze the radix-ON DSA @0.8 directional baseline ladder (3 concurrencies, the loop's comparison column) | AC-2, AC-3 | coding | - |
| task2 | DS-Offload rejection memo with one measured PCIe bandwidth number (document, don't build) | - | analyze | - |
| task3 | DS-mode indexer-cache gate: pool capability flag or DS pool variant, guarded accessors failing loudly, cell-size/configurator accounting update, offload/disagg/radix state-path audit | AC-1.1, AC-7 | coding | task0 |
| task4 | int8 served config + table-aware pool sizing (deduct table bytes before pool sizing) at the task0-measured best fraction; directional ladder at the lifted cap closes M1 | AC-1.1, AC-2, AC-3 | coding | task0 |
| task5 | Absorbed-latent prototype kernel: score-only diagnostic, paged over `req_to_token`, in-kernel fp8 dequant; gated on live-path selection equivalence + oracle recall; validity-invariant analysis documented | AC-5, AC-6 | coding | task3, task4 |
| task6 | Absorbed-latent integration: table-free selector ABI (latent binding), graph capture changes, config validation rejecting cosine/hybrid on the absorbed path, delete `token_label_table.py` + `token_label_write.py` (DEC-2), capacity payoff at mem 0.8; declared value-affecting record | AC-1.2, AC-4, AC-5, AC-6 | coding | task5 |
| task7 | Radix-on enablement + correctness: per-config fixture artifacts (new fixture kind + fail-closed schema for table-free), finalized against the final served selector mode; cold/warm selection equivalence on served workload; recall under radix-on; eviction/partial-hit/page-boundary probes | AC-5, AC-2 | coding | task1 |
| task8 | Per-step tax re-validation at bs64: capture-ladder extension, top-k and score-kernel scaling re-check, same-batch DS-vs-DSA window at mem 0.8 | AC-4 | coding | task6 (falls back to the M1 config if task6 does not land) |
| task9 | Locked AC-11 sweep (3 trials × 600s, once) + loop close-out: results.md regenerated, queue final, evidence pre-flight | AC-2, AC-3, AC-8 | coding | all prior |

Per the draft (binding): `development/loop11/queue.md` is populated at loop kickoff from this table plus kickoff-generated ideas — NOT during plan generation. New mid-loop ideas append to the queue with a one-line compatibility note rather than expanding in-flight tasks; superseded/dropped tasks stay listed with the measured or reasoned cause.

Kickoff queue candidates beyond this table (from the planning idea pass; each enters the queue with the standard one-line compatibility note):

1. EXACT per-step tax reducers, to pull in if the bs64 AC-4 guard runs tight: fuse the radix top-k emit with the logical→physical gather so winners are written as physical slots directly (drops a separate kernel; selected-index plumbing measured ~11.7 ms per 10-step window at bs30); bf16-primary score scratch where the served path is already bf16-authoritative (removes the fp32 scratch plane and the fp32→bf16 copy; must prove selection bit-identity vs the current conversion); workload-bound selector width ladder (e.g. 4096/4352/4608 instead of 5120-only — ~10% less selector work at the cap, more during early decode; only after the graph-memory controls land, since extra buckets add capture variants).
2. Fallback if absorbed-latent slips: trim the table-path label-write projection (avoid projecting full [K_nope|V] just to slice label channels) — exact only if the quantized-linear semantics are preserved; superseded by task6.
3. Parked value-affecting insurance — requires an owner ruling on AC-6 mechanism compatibility BEFORE any work, because selection reuse means not every (layer, step) re-runs query·signature scoring: cross-step lazy top-k refresh (score/reduce/top-k cost ÷ N, recent window force-included) and cross-layer selection sharing in small layer groups.
4. The draft's menu item 6 (serving-side admission/scheduling levers) deliberately remains a conditional queue candidate, not a task — justify with a measured queueing trace first; memory levers treat the cause.

## Claude-Codex Deliberation

### Codex First-Pass Findings (Analysis v1, incorporated)
- Absorbed-latent algebra holds for the raw scorer only; cosine/hybrid token-side normalization defeats the absorption → became the `scorer_norm="off"` hard constraint.
- The draft's GEMM framing was underspecified: scoring walks `req_to_token` logical positions, so the kernel must be paged, not a flat-pool GEMM → adopted as a fixed design constraint.
- Cost parity is a measured risk, not an assumption: arithmetic rises from 32-d label dots to 512-d latent dots per head even though bytes/token are similar (≈528 vs 512 B) → kernel-budget guard retained in task5/task6 gates.
- `kv_b_proj` is a TP-sharded, block-fp8-quantized linear → the v_h build needs the explicit bind-time dequant contract (adopted).
- "Radix equality by construction" was overstated → table-free fixtures and served-workload equivalence checks remain mandatory.
- The written-slot validity semantics need a defined table-free replacement → folded into the task5 prototype contract.
- The DSA indexer sidecar is allocated and unused under DS (~10.3 KB/token) → became task3.
- Capacity criteria must include CUDA-graph capture success, not just boot readouts → folded into AC-1 tests.
- AC ratio bars align with existing `benchmark_compare.py --ac11` constants → adopted as the AC-2/AC-3 anchors.

### Agreements
- Capacity is constrained by memory outside the MLA KV payload (table from headroom + indexer sidecar in-pool); attacking memory is the right root-cause direction.
- Prototype-first absorbed-latent scoring is the right risk split given the table-centric selector ABI.
- `scorer_norm="off"` restriction is technically justified; fail closed on unsupported modes.
- Indexer-cache gating is conceptually valid (DS selection returns before the native indexer) but must be a designed pool capability with corrected cell-size accounting.
- Radix-on validation per served signature config is necessary (artifact fingerprints `signature_dtype`); the table-free mode needs a new fixture kind that replaces, not waives, the label-capture requirement.
- AC-7's strict same-round DSA regression posture is appropriate for the shared surfaces this loop touches.

### Resolved Disagreements
- Indexer-gate scope (draft DEC candidate): Claude initially proposed it as a user decision; Codex argued it is a technical, in-scope implementation choice with safeguards. Resolution: adopted Codex's position — in scope as task3 with the design requirements above; removed from user decisions.
- scorer_norm restriction (draft DEC candidate): Claude initially proposed asking the user; Codex argued it is a hard implementation constraint to enforce in config validation. Resolution: adopted — constraint, not decision.
- Indexer-gate effort estimate: Codex flagged the plan understated the work (many pool methods assume the sidecar exists). Resolution: requirements expanded (capability flag/pool variant, guarded accessors, state-path audit).
- Table-free radix fixture language: Codex flagged it as too loose given the artifact logic requires the label-capture pass. Resolution: new fixture kind + schema with its own fail-closed validator path.
- AC-1 conflation: Codex flagged that M1 levers may hit bs ≥ 64 without reaching ≥390k tokens @0.8. Resolution: AC-1 split into AC-1.1 (floor) and AC-1.2 (op-point), later confirmed by DEC-1 as hard/trend respectively.
- SLO document path: Codex flagged `development/CLIENT_SLOS.md` as stale; verified — only `development/SLOS.md` exists. Resolution: reference corrected throughout the plan (draft text below retains the stale name).
- Equivalence basis: Codex required gating against the live label path, not only the hand-derived formula. Resolution: adopted as the task5 prototype gate.
- Draft-side pending items resolved technically during convergence: absorbed projection precision (bind-time fp32 dequant of W_UK matching attention semantics + fp32 accumulation, confirmed against the per-128-block scale layout); DSA radix-ON baseline timing (frozen early as task1, one directional ladder — not a locked sweep).

### Convergence Status
- Final Status: `converged` (2 rounds: round 1 produced 7 required changes + 2 decision reclassifications, all adopted in candidate v2; round 2 returned zero required changes and zero disagreements; round-2 optional improvements — final-mode artifact gating for task7, the 528-byte read correction, named runtime fields for task0 — were adopted).
- Post-convergence refinement: the four planning-annotation comments from the 2026-06-12 idea pass (Claude + Codex consultation via ask-codex) were integrated as plan text via refine-plan; classifications and dispositions are recorded in `.humanize/plan_qa/plan-qa.md`. No new pending decisions were introduced (the parked AC-6 ruling is a conditional gate that arises only if the related queue candidates are ever pulled).

## Pending User Decisions

All decisions were resolved by the owner in the gen-plan discussion on 2026-06-12. None remain PENDING.

- DEC-1: AC numerics — hard bars vs optimization trends, and the verdict venue
  - Claude Position: bs cap ≥64 hard; TTFT ≤1.10×, aggregate ≥0.95×, tax ≤1.10× hard at the locked AC-11 sweep; ≥390k tokens @0.8 as trend (recommended).
  - Codex Position: same split proposed; flagged that ratios already match the `--ac11` comparator constants; insisted the venue question is a real owner decision.
  - Tradeoff Summary: all-hard risks failing the loop on a capacity number even when bs64 + SLO bars are met; all-trend removes the teeth from a loop whose comparator already encodes the gates.
  - Decision Status: **Proposed split adopted — HARD: bs cap ≥64, TTFT ≤1.10×, aggregate ≥0.95×, tax ≤1.10×, all judged at the closing locked AC-11 sweep (directional ladder = iteration signal only). TREND: ≥390k tokens @ mem 0.8 (AC-1.2).**
- DEC-2: Ship shape for absorbed-latent scoring
  - Claude Position: delete `token_label_table.py` + `token_label_write.py` outright when the gates pass (doctrine: delete old paths when new paths work; git revert is the rollback).
  - Codex Position: acknowledged doctrine favors deletion; kept it as a real policy/rollback decision (operational tradeoff).
  - Tradeoff Summary: deletion gives a single authoritative path and no dual capture variants; a one-loop flag keeps rollback/A-B at the cost of dual-path maintenance and 5.29 GB of allocatable table.
  - Decision Status: **Delete outright when the task5/task6 gates pass.**
- DEC-3: Sequencing posture for the capacity levers
  - Claude Position (Codex-converged default): land task3 early unconditionally; skip task4 unless the absorbed-latent prototype stalls (minimal throwaway work).
  - Codex Position: technical after task0 unless the owner imposes a budget constraint; same default.
  - Tradeoff Summary: the conditional path minimizes throwaway work but delays a served fallback config; full-M1-first gives the earliest capacity floor and a served fallback at the cost of int8-table work that task6 later supersedes.
  - Decision Status: **Full M1 first — land task3 AND task4 and re-run the directional ladder before starting M2 (owner chose the earliest capacity floor + served fallback over minimal throwaway work).**
- DEC-4: Loop budget/scope (draft: "full menu vs capacity-first"; also fixes the lower-bound acceptance)
  - Claude Position: full menu M0–M4 (recommended) — radix-on is required for the honest radix-ON comparison the owner already mandated.
  - Codex Position: genuinely an owner decision; either is coherent.
  - Tradeoff Summary: capacity-first focuses the loop on the root cause but leaves DS serving radix-off against a radix-ON bar; full menu carries fixture risk inside the loop.
  - Decision Status: **Full menu M0–M4. The lower bound stands as written: int8/indexer-gate + lifted cap + radix-on validated is an acceptable loop outcome if absorbed-latent integration misses (prototype-gate evidence still required).**

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", "task-N", "DEC-", or similar workflow markers
- These terms are for plan documentation only, not for the resulting codebase
- Use descriptive, domain-appropriate naming in code instead (see `.claude/rules/speculative-naming.md` trigger when touching speculative/attention-backend identifiers)

### Binding Protocol Sections (carried verbatim from the draft below — they are requirements, not suggestions)
- **Subagent usage / context discipline (MANDATORY):** delegate code reconnaissance, artifact digestion, well-scoped implementation slices, measurement babysitting, and document drafting to subagents by default; the main context holds decisions, gate verdicts, queue updates, owner-decision handling, and final review of every subagent product. Every subagent product is reviewed (diffs in full, claims spot-checked) and passes the same gates as main-agent work.
- **Iterate→measure protocol (per queue task, exactly one cycle):** implement → verify quality FIRST (recall gate, cross-rank bit-identity, selection capture or declared record, radix checks where relevant) → measure (capacity probe → one-batch kernel guard → targeted serving spot-check → full ladder only at milestone gates; locked AC-11 sweep once at close) → read the gap (shifted bottleneck = queue-feeding event) → keep or revert.
- **Measurement discipline:** one TP=8 server at a time; frozen references never re-run; 1-trial honesty until AC-11; graph-mode numbers binding; `git push` every round boundary.
- **Hardware/op-point:** single node 8×H200, TP=8, GLM-5.1-FP8, fp8_e4m3 KV, page 64, custom-all-reduce ON, `flashmla_kv` both phases, CUDA graph ON; never `expandable_segments` for serving; workload gsp 4096/512, ~55% prefix, seeds {16:213, 32:431, 64:31234}, server seed 20260607; the 20260612 drivers are the serving recipe of record.
- **Owner decisions (2026-06-12, fixed):** lossiness bar (recall-gated ±0.5pp fail-closed + cross-rank bit-identity, declared value-affecting records — bitwise identity vs the fp16-label path NOT required for latent scoring); int8 signatures approved as a served config; the comparison bar is radix-ON DSA and DS radix-on enablement + correctness validation is in scope.

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

# Loop 11 Draft — Lift the DS batch cap: kill the TokenLabelTable tax, fix the TTFT tail, serve radix-on

> Written 2026-06-12, after **Loop 10 closed** (Case-1 one-batch decode 480,989 → **361,824 µs**,
> 1.403× → **1.055×** vs the frozen DSA floor; width-bucketed selector graphs, compact W=5120
> buffers, pinned bf16 two-shot score reduce, bf16-authoritative radix top-k all landed) and after
> the **2026-06-12 serving re-measurement** (`development/profiling/runs/20260612/`,
> `development/profiling/results.md`) confirmed the per-step work is done: at matched batch DS
> decode is **1.05× DSA** and DS now **clears the 30 tok/s decode-SLO floor at conc 16/32/64**.
> **This loop changes the goal and the scoring method**: the gap is no longer per-step kernel
> time — it is **serving behavior at the SLO concurrency range (16–64)**: tail TTFT (queueing),
> aggregate throughput, and the memory footprint that causes both.
> Feed this through `gen-plan` once scope is confirmed.

---

## What this work is (and is NOT) — read first

**This is not frontier LLM development. There is no novelty here.** We are reproducing the results
of a **2-year-old paper** (Double Sparsity, arXiv:2408.07092,
`development/past_implementations/double_sparsity_paper_2408.07092.pdf`) on top of a **fully
open-source codebase** (SGLang), and checking whether it can be made to perform on **MLA models
that are well behind the frontier** (GLM-5.1). The contribution, if any, is engineering: making an
existing sparse-attention idea cheap enough to be worth its opt-in slot on a model that already
ships a *trained* sparse indexer (DSA). This is purely educational; the algorithm theoretically
has no hope of beating the highly optimized DSA algorithm — there's a reason it was introduced two
years ago and nobody adopted it in a frontier model. We want to see how far it can go and learn
performance engineering by doing so.

## Objective

Close the remaining **serving-level** DS-vs-DSA gap at concurrency 16–64 on the canonical client
workload (GLM-5.1-FP8, 8×H200 TP=8, gsp 4096-ISL / 512-OSL / ~55% prefix hit, per
`development/CLIENT_SLOS.md`):

1. **Lift the DS decode-batch cap** (bs30 → ≥64) by eliminating or radically shrinking the
   per-rank **TokenLabelTable** (measured **5.29 GB/rank**) and the headroom it consumes, so DS
   serves at the same memory operating point as DSA (mem 0.8, ~410k-token KV pool).
2. **Fix the tail TTFT** that the cap causes (queueing at conc 32/64), and
3. **Serve and validate radix-on for DS** — clear the fixture gate AND prove radix-cache
   correctness with DS end-to-end — so the comparison against the production DSA default
   (radix-ON) is honest and DS benefits from the same 55%-prefix reuse.

while **keeping the per-step decode tax bounded** (the loop-10 win must not regress) and keeping
the DS concept intact.

**Scope (per user, same posture as loop 10): any scale of refactoring — from complete rewrites to
small code changes — is welcome**, provided it:

1. **Keeps the core DS concept**: offline channel mask → per-token signatures (materialized **or
   implicitly derived**) → query·signature scoring → top-k selection → sparse MLA decode. No dense
   fallback, no DSA-indexer substitution.
2. **Meets the quality bar (owner-decided 2026-06-12, do not relitigate):** NIAH recall@2048
   within ±0.5pp fail-closed per landed change + cross-rank selection bit-identity HARD.
   Exact-by-design changes additionally prove bit-identical selection. Changes that are
   value-affecting by construction (fp8-sourced signatures, int8 tables) are **admissible** under
   the recall gate and are recorded as declared value-affecting decisions (DEC-L10-1 pattern).

---

## Current state (measured 2026-06-12, HEAD `ce914e5b9`)

### The serving gap (directional ladder, radix-off both sides, 1 trial / 180s window)

| conc | config | decode-TPS p50 | agg tok/s | achieved conc | TTFT med/p99 (s) |
|---|---|---:|---:|---:|---:|
| 16 | DS @0.7 (bs30) | 39.4 | 494 | 16.0 | 3.6 / 3.7 |
| 16 | DSA @0.7 (bs30-cap) | 40.9 | 418 | 16.0 | 7.1 / 7.1 |
| 16 | DSA @0.8 (bs64) | 38.7 | 404 | 16.0 | 7.1 / 7.2 |
| 32 | DS @0.7 (bs30) | 33.0 | 574 | 26.0 | 6.4 / **28.4** |
| 32 | DSA @0.7 (bs30-cap) | 31.9 | 464 | 27.2 | 12.8 / **41.6** |
| 32 | DSA @0.8 (bs64) | 31.5 | 541 | 32.0 | 14.2 / 14.2 |
| 64 | DS @0.7 (bs30) | 33.3 | 577 | 39.4 | 23.9 / **45.2** |
| 64 | DSA @0.7 (bs30-cap) | 32.0 | 465 | 41.4 | 34.8 / **60.2** |
| 64 | DSA @0.8 (bs64) | 25.1 | **676** | 64.0 | 28.1 / 28.1 |

Three facts this table proves (full analysis: `development/profiling/results.md` §1):

- **The selector is no longer the problem.** At matched bs30 DS and DSA decode-TPS are within ~4%
  (conc16: 40.9 vs 39.4 = the 1.05× kernel ratio exactly). Per-step DS-specific kernel work is
  ~74.5k µs vs DSA's 18.4k fused indexer — net +18k µs/window (was +289k pre-loop-10).
- **The TTFT tail is the batch cap, not DS.** DSA *also* capped to bs30 has a *worse* p99 tail
  (41.6 / 60.2 s) than DS (28.4 / 45.2 s). Whoever runs batch 30 at conc ≥32 queues.
- **DSA's remaining advantage is purely the batch it can admit**: at conc 64 it wins aggregate
  (676 vs 577) and tail (28.1 vs 45.2 p99) because it admits 64 while DS admits ~30.

### The memory diagnosis (measured, boot logs in `runs/20260612/*/serve.log`)

| quantity | value | source |
|---|---|---|
| KV pool @ mem 0.7 (DS **and** DSA — identical) | **142,208 tokens / 8.14 GB** | case1/case2/cap30 serve.log |
| KV pool @ mem 0.8 (DSA) | **410,560 tokens / 23.51 GB** | case3 serve.log |
| Decode-batch cap at ISL 4096+512 | floor(cap/4608): **30** vs **64+** (0.8 pool supports ~89) | derived |
| **TokenLabelTable** | **5.29 GB/rank** — `L=78, T=142,272, H=8 heads/rank, D=32 ch, fp16` ≈ 39.9 KB/token | case1 boot log |
| The KV pool the table indexes | 8.14 GB → table = **~65% of the KV cache it serves** | ratio |

**The precise causal chain** (note: it is NOT "the table shrinks the KV pool" — at the same
mem fraction DS and DSA get identical pools): the table (5.29 GB) + DS capture state is allocated
**after** the static pool, out of runtime headroom → DS cannot raise `mem_fraction_static` above
0.7 (DSA serves at 0.8/0.85) → DS is stuck with the 142k-token pool → bs30 cap → admission
queueing at conc ≥32 → p99 TTFT tail + aggregate loss. **Memory is the root; everything else is
symptom.**

---

## First-principles analysis — do we even need the TokenLabelTable?

### What the table is for

At decode, DS scores every cached token: `score[t] = Σ_{c∈S_h} w_c · q_c · k_nope[t,h,c]` per
head, then cross-head max, cross-rank reduce, top-k. The table caches `k_nope[t, h, S_h]` (32 of
192 nope dims per head, fp16) so scoring never touches the full K. That is the paper's design —
**for MHA, where full K is materialized in cache anyway** and labels are r/D = 32/192 ≈ 17% of K
(≈8% of KV). The label cache is "small" by construction *in that architecture*.

### Why MLA breaks the paper's premise

GLM-5.1 is MLA: the KV cache stores a **shared 512-dim latent** `c_kv` (+64 rope) per token —
~57 KB/token for all 78 layers — and `k_nope[h] = W_UK[h] · c_kv` is **never materialized** in
cache. DS-for-MLA therefore materializes per-head signatures **on top of** an already-compressed
cache: 78 × 8 heads/rank × 32 ch × 2 B ≈ 39.9 KB/token — **nearly doubling per-token decode
memory**. The paper's "labels are a small fraction" premise simply does not transfer; this is the
structural reason DS is batch-capped while DSA (whose indexer keys off compact per-token
fp8 representations) is not.

### How the past implementations handled it — and why none of it transfers

(code-grounded, verified 2026-06-12 against the actual repos; survey context in
`development/past_implementations/study/00-survey.md`)

None of the three implementations ever *faced* this problem — each lived in a regime where the
label cache was structurally cheap, and each "handled" memory accordingly:

- **DoubleSparse standalone (paper repo)** — `models/model.py` `KVCache` (71–96): `k_label`
  `[B, S, H, r]` bf16, `heavy_channel_num=32` of `D=128` (Llama) → labels = 25% of K = **~12.5%
  of KV**. Dense GPT-Fast-style pre-allocation (`max_batch_size × max_seq_length`), no paging, no
  continuous batching, batch ≈ 1. Memory capacity was never a tracked metric; the overhead is
  small by MHA construction. **Handling = none needed.**
- **DS-Offload (the paper's ONLY memory tool)** — `offloading/model.py:82–148`: when GPU memory
  *did* bind (long context), they did **not** shrink the label cache — they kept `k_label` fully
  GPU-resident (it is read **densely** every decode step) and pushed the **full K/V** to
  CPU-pinned memory (`k_cache_cpu`/`v_cache_cpu`, `pin_memory=True`), keeping only a
  `heavy_const=256`-row GPU staging buffer and gathering the selected rows per step with DGL
  `gather_pinned_tensor_rows`. The code is research-grade (synchronous `k_val.cpu()` writes in
  `update()`). Their trade: GPU bytes/token drop from `2·H·D` (K+V) to `H·r` (~8× for Llama) at
  the price of a per-step PCIe gather — affordable at batch ≈ 1 with no latency SLO.
  **Handling = offload the sparsely-read tensor, keep the densely-read one resident.**
- **sglang fork** — `DoubleSparseTokenToKVPool` (`memory_pool.py:1972–2060`): `label_buffer`
  `[size, H, r]` per layer, same dtype as KV, allocated **inside the KV-pool budget** beside
  `k_buffer`/`v_buffer` — so there the labels really did shrink token capacity, by
  r/(2D) ≈ 12.5%… which nobody noticed, because the fork ran MHA models with CUDA graphs
  disabled and was never pushed to SLO serving. No int8, no offload port.
  **Handling = ignore it (affordable in MHA).**
- **Twilight**: monkey-patches attention and reuses the base model's existing caches — selector
  policy research; cache memory entirely out of scope.

**The synthesis that matters for this loop:** the problem is genuinely novel to our setting — it
is the *product* of (MLA latent compression, which makes per-head labels ~65% of the KV they
index instead of ~12%) × (continuous-batching SLO serving, which makes KV capacity the throughput
currency). And DS-Offload's one transferable lesson **inverts** on us: their densely-read tensor
(labels) was small and their sparsely-read tensor (full KV) was big, so offload worked; our
densely-read tensor (the table) is the big one and our sparsely-read tensor (fp8 latent) is
already compact — there is nothing profitable to move. There is no prior-art answer to copy;
the escape hatch above comes from MLA's own structure. (Note the convergence: DSA's production
indexer made the same structural choice — its "signature" is a compact per-token representation
co-located with the cache rather than a separate per-head table.)

**DS-Offload evaluated for our op-point — rejected by arithmetic (document, don't build):** the
gather is top-2048 × 78 layers × 576 B ≈ **92 MB per decode step per request**; at bs30 ≈
2.8 GB/step over ~50 GB/s PCIe ≈ **55 ms/step** — alone double the 33 ms/step budget that
30 tok/s implies. And the *labels* (the thing we'd want to offload — our table) **cannot** be
offloaded at all: they are read densely for every cached token every step. Offloading helps the
paper's single-request long-context regime, not TP=8 SLO serving. The loop should write this
rejection down with one measured PCIe number and move on — unless it finds a genuinely new angle
(e.g. context-horizon hybrid), which would be a new lossiness discussion.

### The MLA escape hatch — absorbed-latent scoring (signatures without a table). PRIMARY idea.

MLA's structure offers what MHA never could: since `k_nope[h] = W_UK[h] · c_kv`,

```
score[t,h] = Σ_{c∈S_h} w_c · q_c · k_nope[t,h,c]
           = ( Σ_{c∈S_h} w_c · q_c · W_UK[h][c,:] ) · c_kv[t]   =   v_h · c_kv[t]
```

with `v_h ∈ R^512` computed **once per step per head** (32×512 MACs — trivial). Scoring becomes a
GEMM-shaped op `[bs, 8, 512] × [T, 512]^T` per layer **against the resident fp8 KV latent** — the
signature IS the latent; the table, the prefill label-write hook, and 5.29 GB/rank all **disappear
exactly**, while the offline channel mask (`S_h`, `w_c`) keeps doing its job on the query side.

Cost sanity (verify by measurement, not assumption): bytes read per layer ≈ T × 512 B (fp8) —
**identical** to today's label read (T × 8 heads × 32 ch × 2 B = 512 B/token), but one read now
serves all 8 heads (8× better arithmetic intensity) and the op is tensor-core-shaped. The current
`_logical_score_kernel` budget is 23.5k µs/window; this should land at-or-under, but the kernel
must dequantize fp8 in-kernel (scale-layout risk, below).

Known risks/caveats (design for them):
- **(a) Value-affecting by construction**: signatures derived from the **fp8-quantized** latent vs
  today's bf16-pre-quant labels → selection is NOT bit-identical to the current path. Owner
  decision (2026-06-12): admissible under the recall gate (±0.5pp fail-closed + cross-rank
  bit-identity), recorded as a declared value-affecting change. Note the self-consistency
  argument: attention itself reads the same fp8 latent — scoring now sees exactly what attention
  sees.
- **(b) fp8 scale layout**: the pool stores fp8_e4m3 with scale metadata (see
  `_quantize_k_cache_fast_kernel` in the traces). The scoring kernel needs cheap in-kernel
  dequant; verify the scale granularity (per-token/per-page/per-128) before committing to a
  kernel design.
- **(c) `W_UK` availability**: per-head slices are already sharded per rank for attention
  (`kv_b_proj` in `deepseek_v2.py`); the absorbed `v_h` build must read them without a new
  all-gather.
- **(d) rope dims excluded**: today's labels are k_nope-only (rope prefix excluded by
  calibration) — the latent path is consistent by construction; assert it.
- **(e) Transport unchanged**: per-rank local head-max → cross-rank reduce (the loop-10 pinned
  bf16 two-shot) is untouched.
- **(f) Radix synergy (this is large)**: with signatures derived from cached KV, **cold-vs-warm
  label equality holds by construction** — the radix label-capture fixture's failure mode
  structurally disappears; only the FP8 scale-stability half remains to prove.
- **(g) Capture surface**: the scoring kernel changes shape under CUDA graphs (width buckets
  carry over; the labels-gather stage drops out). Budget capture memory; reuse the loop-10
  shared-DSGraphState pattern.

### The full option space (so gen-plan sequences consciously)

| option | table size | mechanism | lossiness posture | est. effort |
|---|---|---|---|---|
| A. Absorbed-latent scoring | **0** | derive signatures from fp8 latent | recall-gated, declared (owner-approved) | the big build |
| B. int8 signatures (EXISTS) | 2.97 GB (0.5625×) | `signature_dtype=int8`, ≥0.99-overlap gated | already characterized + gated | config flip + sweep |
| C. Headroom audit / mem-split optimum | n/a | account every GB at 0.7; find max stable fraction per table variant | exact | measurement task |
| D. int4 / blockwise-quant table | ~1.6 GB | new quant path | new recall/overlap gate needed | medium; superseded by A |
| E. DS-Offload | n/a | CPU-pinned KV + gather | rejected by PCIe math above | document only |
| F. Radix-on for DS | n/a (TTFT lever) | clear fixture gate + correctness validation | exactness fixtures + recall under reuse | bounded chore + validation |

---

## Candidate menu (ranked — `gen-plan` picks/sequences; each is one implement→measure cycle)

0. **Memory accounting + cheap-cap probes (DO FIRST — one day, may move the headline alone).**
   Account exactly where the headroom goes at mem 0.7 (table 5.29 GB + capture + workspace +
   activations, per-rank). Then probe the max stable `mem_fraction_static` for: fp16 table, int8
   table (owner-approved served config), and table-free (override/mock) — each probe is a boot +
   capacity readout + short serve smoke. Output: a measured `(config → max fraction → KV tokens →
   bs cap)` table. int8@~0.75 alone may lift bs30 → ~45+; this is the loop's floor result and the
   baseline every later task is judged against.
1. **PRIMARY — absorbed-latent scoring: eliminate the TokenLabelTable** (§escape-hatch above,
   risks a–g). Gates: recall ±0.5pp fail-closed, cross-rank bit-identity, declared-value-affecting
   record, per-step kernel guard (logical-score bucket ≤ current 23.5k µs/window at bs30; total
   window ≤ guard), THEN the capacity payoff: serve at mem 0.8, capacity ≥ ~390k tokens, bs cap
   ≥ 64, re-run the serving ladder.
2. **Radix-on for DS — enablement AND correctness (owner-scoped 2026-06-12).** Two halves:
   (a) *Enablement*: run the two fixtures on the GLM-5.1 config under `SGLANG_DS_RADIX_OVERRIDE=1`
   (label-capture cold/warm bit-equality: `test/manual/test_dsv32_radix_label_capture_fixture.py`;
   FP8 scale stability: `test_dsv32_fp8_scale_stability.py`), `write_radix_fixture_state(...)`,
   serve via `RADIX_FIXTURE_ARTIFACT`. (b) *Correctness beyond the fixtures*: cold-vs-warm
   response/selection equivalence on the served workload (selection-capture diff across a cache
   hit), recall@2048 under radix-on, and an eviction/partial-hit edge probe (page-boundary hits at
   page 64). If task 1 has landed, half the fixture risk is gone by construction (risk f). Payoff:
   TTFT at the 55%-prefix workload + the honest comparison vs radix-ON DSA.
3. **int8 served config as the stepping stone** (owner-approved): flip `signature_dtype=int8` +
   the task-0 best fraction into `serve_double_sparsity.sh` defaults for the loop's served DS
   column until task 1 supersedes it. Re-run the directional ladder at the lifted cap.
4. **Per-step tax at the lifted batch.** Loop-10 selector kernels were tuned/captured at bs≤30
   and width buckets [5120]. At bs64: extend the capture ladder, re-check radix top-k and
   logical-score (or its absorbed successor) scaling with bs, re-measure the DS-vs-DSA same-batch
   window at bs64 (`bench_one_batch_server`, both at mem 0.8). Guard: ratio ≤ ~1.10.
5. **Fresh DSA radix-ON baseline ladder** (the bar this loop is judged against — owner decision):
   one early run of the 3-concurrency directional ladder against DSA @0.8 radix-ON, frozen as the
   loop's comparison column. (Today's 20260612 ladder is radix-off and stays the radix-off
   reference.)
6. **WILDCARD — serving-side levers** (only if memory levers stall): admission/scheduling
   interaction with the cap (priority, chunked-prefill interleave at high conc). These treat the
   symptom; the memory levers treat the cause. Justify with a measured queueing trace first.

Ideas found while working that beat these **replace them** — the menu is a starting point, not a
contract. Complete rewrites of the signature/scoring pipeline are acceptable per the scope
statement, under the same gates.

---

## Open scope + the task queue (`development/loop11/queue.md`)

**The scope of this loop is deliberately NOT fixed to the menu above.** The agent is expected —
and incentivized — to invent additional optimization ideas while working (reading code, reading
profiles and serving traces, watching where the bottleneck moves after each landed change) and to
pursue them **in the same loop**, as long as each new idea is compatible with the optimizations
already completed and accepted.

`queue.md` (already created, deliberately empty) is the loop's **self-contained task queue and
checklist** — the single source of truth for what is planned, in flight, done, or dropped:

- **Populating the queue is the FIRST task of the loop**, once plan refinement has completed and
  the loop kicks off — seed it from the final plan's tasks plus any further ideas generated at
  kickoff. Do NOT populate it during plan generation.
- Every task gets a queue entry: id, description, targeted quantity (GB freed / tokens gained /
  TTFT-p99 / window µs), expected effect, lossiness posture, compatibility note vs already-landed
  changes, status. New ideas discovered mid-loop are **appended as queued entries** (with that
  one-line compatibility check) rather than expanding the current task's scope.
- A task is marked completed only after its gates pass (quality teeth + the relevant measurement).
  Dropped or superseded tasks stay listed with the measured/reasoned cause — no silent deletions.
- The queue is committed with each round so reviews see the same checklist the agent works from.

## Subagent usage (context discipline — MANDATORY, not optional)

**The main agent's context window is the scarcest resource in this loop.** This is a long
multi-round loop over a large codebase with multi-thousand-line traces, serving JSONLs, fixture
logs, and serve logs — reading those raw in the main context is how loops degrade: the agent
forgets early decisions, re-reads files it already understood, and burns budget re-deriving
state. Use subagents **liberally and by default**; the main context should hold decisions,
verdicts, and the queue — not raw artifacts.

**Delegate by default (the main agent should almost never do these inline):**

- **Code reconnaissance** — call chains, "where is X allocated/bound/freed", shape/dtype tracing
  across `dsa_backend.py` / `memory_pool.py` / `model_runner.py` → Explore subagents; only the
  conclusion (file:line + one-paragraph mechanism) returns.
- **Artifact digestion** — torch-trace/`kern_sum.csv` parsing, serving-JSONL ladder extraction,
  serve-log capacity/headroom readouts, fixture-run logs → analysis subagents that return the
  numbers table, never the raw dump. (The 20260612 run did exactly this: a subagent re-parsed
  three traces with a corrected classifier and returned a one-page `breakdown.md` — and caught
  two attribution errors in the process.)
- **Well-scoped implementation slices** — a kernel variant, a fixture runner script, a probe
  driver — implementation subagents with a tight contract (inputs, outputs, gates to run,
  files it may touch).
- **Long-running measurement babysitting** — boot→sweep→teardown drivers run detached/in
  background; the main agent reads the terminal marker and the summary file, not the stream.
- **Document drafting** — first drafts of round summaries / close-outs from the ledger, reviewed
  and corrected by the main agent before commit.

**Keep in the main context (never delegate):** keep/revert decisions, gate verdicts, queue
updates, owner-decision handling, and the final review of every subagent product.

Two hard rules (unchanged from loops 9/10):

1. **Every subagent's work is carefully reviewed by the main agent before it is trusted** — diffs
   read in full, claims spot-checked against the actual artifact (subagents have measurably
   non-zero error rates on attribution and edge cases — the 20260612 sweep's classifier catch cut
   both ways), never relayed unverified.
2. Nothing a subagent produced lands without passing the **same verification gates** as
   main-agent work. Subagents save context, not review.

---

## The iterate→measure protocol (serving-level heartbeat; loop-9/10 tooling reused)

For **each** queue task, exactly one cycle — run it **context-lean**: reconnaissance, artifact
digestion, and measurement babysitting go to subagents per the section above; only verdicts,
numbers tables, and diffs-under-review enter the main context:

1. **Implement** (DS path; shared surfaces — memory accounting, radix, graph runner — trigger the
   stricter AC-7 DSA regression in the same round).
2. **Verify quality FIRST** (the teeth, all existing tools):
   a. **NIAH recall@2048** (`loop7/niah_oracle_sweep.py` + `loop9/oracle_recall_summary.py`),
      fail-closed ±0.5pp vs the frozen `loop9/runs/20260610_m0/recall_baseline.json`.
   b. **Cross-rank selection bit-identity** (hard, every gate run).
   c. **Exact changes** (config/mem-split/capture plumbing) additionally prove bit-identical
      selection via `loop9/selection_capture_tool.py` vs the pre-change digest. Value-affecting
      changes (latent signatures, int8) instead record the declared decision + their gate
      evidence.
   d. **Radix-on changes**: the two fixtures + cold/warm selection-capture equivalence + recall
      under radix-on (task-2 definition).
3. **Measure the loop's primary quantities** (cheap heartbeat, in order of cost):
   a. **Capacity probe** — boot + `max_total_num_tokens` readout → derived bs cap (instant; this
      is the loop's tightest feedback signal).
   b. **One-batch kernel guard** — 10-step torch decode window at the common batch
      (`runs/20260612` recipe): DS total vs the 361,786 µs reference and vs DSA same-batch
      (ratio ≤ ~1.10). Protects the loop-10 win.
   c. **Targeted serving spot-check** — conc-64 (and 32 when TTFT-relevant) directional run
      (1 trial, 60s/180s, the `runs/20260612/stage*` driver pattern) against the frozen
      comparison columns.
   d. **Full 3-concurrency ladder** only at milestone gates (capacity landed; radix-on landed;
      loop close). **Locked AC-11 sweep (3 trials × 600s, `benchmark_compare.py --ac11`) once, at
      loop close**, as the publication artifact.
4. **Read the gap** against the loop's AC quantities (capacity, TTFT p99 per conc, aggregate,
   tax ratio). A shifted bottleneck is a queue-feeding event: append the newly exposed target to
   `queue.md` as a candidate task.
5. **Keep or revert.** Bank only what passes step 2 and moves its targeted quantity. Ledger:
   `development/loop11/results.md` — **rewrite-over-append** when state changes; one
   authoritative current-state section (loop-9/10 lesson, reinforced by the loop-10 methodology
   report: regenerate terminal docs, never layer corrections).

Measurement discipline carried forward: one TP=8 server at a time; frozen references
(loop-9/10 run dirs, the 20260612 radix-off ladder, the Case-2 342,857 floor) are **never
re-run**; serving numbers are 1-trial directional until the closing AC-11 sweep — claim
accordingly (`bound_valid`-style honesty: no lower-bound/SLO-pass claims beyond what the trial
count supports); eager microbenches lie about captured-replay behavior — graph-mode numbers are
binding; `git push` at every round boundary (cluster pre-emption insurance).

---

## Acceptance criteria (draft — `gen-plan` formalizes the numbers)

1. **AC-1 Capacity (the root cause).** DS serves at the DSA memory op-point: mem 0.8 with
   KV capacity within ~5% of DSA's at the same fraction (≥ ~390k tokens), derived decode-batch
   cap **≥ 64** at the 4608-token workload shape. Suggested hard bar: bs cap ≥ 64; stretch:
   table bytes = **0** (absorbed-latent landed).
2. **AC-2 Tail TTFT at the SLO range (the symptom).** At conc 16/32/64 against the **frozen
   radix-ON DSA @0.8 baseline** (task 5): DS p99 TTFT ≤ ~1.10× DSA's per concurrency, and DS
   meets the absolute bar (p99 < 22 s) wherever DSA does. DS must no longer be
   admission-capped below nominal concurrency at conc ≤ 64 (achieved ≈ nominal).
3. **AC-3 Throughput.** DS per-request decode-TPS p50 ≥ 30 maintained at conc 16/32/64
   (already met — must not regress), and DS aggregate ≥ ~0.95× DSA radix-ON at conc 64.
4. **AC-4 Per-step tax guard.** DS-vs-DSA same-batch one-batch decode window ratio ≤ ~1.10 at
   the new common batch (bs64, both mem 0.8), and the bs30 window stays ≤ ~380k µs — the
   loop-10 win is not traded away for capacity.
5. **AC-5 Quality (owner-decided bar).** Recall@2048 ±0.5pp fail-closed per landed change;
   cross-rank bit-identity HARD; exact changes prove selection bit-identity; value-affecting
   changes (fp8-sourced signatures, int8 table) recorded as declared decisions with gate
   evidence. **Radix-on correctness**: both fixtures pass on the GLM config, cold-vs-warm
   selection equivalence demonstrated on the served workload, recall under radix-on within the
   same bar, eviction/partial-hit probe clean.
6. **AC-6 DS concept intact.** Offline mask → signatures (materialized or absorbed) →
   query·signature scoring → top-k → sparse MLA decode. No dense fallback; no DSA-indexer
   substitution.
7. **AC-7 DSA-native default un-regressed — strict.** Changes to shared surfaces (memory
   accounting/pool sizing, radix plumbing, graph runner) trigger the mandatory DSA regression in
   the same round (DS-off smoke + frozen Case-2 recipe re-validation + radix-ON DSA serving
   smoke). The shipped DSA default stays untouched in behavior and performance.
8. **AC-8 Protocol/ledger/queue discipline.** Queue current every round (statuses, appended
   ideas with compatibility notes, recorded drops); evidence pre-flight before each round handoff
   (artifact exists + tracked + claim matches artifact — pre-flight sentences describe the
   POST-commit state); results.md rewritten not layered; one trial honesty; frozen references
   intact; push every round.

## Files to read first

- **Current state:** `development/profiling/results.md` (the 2026-06-12 re-measurement — the
  whole motivation), `development/profiling/runs/20260612/breakdown.md` (per-kernel ground truth),
  `runs/20260612/serving/SUMMARY.txt`, `development/loop10/reviews/task10_closeout.md` (what
  landed + decision record), `development/loop10/results.md`.
- **The table and its writers (what gets deleted/replaced):**
  `python/sglang/srt/layers/attention/double_sparsity/token_label_table.py` (the 5.29 GB),
  `token_label_write.py` (prefill write hook), `selection_kernel.py` (`_logical_score_kernel`,
  `project_query_onto_channels`, `reduce_token_scores`, `retrieve_topk_graph_safe`),
  `cuda_graph.py` (DSGraphState, capture parity), `config.py`.
- **The MLA structure to absorb against:** `python/sglang/srt/models/deepseek_v2.py`
  (`kv_b_proj` / `W_UK`, `_select_topk_indices` bind site), the fp8 KV pool layout + scale
  handling (`memory_pool.py`, `_quantize_k_cache_fast_kernel` call sites in
  `dsa_backend.py`).
- **Memory accounting:** `model_runner.py` (mem-fraction → pool sizing → headroom),
  `serve_double_sparsity.sh` (the 0.6/0.7 history in comments), capture-memory notes in
  loop-10's task4/task6 records.
- **Radix gate:** `validator.py` (`record_radix_fixture_passed`, the refusal at :323),
  `page_table_adapter.py`, `radix_fixture_capture.py`, the three fixtures under `test/manual/`
  (`test_dsv32_radix_label_capture_fixture.py`, `test_dsv32_fp8_scale_stability.py`,
  `test_dsv32_radix_cache_fixture.py`), the launcher's `RADIX_FIXTURE_ARTIFACT` path.
- **Serving measurement recipes:** `development/profiling/runs/20260612/_env.sh` +
  `stage1..5_*.sh` (boot→capacity→sweep→profile drivers), `development/benchmark.sh`,
  `benchmark_baseline.sh`, `benchmark_compare.py` (AC-11 mode + its refusal rules),
  `development/CLIENT_SLOS.md`.
- **Past implementations (the as-built evidence):** `development/past_implementations/study/`
  (`00-survey.md` §DS-offload, `08-current-system-architecture.md`), `DoubleSparse/offloading/`
  (DS-offload), `sglang-last-with-double-sparsity` pool/backend files.
- **Gates:** `loop9/selection_capture_tool.py`, `loop9/oracle_recall_summary.py`,
  `loop7/niah_oracle_sweep.py`, frozen `loop9/runs/20260610_m0/recall_baseline.json`.
- **Doctrine:** `CLAUDE.md`; `.pensieve/` maxims; `.humanize/bitlesson.md`.

## Hardware / op-point

Single node 8×H200, TP=8, GLM-5.1-FP8, fp8_e4m3 KV, page 64, custom-all-reduce ON,
`flashmla_kv` both phases, CUDA graph ON. **Never set
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments` for serving** (breaks custom-all-reduce IPC at GLM
TP=8 — BL-20260608). One TP=8 server at a time. Workload: gsp 4096/512, ~55% prefix, seeds
{16:213, 32:431, 64:31234}, seed 20260607 server-side. The 20260612 drivers are the serving
recipe of record for this loop.

## Decisions already made by the owner (2026-06-12 — do NOT relitigate in gen-plan)

1. **Lossiness bar:** recall-gated (±0.5pp fail-closed + cross-rank bit-identity) with declared
   value-affecting records — bitwise identity vs the fp16-label path is NOT required for the
   latent-scoring candidate.
2. **int8 signatures are an approved served config** for this loop (stepping stone; the existing
   ≥0.99-overlap-gated path).
3. **The comparison bar is radix-ON DSA** (production default), and **DS radix-on enablement plus
   radix-cache correctness validation with DS is in scope**.

## Pending decisions (resolve in `gen-plan` discussion)

- **AC numerics**: exact bars for capacity (≥64 bs vs ≥0.95× DSA tokens), TTFT ratio (1.10×?),
  aggregate ratio (0.95×?), tax guard (1.10×?), and whether AC-2/3 are judged on the directional
  ladder or only the closing AC-11 sweep.
- **Ship shape for absorbed-latent scoring**: replace the table outright (delete
  `token_label_table.py`/`token_label_write.py` when gates pass — preferred per doctrine: delete
  old paths when new paths work) vs a one-loop config flag with the fp16 table as fallback
  (costs capture variants + keeps 5.29 GB allocatable). Decide before implementation, not after.
- **Sequencing**: int8 stepping stone first (quick capacity floor, then superseded) vs straight
  to absorbed-latent (single big build, no throwaway work)? Task 0's probe table informs this.
- **Absorbed projection precision**: build `v_h` in fp32 and score in bf16-accumulate-fp32 (match
  current `_logical_score` semantics) — confirm against the fp8 scale-layout findings.
- **DSA radix-ON baseline timing**: frozen early (recommended: it is the bar everything is judged
  against) — confirm it's one directional ladder, not a locked sweep.
- **Loop budget**: full menu vs capacity-first (tasks 0+1+5 only, radix-on as a follow-on loop if
  the fixtures fight back).

--- Original Design Draft End ---
