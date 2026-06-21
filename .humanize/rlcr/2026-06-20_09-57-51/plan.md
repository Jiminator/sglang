# Loop 13 — Root-Cause the DS-vs-DSA Accuracy Degradation (Diagnosis Loop)

## Goal Description

This is a **diagnosis loop, not a fix loop**. Produce a **root-cause verdict with live-measured evidence** explaining why table-free Double Sparsity (DS) token selection on GLM-5.1-FP8 is far less accurate than the native DSA indexer. Measured on the dev clone (`/sgl-workspace/sglang`, branch `dev/double-sparsity-standalone`, the default editable `import sglang`), GSM8K on 8×H200 TP=8, greedy temp 0, completion API:

| GSM8K | DSA (native) | DS (current table-free) |
|---|---|---|
| 5-shot / 200 — dense (~763 tok, seq < top_k 2048) | 0.970 | 0.625 (serial control 0.700) |
| 24-shot / 150 — sparse (~4.2k tok, seq > 2048) | 0.953 | 0.000 |

The deliverable is a **verdict plus the reusable reference selector built to find it** — explicitly NOT a fix. The verdict must localize the cause to exactly one (or a ranked combination) of:

- **H0** — the channel-selection algorithm does not transfer to GLM MLA (the accuracy ceiling is bad even with an exact, slow, optimization-free implementation).
- **H1** — a loop 6–12 performance optimization corrupts selection (the ceiling is good; one toggle regresses it; identify the toggle, its GSM8K cost, and the commit).
- **H2** — the offline channel mask is bad for GLM-5.1 (a sub-branch of H0).
- **H3 (added during planning)** — selection is fine; the regression is **downstream of selection**: the `logical_to_physical` → `transform_index_page_table_decode` index adapter, the selected-index set, or KV-slot validity (`_slot_written`), feeding the **same** `flash_mla_with_kvcache` decode kernel DSA uses. Motivated by the dense regime: at seq ≈ 763 < top_k 2048 the selector keeps **all** live tokens (selection is a no-op there) yet still scores 0.625 — the signature of a downstream bug, not a scorer bug.

### Two assumptions in the draft that code inspection corrected (load-bearing)

1. The draft states "only the token-selector differs" and "the existing FlashMLA sparse decode path is fine." **Largely true for the served config, with one caveat.** The served config sets `--dsa-decode-backend flashmla_kv` for both `dsa` and `ds`, and lifted-budget is off in the DS JSON, so **served DS and DSA share the same `flash_mla_with_kvcache` decode kernel** (`dsa_backend.py` decode dispatch). `flash_mla_sparse_fwd` only runs under lifted-budget decode (off here). The genuinely DS-specific, uncontrolled downstream surface is therefore the **index adapter** (`logical_to_physical` + `transform_index_page_table_decode`), the **selected-index set**, and **slot validity** — not a divergent attention kernel. This is exactly what H3 must instrument.
2. The draft lists "approximate radix top-k" and "selector-width ladder / W=5120" as regression suspects. **Code inspection shows the radix top-k is exact** (bit-identical to the torch reference, `topk_kernel.py`) and selector-width buckets are claimed selection-neutral prefix windows. These should be **retired or confirmed by selected-index equivalence**, not assumed culprits — cheaply, before any GSM8K spend.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive tests (expected to PASS) and negative tests (expected to FAIL when the system is working correctly). All measurements run on the dev clone via the `development/loop13/` guarded scripts; one TP=8 server at a time.

- **AC-1: Pinned baseline reproduction.** DSA (native), a DSA-radix-OFF control, and current production DS are reproduced on a fixed GSM8K sample set, with the production regression reproduced (sanity that the harness/build are sound). Every arm records: git SHA, model snapshot path, mask `content_sha256`, full server args, CUDA-graph on/off, eval sample IDs and order, `max_tokens`, concurrency, and serial-vs-batched mode.
  - Positive Tests (PASS):
    - DSA reproduces ≈ 0.970 dense / ≈ 0.953 sparse; production DS reproduces ≈ 0.625 dense / ≈ 0.000 sparse (within run-to-run noise) on the pinned samples.
    - The DSA-radix-off control produces GSM8K bit-comparable to DSA-radix-on at temp 0 (confirming the draft's radix-output-neutrality claim instead of assuming it).
    - A metadata record exists for every arm with all fields above populated.
  - Negative Tests (FAIL when correct):
    - An arm whose `import sglang` does not resolve to the dev clone, or with `PYTHONPATH` pointing at the v2 clone, is refused by the `_env.sh` guard (rc=1) and produces no numbers.
    - An arm missing any pinned-metadata field is rejected as non-reproducible.
    - Production DS that does NOT reproduce the regression (e.g. sparse ≠ ~0.000) halts the loop — the build/harness is unsound and must be fixed before diagnosis.

- **AC-2: Cheap localization controls and the H3 fork.** Before building the reference selector, run the cheap controls that distinguish "scorer is the problem" from "downstream of selection is the problem," and record an explicit H3 fork.
  - **AC-2.1 — Dense forced-all downstream control.** A selector path that, for seq ≤ top_k, emits logical `[0 .. seq_len-1]`.
    - Positive: physical slots returned by `logical_to_physical` equal `req_to_token[req_pool, 0:seq_len]` exactly; no duplicates; no `-1` in the live region; all selected slots are written (`_slot_written`); adapter error count == 0. If dense GSM8K **recovers toward DSA** under forced-all, the scorer mis-selects even in dense (does not keep all live tokens) → scorer/selection branch.
    - Negative: if dense forced-all is **still bad** (≈ 0.625) with the physical-slot assertions all passing, selection is exonerated in dense → **H3 (downstream)**; the verdict points at the adapter / slot-validity / kernel-feeding path, not the scorer. A run where the physical slots do NOT equal `req_to_token[req_pool, 0:seq_len]` localizes the bug to the adapter itself.
  - **AC-2.2 — TP head-aggregation micro-test.** From captured per-head dot products (`score_capture` / `selection_capture`), compute and compare `local-max-per-rank then SUM-across-TP` vs `global-max-over-all-heads` vs `global-mean`.
    - Positive: the test states, with numbers, whether the served `head_agg="max"` + cross-TP SUM equals a global max over heads.
    - Negative: if local-max + SUM ≠ global-max on captured data, the shared-max-across-TP semantics is flagged as a concrete suspect for Phase B (and a candidate H1 mechanism).
  - **AC-2.3 — Selected-index equivalence for the contradicted suspects.** Confirm production radix top-k == `torch.topk` reference selection, and `selector_width_buckets=[5120]` vs `[]` produce identical selected indices, on captured score rows.
    - Positive: index sets are identical → suspects retired by inspection.
    - Negative: any index-set mismatch promotes that suspect to a measured Phase-B arm.
  - **AC-2.4 — Recall-oracle corroboration (not exoneration).** `recall_oracle` recall@2048 is recorded for dense and sparse as **corroborating** evidence. It is explicitly NOT a generic selected-index equivalence proof.

- **AC-3: Reference ("naive") selector served and proven faithful.** A performance-naive, algorithmically-faithful DS reference selector serves GLM-5.1-FP8 from the dev clone via a `serve.sh ref` mode (a new config field, e.g. `selector_impl="reference"`, reusing the same guard and `run_gsm8k.sh`), with all selection-side perf optimizations off (resident-fp8-latent dequant to fp32, exact full-width `torch.topk`, fp32 score-reduce, eager / `--disable-cuda-graph` allowed). Raw-dot uses the existing fp32 absorbed scoring; the served **cosine** variant uses a materialized per-head signature.
  - **AC-3.1 — Raw-dot reference equivalence.** The fp32 absorbed raw-dot reference (`absorbed_latent_score_logical`) is proven equal to an offline/blockwise materialized fp32 `K_label` score (selected-index equality @ 2048) on captured decode steps, establishing the raw-dot ceiling is trustworthy.
    - Positive: selected-index sets match @ 2048 on captured steps.
    - Negative: a mismatch means the absorbed identity is not being computed faithfully → fix the reference before trusting any ceiling number.
  - **AC-3.2 — Served cosine arm.** A served cosine reference selector (materialized per-head signature; normalization defined exactly per the Loop-7 lever — normalize after mask-channel gather) runs end-to-end and produces a real GSM8K number.
    - Positive: the cosine arm serves, DS is genuinely active on the sparse regime, and a GSM8K dense+sparse pair is recorded.
    - Negative: a cosine arm that cannot serve (e.g. config validation rejects it) is surfaced as a build gap, not silently dropped.
  - **AC-3.3 — DS-active invariants by regime.** On the **sparse** regime the reference reports DS genuinely active (`selected < total`, `dense_fallback == 0`). On the **dense** regime the reference reports `selected == seq_len` (all live tokens kept) — selection is correctly a no-op there.
    - Positive: sparse shows `selected < total`; dense shows `selected == seq_len`.
    - Negative: a dense arm reporting `selected < seq_len`, or a sparse arm reporting `dense_fallback != 0`, fails the invariant.
  - **AC-3.4 — Leak-free fp32.** Any GPU "fp32" reference disables TF32 (`torch.backends.cuda.matmul.allow_tf32 = False`, cuDNN likewise) or is explicitly labelled "GPU-fp32-with-TF32-risk"; pure-CPU references are exempt.

- **AC-4: Per-arm GSM8K evidence table.** GSM8K is measured for DSA, naive-DS (raw-dot), naive-DS (cosine), and production DS on the validated configs (5-shot/200 dense, 24-shot/150 sparse, temp 0, completion API), **serial and batched**. Each row records: dense score, sparse score, serial/batched mode, radix on/off, selector width used, score-reduce dtype/backend, head-aggregation mode, selected-vs-total summary, and a concretely-defined length-cap-garbage rate (count of invalid physical slots, unwritten slots, duplicate indices, and out-of-range lanes per layer/step).
  - Positive Tests (PASS):
    - Every arm × {serial, batched} cell is populated with both scores and all metadata columns.
    - The serial-vs-batched gap (the known 0.625 vs 0.700) is reported as an explicit axis, not averaged away.
  - Negative Tests (FAIL when correct):
    - A row with a GSM8K delta but no corroborating selected-index / recall figure for a sub-5-point gap is rejected (GSM8K n=150 binomial stderr ≈ 4 points → sub-5-point deltas require selected-index or recall corroboration, or repeats; the large 0.97-vs-0.00 and 0.97-vs-0.625 gaps are unambiguous on a single run).

- **AC-5: Decision gate recorded with the confirmed numeric threshold.** The ceiling-good-vs-bad gate is recorded with the user-confirmed threshold and the loop branches accordingly. **Threshold (user-confirmed): ceiling is GOOD iff naive-DS (best of raw-dot/cosine) sparse 24-shot GSM8K is within 5 points of DSA AND does not collapse (> 0), and dense is within 3 points of DSA; otherwise BAD.**
  - Positive Tests (PASS):
    - The recorded gate names the measured naive-DS sparse/dense scores, the DSA scores, the computed gaps, and the GOOD/BAD outcome.
    - GOOD routes to AC-6 (bisection); BAD routes to AC-7 (accuracy-ceiling knob sweep).
  - Negative Tests (FAIL when correct):
    - A "GOOD" verdict where naive-DS sparse collapsed (== 0) is rejected — collapse is BAD by definition regardless of the dense number.
    - A branch chosen without recording the numeric gap is rejected.

- **AC-6 (conditional — ceiling GOOD → H1): culprit isolated by single-variable bisection.** Walking from the reference toward the production path, exactly one variable changes per arm; GSM8K (dense + sparse) is measured at each step until accuracy drops. The first drop names the culprit (there may be more than one). Suspicion order, informed by Phase 1: (1) `head_agg` shared-max-over-TP semantics, (2) raw-dot vs cosine scorer, (3) fp8 absorbed-latent scoring vs materialized fp32 `K_label`, (4) bf16 vs fp32 score-reduce, (5) approximate vs exact radix top-k (likely retired by AC-2.3), (6) selector-width ladder / W=5120 (likely retired by AC-2.3). Prefer existing config toggles (`score_reduce_dtype`, `head_agg`, `selector_width_buckets`, `recall_oracle`) over git reverts; fall back to git-stepping loop6→loop12 commits where no toggle exists.
  - Positive Tests (PASS):
    - The culprit toggle is named with the GSM8K accuracy it costs and the responsible commit(s).
    - Each measured delta is corroborated by recall@2048 and/or selected-index/score-rank mismatch versus the reference.
  - Negative Tests (FAIL when correct):
    - An arm that changes more than one variable is rejected (not a clean bisection step).
    - A sub-5-point GSM8K delta asserted as the culprit without selected-index/recall corroboration is rejected.

- **AC-7 (conditional — ceiling BAD → H0/H2): accuracy-ceiling knob sweep.** When the ceiling is bad, characterize the true accuracy ceiling by tuning **every** knob toward accuracy (not speed) and mapping how accuracy responds. Includes a **no-mask ablation** (DS with no channel mask / full-channel signature) and, per the user's direction, a sensitivity study of accuracy as each knob moves toward its accuracy-favoring setting.
  - **AC-7.1 — No-mask ablation.** DS run with no channel mask (full signature) GSM8K is measured; if no-mask recovers accuracy toward DSA, the mask is the bottleneck (H2 confirmed); if no-mask is still bad, the algorithm itself does not transfer (H0).
  - **AC-7.2 — Knob-response sweep.** For each accuracy-relevant knob (scorer raw-dot vs cosine; `head_agg` max vs mean vs per-head oracle; `score_reduce_dtype` fp32; `selector_width_buckets` full; `top_k` larger; mask recalibration; `label_dim` variation), the resulting GSM8K is recorded and the accuracy trend as the knob moves toward accuracy-over-speed is reported (monotone improvement, saturation, or no effect).
  - **AC-7.3 — Per-head offline oracle.** Per-head selection is reported as an OFFLINE oracle upper bound (recall@2048 and would-be top-k from captured steps), because the shared-set `flash_mla_with_kvcache` kernel cannot serve a per-head token set — see DEC-1.
  - Positive Tests (PASS):
    - The no-mask number and the per-knob accuracy values are recorded, with an explicit statement of which knob (if any) moves accuracy and by how much.
    - The sweep states whether ANY accuracy-favoring configuration recovers naive-DS sparse to within 5 points of DSA (i.e. whether the ceiling is recoverable at all this loop).
  - Negative Tests (FAIL when correct):
    - A "the algorithm doesn't transfer (H0)" verdict asserted without the no-mask ablation having been run is rejected.
    - A knob-sweep arm that silently changes more than the one knob under study is rejected.

- **AC-8: Root-cause writeup.** A writeup in `development/loop13/` contains the per-arm GSM8K evidence table, the recall-oracle/selected-index corroboration, the verdict (H0 / H1 / H2 / H3, or a ranked combination), and a recommendation (research vs targeted fix) — explicitly NOT a fix.
  - Positive Tests (PASS):
    - The writeup names exactly one primary hypothesis (or a ranked set) and ties it to specific numeric evidence and negative controls.
    - For an H1 verdict it names the commit(s) and accuracy cost; for H3 it names the adapter/slot mechanism; for H0/H2 it cites the no-mask ablation and knob sweep.
  - Negative Tests (FAIL when correct):
    - A writeup that proposes or lands a code fix to the selection/adapter path is out of scope and rejected.
    - A verdict unsupported by a live GSM8K number (theory only) is rejected.

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)
A full diagnostic sweep on the dev clone: pinned baselines (DSA, DSA-radix-off, production DS) serial+batched; the Phase-1 cheap controls (forced-all dense with physical-slot assertions, TP head-aggregation micro-test, selected-index equivalence for radix/width, recall-oracle corroboration); a served reference selector with raw-dot (fp32 absorbed) and served cosine (materialized per-head signature) plus an offline materialized-`K_label` cross-check; the full per-arm GSM8K evidence table; the decision gate at the confirmed threshold; and **both** conditional branches as applicable — Phase-B single-variable bisection (GOOD) and the Phase-C accuracy-ceiling knob sweep with no-mask ablation and per-head offline oracle (BAD). All wrapped in a root-cause writeup with the verdict and a research-vs-fix recommendation. No selection/adapter code fix is landed this loop.

### Lower Bound (Minimum Acceptable Scope)
Pinned reproduction of DSA and production DS (AC-1); the dense forced-all downstream control with physical-slot assertions and the recall-oracle/selected-index cheap controls sufficient to decide the H3 fork (AC-2); the light `reference_absorbed_fp32` raw-dot reference served and proven faithful (AC-3.1, AC-3.3, AC-3.4); the per-arm GSM8K table for DSA / naive-raw / production with corroboration (AC-4); the decision gate at the confirmed threshold (AC-5); and a writeup with a defensible verdict and recommendation (AC-8). The served cosine arm (AC-3.2), the full knob sweep (AC-7.2), and git-stepping bisection beyond config toggles are the first things to trim if blocked — but the no-mask ablation (AC-7.1) is retained whenever the BAD branch is taken.

### Allowed Choices
- **Can use:** the existing `development/loop13/` guarded scripts (`_env.sh`, `serve.sh`, `probe_ds_active.sh`, `run_gsm8k.sh`, `teardown.sh`), extended with a `ref` mode; existing config toggles (`score_reduce_dtype`, `head_agg`, `selector_width_buckets`, `recall_oracle`, `score_capture`, `selection_capture`); existing fp32 reference functions (`absorbed_latent_score`, `absorbed_latent_score_logical`, `select_topk_sequence_order`); the `_select_topk_indices` seam and `logical_to_physical` adapter for instrumentation; pure-torch / eager / `--disable-cuda-graph` for the reference (perf is irrelevant); git-stepping loop6→loop12 commits where no toggle exists.
- **Cannot use:** the v2 clone (`/sgl-workspace/double-sparisty-v2/sglang`) or any `PYTHONPATH` override; `PYTORCH_CUDA_ALLOC_CONF=expandable_segments` for serving; the chat API for GSM8K (must use `--api completion`); blanket GPU process kills / `pkill -f` parent-matching; landing any selection/adapter code fix this loop; running more than one TP=8 server at a time.

> **Note on determinism:** the harness is largely fixed by the draft (model, base, dsa backends, page 64, fp8 KV, seed 42, completion API, the two GSM8K configs). Those are fixed constraints, not choices. The genuine choices are the reference-selector implementation depth and the breadth of the knob sweep, bounded above and below as stated.

## Feasibility Hints and Suggestions

> Reference and understanding only — conceptual, not prescriptive.

### Conceptual Approach
1. **Pin & reproduce → verify, don't assume.** Boot DSA, DSA-radix-off, and production DS via `serve.sh`; record full metadata; reproduce the regression. Confirm radix output-neutrality at temp 0 by direct comparison.
2. **Cheapest decisive experiment first.** The dense regime is the lever: since the selector keeps all ~763 < 2048 tokens, dense selection is a no-op, so a forced-all control with physical-slot assertions (`logical_to_physical` output == `req_to_token[req_pool, 0:seq_len]`) cleanly separates scorer from downstream. Run the TP head-aggregation micro-test and selected-index equivalence checks from captured scores — these are minutes, not GSM8K runs.
3. **Build the light reference, prove it faithful.** Branch at `_select_topk_indices`; route raw-dot through the existing fp32 `absorbed_latent_score_logical`; prove it equals an offline materialized fp32 `K_label` (the absorbed identity is exact algebra in fp32). Add the served cosine path on a materialized per-head signature.
4. **Measure, gate, branch.** Fill the evidence table; apply the 5-point/no-collapse gate; bisect (GOOD) or knob-sweep + no-mask ablation (BAD).
5. **Write the verdict.** One primary hypothesis (or ranked set), tied to numbers and negative controls; recommend research vs targeted fix; land no fix.

### Relevant References
- `python/sglang/srt/models/deepseek_v2.py` — `_select_topk_indices` (the selector seam) and the decode handoff to `logical_to_physical` / `transform_index_page_table_decode`.
- `python/sglang/srt/layers/attention/dsa_backend.py` — decode dispatch (`flash_mla_with_kvcache` for the served config; `flash_mla_sparse_fwd` only under lifted-budget); per-request DS summary (`selected`/`total`/`dense_fallback`).
- `python/sglang/srt/layers/attention/double_sparsity/config.py` — config knobs and the `scorer_norm="off"` hard-lock with its absorbed-identity rationale.
- `python/sglang/srt/layers/attention/double_sparsity/absorbed_latent.py` — fp32 reference scorers (`absorbed_latent_score`, `absorbed_latent_score_logical`).
- `python/sglang/srt/layers/attention/double_sparsity/selection_kernel.py` — `reduce_token_scores` (the bf16/fp32 cross-TP reduce), `select_topk_sequence_order` (exact torch top-k reference).
- `python/sglang/srt/layers/attention/double_sparsity/topk_kernel.py` — the exact radix top-k (suspect to retire by equivalence).
- `python/sglang/srt/layers/attention/double_sparsity/page_table_adapter.py` — `logical_to_physical` (the H3 adapter; `-1`-padding handling, error count).
- `python/sglang/srt/layers/attention/double_sparsity/selection_recall_oracle.py`, `score_capture.py`, `selection_capture.py` — corroboration instruments.
- `development/loop13/{_env.sh,serve.sh,probe_ds_active.sh,run_gsm8k.sh,teardown.sh}` — the guarded harness to extend with `ref` mode.
- `development/loop12/gsm8k_evidence/`, `development/loop12/RUN_AND_EVALUATE.md`, `development/loop12/V2_PERFORMANCE.md` — the measured regression and runbook.
- `development/past_implementations/study/{00-survey,06-proposed-architecture,07-mvp-proposed-architecture,08-current-system-architecture}.md` — the paper-faithful algorithm (the ceiling target).
- Optimization history (bisection list): `git log --oneline -- python/sglang/srt/layers/attention/double_sparsity/` — Loop 6 `84d3410b9` (int8 table), `ece26eb52` (over-scan), `2715b7382` (tie-break); Loop 7 `599d7cc99`/`e2674f4f4` (cosine, recall 5%→40%); Loop 8 `4e49d8416`/`43709a761` (GLM-5.1 port + calibration); Loop 9 `c877d7fa1` (bf16 reduce), `859c8ee2c` (approx radix top-k); Loop 10 `6c92240b9`/`9956c240e` (selector-width ladder, W=5120); Loop 11 `776f3e613`→`01e3ff238` (table-free absorbed-latent, delete TokenLabelTable).

## Dependencies and Sequence

### Milestones
1. **Pin & reproduce baselines (AC-1).**
   - Phase 0: boot DSA, DSA-radix-off, production DS via the guarded scripts; record full per-arm metadata; reproduce the regression serial + batched.
2. **Cheap localization & the H3 fork (AC-2).** Depends on Milestone 1.
   - Phase 1: forced-all dense control with physical-slot assertions; TP head-aggregation micro-test; selected-index equivalence for radix/width; recall-oracle corroboration. Records the H3 fork.
3. **Reference selector & evidence table (AC-3, AC-4).** Depends on Milestone 1; informed by Milestone 2.
   - Phase A: extend `serve.sh` with a `ref` mode and a `selector_impl="reference"` config field; serve raw-dot (fp32 absorbed) and served cosine (materialized signature); prove raw-dot faithful; measure DSA / naive-raw / naive-cosine / production serial+batched; build the evidence table.
4. **Decision gate (AC-5).** Depends on Milestones 2–3.
   - Apply the 5-point / no-collapse threshold; branch.
5. **Conditional branch.** Depends on Milestone 4.
   - Phase B (GOOD → H1): single-variable bisection (AC-6).
   - Phase C (BAD → H0/H2): no-mask ablation + accuracy-ceiling knob sweep + per-head offline oracle (AC-7).
6. **Verdict writeup (AC-8).** Depends on the chosen branch.

> Dependencies are logical, not temporal: the cheap controls (Milestone 2) can pre-empt the reference build — if Phase 1 convicts H3, the reference is still built (DEC-2 resolution) but its ceiling number is caveated as confounded until the downstream bug is fixed in a follow-up loop.

## Task Breakdown

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Boot DSA / DSA-radix-off / production DS via guarded scripts; capture full per-arm metadata; reproduce regression serial+batched | AC-1 | coding | - |
| task2 | Implement & run forced-all dense downstream control; assert physical slots == `req_to_token[req_pool, 0:seq_len]`, no dup/`-1`/unwritten, adapter errors == 0 | AC-2.1 | coding | task1 |
| task3 | TP head-aggregation micro-test from captured per-head dots (local-max+SUM vs global-max vs global-mean) | AC-2.2 | coding | task1 |
| task4 | Selected-index equivalence: radix top-k vs `torch.topk`; width `[5120]` vs `[]`; record recall-oracle@2048 | AC-2.3, AC-2.4 | coding | task1 |
| task5 | Adversarially review the Phase-1 control results: is the H3 fork sound, or could a control be subtly wrong (false ceiling)? | AC-2 | analyze | task2, task3, task4 |
| task6 | Extend `serve.sh` with `ref` mode + `selector_impl="reference"`; route raw-dot through fp32 `absorbed_latent_score_logical`; eager / no-cuda-graph; disable TF32 | AC-3.1, AC-3.4 | coding | task1 |
| task7 | Prove raw-dot reference == offline materialized fp32 `K_label` (selected-index equality @2048) on captured steps | AC-3.1 | coding | task6 |
| task8 | Build served cosine reference (materialized per-head signature; Loop-7 normalization); verify DS-active invariants by regime | AC-3.2, AC-3.3 | coding | task6 |
| task9 | Measure GSM8K for DSA / naive-raw / naive-cosine / production serial+batched; build the per-arm evidence table with garbage-rate columns | AC-4 | coding | task6, task8 |
| task10 | Apply the decision gate (sparse within 5 pts + no collapse; dense within 3 pts); record outcome and branch | AC-5 | analyze | task9 |
| task11 | (GOOD branch) Single-variable bisection across head_agg/scorer/fp8/reduce/topk/width; corroborate each delta with recall/selected-index; name culprit + commit + cost | AC-6 | coding | task10 |
| task12 | (BAD branch) No-mask ablation + accuracy-ceiling knob sweep + per-head offline oracle; report accuracy response per knob | AC-7 | coding | task10 |
| task13 | Adversarially verify the chosen verdict against the evidence (could H3 masquerade as H0? is a sub-5-pt delta real?) | AC-8 | analyze | task11, task12 |
| task14 | Write the root-cause writeup in `development/loop13/` (evidence table, verdict, research-vs-fix recommendation; no fix) | AC-8 | coding | task13 |

## Claude-Codex Deliberation

### Agreements
- Phase ordering: reproduce/pin first, run cheap localization before building a large reference path, then bisect only if the ceiling is good.
- `_select_topk_indices` is the right selector seam; `logical_to_physical` is the right selection→attention handoff to instrument.
- **H3 is a necessary hypothesis** — the dense `seq < top_k` degradation is downstream-shaped, not scorer-shaped.
- Elevating the `head_agg="max"` cross-TP SUM semantics is correct (local per-rank max then cross-rank SUM is not a global max over heads).
- Retire the radix-top-k and selector-width suspicions by selected-index equivalence, not GSM8K.
- The served DS config (lifted-budget off) and DSA share the `flash_mla_with_kvcache` decode kernel; H3 centers on `logical_to_physical` → `transform_index_page_table_decode` and slot validity, not a divergent attention kernel.

### Resolved Disagreements
- **Primary reference implementation.** Codex: a materialized fp32 `K_label` served reference is memory-hostile at GLM scale; use the existing fp32 absorbed scoring for raw-dot and materialize signatures only offline/blockwise (and for cosine). Claude (draft) leaned to a standalone materialized reference. **Resolution:** raw-dot ceiling uses the cheap fp32 `absorbed_latent_score_logical` (proven equal to materialized fp32 `K_label` since the absorbed identity is exact algebra in fp32); a materialized per-head signature is built only for the **served cosine** arm (which the user chose) and as an offline cross-check. Rationale: same number, far less code/memory.
- **Recall-oracle's role.** Codex: `recall_oracle` is a NIAH recall diagnostic, not a generic selected-index equivalence checker; it cannot by itself exonerate the scorer. **Resolution:** recall-oracle is corroborating evidence (AC-2.4); the decisive cheap control is the forced-all dense test with physical-slot assertions (AC-2.1).
- **DS-active invariant in the dense regime.** Codex: "selected < total" conflicts with the dense no-op test. **Resolution:** AC-3.3 splits the invariant by regime — sparse asserts `selected < total`; dense asserts `selected == seq_len`.
- **Cosine's verdict category.** Codex: a "cosine wins" result means the table-free raw-dot design constraint is bad for GLM (H0/H2-adjacent), not a perf regression (H1). **Resolution:** the plan classifies a cosine-only recovery as a design-constraint verdict (H0/H2 family), and the user elected to measure cosine as a full served GSM8K arm so the number is real.
- **Contradicted suspects.** Draft listed approximate radix top-k and selector-width as regression suspects; code shows both are exact/selection-neutral. **Resolution:** retire by selected-index equivalence (AC-2.3) before any GSM8K spend; promote to a measured arm only on mismatch.
- **fp32 leakage.** Codex: a GPU "fp32" reference with TF32 enabled is not exact. **Resolution:** AC-3.4 requires TF32 disabled or explicit labelling.

### Convergence Status
- Final Status: `converged` (one second-pass Codex review round; all `REQUIRED_CHANGES` accepted and code-verified; remaining open items are owner decisions captured below, not Claude/Codex disagreements).

## Pending User Decisions

- **DEC-1: Per-head selection — served arm or offline oracle only?**
  - Claude Position: offline oracle upper bound only — the shared-set `flash_mla_with_kvcache` decode kernel consumes one token set across heads, so a per-head token set is not serveable without a kernel/API change; report per-head as recall@2048 / would-be top-k from captured steps (AC-7.3).
  - Codex Position: agree — restrict served arms to shared-token-set selectors; treat per-head as an offline oracle, not a served GSM8K arm.
  - Tradeoff Summary: a served per-head arm would give a true per-head accuracy number but requires substantial kernel/API work outside a diagnosis loop; the offline oracle answers "would per-head help?" cheaply. The user asked to "tune every knob," so per-head is included — the open point is solely served-vs-offline.
  - Decision Status: `PENDING` (Claude recommends offline oracle).

- **DEC-2: If Phase-1 convicts H3, is the served reference selector still required for the deliverable?**
  - Claude Position: always build the light `reference_absorbed_fp32` selector (it is the reusable deliverable artifact and is cheap), but caveat its GSM8K ceiling as confounded until the downstream H3 bug is fixed in a follow-up loop.
  - Codex Position: decide this explicitly; either still build the lighter absorbed reference or amend the deliverable.
  - Tradeoff Summary: building it always preserves the reusable artifact at low cost; skipping it tightens scope but weakens the deliverable.
  - Decision Status: **Always build the light reference** (user-confirmed during planning).

### Decisions resolved during planning (recorded for traceability)
- **Ceiling threshold (was DEC):** GOOD iff naive-DS sparse within 5 points of DSA AND no collapse (> 0); dense within 3 points (user-confirmed). Drives AC-5.
- **BAD-branch scope (was DEC):** do NOT merely stop at the verdict — run the accuracy-ceiling knob sweep: a no-mask ablation plus tuning every knob toward accuracy (not speed) and characterizing how accuracy responds as knobs move (user-directed). Drives AC-7.
- **Cosine variant (was DEC):** full served GSM8K arm, not an offline oracle (user-confirmed). Drives AC-3.2; revives the materialized per-head signature path for cosine only.
- **Significance convention:** the large 0.97-vs-0.00 / 0.97-vs-0.625 gaps are unambiguous on a single run; Phase-B/knob-sweep deltas under ~5 points (GSM8K n=150 binomial stderr ≈ 4 points) require selected-index / recall corroboration or repeats. Drives AC-4 / AC-6 negative tests.

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-workflow terminology such as "AC-", "Milestone", "Phase", "Step", "H0/H1/H2/H3", or similar markers. Those belong to this plan document only.
- Use descriptive, domain-appropriate names in code (e.g. `reference_absorbed_fp32_selector`, `forced_all_dense_control`, `head_agg_tp_aggregation_check`), not plan identifiers.
- Follow the Double Sparsity speculative-naming conventions when adding identifiers in the selection path (see `.claude/rules/speculative-naming.md`).
- This loop lands **no fix** to the selection/adapter path; all new code is diagnostic (reference selector, controls, instrumentation, evidence harness).

--- Original Design Draft Start ---

# Loop 13 Draft — Find the cause of the DS-vs-DSA accuracy degradation

> Written 2026-06-20. This is a **diagnosis loop, not a feature/fix loop.** The goal is to
> explain *why* table-free Double Sparsity (DS) on GLM-5.1-FP8 is far less accurate than the native
> DSA indexer, and to localize the cause to either (a) the algorithm itself not transferring to
> GLM/MLA, or (b) a specific performance optimization we added during loops 6–12. We are not
> required to fix it this loop — the deliverable is a **root cause with evidence**, plus the reusable
> reference selector built to find it.
>
> Run everything on the **dev clone** `/sgl-workspace/sglang` (branch `dev/double-sparsity-standalone`)
> — the default editable `import sglang`. This is the implementation under investigation. (The
> `double-sparsity-v2` clone is a separate, refactored codebase; out of scope here.)
>
> Feed this through `gen-plan` once scope is confirmed.

---

## What this is (and is NOT) — read first

**This is a measurement-driven root-cause investigation.** No new selection algorithm, no perf
work, no SLO chasing. We build the *simplest correct* version of DS to establish an accuracy
ceiling, then bisect our own optimization history against that ceiling. Every claim must be a
GSM8K number from a live 8×H200 run, not a theory.

**The deliverable is a verdict, not a patch.** Either "the channel-selection algorithm does not
transfer to GLM-5.1 MLA (ceiling is bad — recovering accuracy needs research)" or "the algorithm is
fine (ceiling ≈ DSA); optimization X regressed it (here is the commit and the accuracy it costs)."
Fixing X is a *follow-up* loop.

---

## Background — the measured regression (Loop-12 session, 2026-06-20)

GSM8K, GLM-5.1-FP8, 8×H200 TP=8, same base/backend, only the token-selector differs. Greedy
(temp 0), completion API (no thinking-mode, no shot/question leakage). DS ran `--disable-radix-cache`
(the dev clone gates DS+radix; radix is output-neutral at temp 0). Both regimes single-request
reproducible.

| GSM8K (dev clone) | DSA (native) | DS (current table-free) |
|---|---|---|
| 5-shot / 200 — **dense** (~763 tok, seq < top_k 2048) | 0.970 | **0.625** (serial control 0.700) |
| 24-shot / 150 — **sparse** (~4.2k tok, seq > 2048) | 0.953 | **0.000** |

Failure mode (verified single-request, `finish_reason=length`): at long context DS **degenerates
into garbage** (`"...the the8�a00 the the … RRRRRRR0R0QRQRQ…"`) and runs to the token cap. DSA scored
0.953 on the *exact same* 24-shot prompts, so the prompt/model are fine — only the DS selector
corrupts generation. The server returns 200 throughout: the corruption is **silent** (the selector
picks a bad top-2048 set → the model attends to wrong KV). DS is also degraded in the **dense**
regime (~0.63–0.70 vs 0.97) where it selects *all* tokens and should equal plain attention — so the
problem is not purely the sparse cap.

Evidence: `development/loop12/gsm8k_evidence/` (the `*_short.out` / `*_long.out` run logs and the
single-request probe outputs).

---

## Hypotheses to discriminate (the whole point of the loop)

**H0 — the algorithm doesn't transfer.** Channel-importance selection (offline `mean|Q·K|` heavy
channels → approximate `Q_label·K_labelᵀ` → top-k → full attention) is fundamentally weaker than
GLM's *learned* DSA indexer on MLA. If true, even a perfect, slow, exact implementation is far below
DSA, and perf is irrelevant.

**H1 — a perf optimization regressed it.** The exact algorithm matches DSA, but one of the speed
optimizations we layered in loops 6–12 corrupts the selection. Leading suspects, with the history
that makes each plausible:

- **Raw-dot scoring (`scorer_norm` locked to `"off"`).** Loop 7 measured that a **cosine scorer
  took 16K NIAH recall from 5% → 40%** (`e2674f4f4`, `599d7cc99`). The table-free rewrite (Loop 11)
  then **removed cosine**: `config.py` now hard-locks `scorer_norm="off"` because the absorbed-latent
  identity `score = max_h v_h·c_kv` only holds for the raw dot — "direction-only norms would operate
  on a materialized per-head signature the selector never builds." So going table-free may have
  *reverted to the known-bad scorer*. **Top suspect.**
- **Table-free absorbed-latent fp8 scoring (Loop 11, `01e3ff238` deletes `TokenLabelTable`).** Scores
  are computed from the resident fp8 MLA latent with per-128-block dequant in-register, instead of a
  materialized signature. fp8 precision + the absorbed approximation could wreck score ordering.
- **bf16 score-reduce across TP (Loop 9, `c877d7fa1`).** The per-head scores are all-reduced in bf16;
  precision loss in the reduction can flip the top-k.
- **Approximate sequence-aware radix top-k replacing exact `torch.topk` (Loop 9, `859c8ee2c`).** A
  blocked/deterministic top-k that may not return the exact top-2048.
- **Selector-width ladder / compact W=5120 (Loop 10, `9956c240e`, `6c92240b9`).** Scores only a
  covering width; a 24-shot prompt (~4.2k) sits under the 5120 bucket, but the bucketing/keying logic
  is worth ruling out.
- **int8-symmetric signature compaction (Loop 6, `84d3410b9`).** Likely gone post-table-free, but
  confirm it isn't on any path.
- **Cross-head `head_agg="max"` shared selection.** MLA forces one token set shared across heads
  (the paper selects per-head). This may itself cost accuracy independent of the above.

**H2 — the offline channel mask is bad for GLM-5.1.** The calibrated mask (`mean|Q·K|` on the noPE
reconstruction) may simply not capture GLM's important channels. This sits under the H0 branch — the
reference selector reuses the existing mask, so if the ceiling is bad, mask quality is the next thing
to test (recalibrate, or vary `label_dim`).

---

## Execution harness — clone safety (do NOT launch the v2 clone)

**The hazard, stated plainly.** The default `import sglang` (the editable install) is the **dev clone**
`/sgl-workspace/sglang`. The **v2 clone** `/sgl-workspace/double-sparisty-v2/sglang` is a *different,
refactored* DS codebase. In the Loop-12 session we accidentally served the v2 clone by passing
`PYTHONPATH=$V2/python`, which silently invalidated the first DSA/DS numbers. **Every server in this
loop must run from the dev clone, with no `PYTHONPATH` override.**

**Ready-made scripts (shipped in `development/loop13/`, guard-enforced).** Use these; do not hand-roll
launch commands. Run `serve.sh` / `teardown.sh` **backgrounded** (they poll with `sleep`).

| Script | Purpose |
|---|---|
| `_env.sh` | Sourced guard + env. **Refuses to proceed** unless `import sglang` resolves to the dev clone and `PYTHONPATH` is clean; also blocks `expandable_segments`. Exports `MODEL`/`MASK`/`HOST`/`PORT`/`EVID`/`PIDFILE`. |
| `serve.sh <dsa\|ds>` | Boots GLM-5.1-FP8 from the dev clone (**no `PYTHONPATH`**), writes `$PIDFILE`, waits for `/health`. `ds` adds `--disable-radix-cache` + the dev-ABI DS config. |
| `probe_ds_active.sh` | Long-context request → asserts `selected < total`, `dense_fallback == 0` (DS genuinely active). |
| `run_gsm8k.sh <label>` | Dense (5-shot/200) + sparse (24-shot/150) GSM8K via the dev clone's `run_eval --api completion`. |
| `teardown.sh` | Kills only `$PIDFILE`, waits all 8 GPUs to ~0 MiB. Never blanket-kills. |

The guard is the load-bearing part — it has been verified to **stop** (FATAL, rc=1) the instant
`PYTHONPATH` points at the v2 clone, and to pass only when `import sglang` is the dev clone.

**Canonical launch commands** (what `serve.sh` runs — note **no `PYTHONPATH`**):

```bash
# DSA (native indexer, DS off) — the accuracy target
python3 -m sglang.launch_server --model-path "$MODEL" --host 127.0.0.1 --port 30000 \
  --tp-size 8 --kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.8 \
  --max-running-requests 64 --cuda-graph-max-bs 64 --page-size 64 \
  --dsa-prefill-backend flashmla_kv --dsa-decode-backend flashmla_kv \
  --disable-overlap-schedule --disable-piecewise-cuda-graph \
  --random-seed 42 --trust-remote-code

# DS (current table-free) — add radix-off + the dev-clone-ABI config (lifted fields accepted)
#   ...same flags as above, PLUS:
#   --disable-radix-cache --enable-double-sparsity \
#   --double-sparsity-config '{"top_k":2048,"page_size":64,"channel_mask_path":"'"$MASK"'","device_buffer_size":4096,"scorer_norm":"off","head_agg":"max","anchor_mode":"off","anchor_budget":0,"enable_lifted_budget_decode":false,"lifted_budget_top_k":0}'
```

**Standard per-arm sequence** (one TP=8 server at a time):

```bash
cd /sgl-workspace/sglang/development/loop13
bash serve.sh dsa            # (backgrounded) boot + wait READY
bash run_gsm8k.sh dsa        # (backgrounded) dense + sparse
bash teardown.sh             # (backgrounded) kill + wait GPU idle
# then repeat with `ds` (and, once built, the reference-selector arm)
bash serve.sh ds && bash probe_ds_active.sh && bash run_gsm8k.sh ds && bash teardown.sh
```

> The **reference (naive) selector** arm (Phase A) does not exist yet — it is what the loop builds.
> `serve.sh` must be extended with a `ref` mode (a config flag / env that selects the naive path)
> once that selector lands, reusing the same guard and the same `run_gsm8k.sh`.

---

## Approach

### Phase A — Establish the accuracy ceiling (the naive-but-correct port)

Build the **most performance-naive, algorithmically-faithful** DS selector for GLM-5.1-FP8 and
measure its GSM8K accuracy. "Naive" = strip *every* perf optimization at once; correctness over speed
(eager, `--disable-cuda-graph` allowed, pure-torch selector acceptable — it can be 100× slow, it only
runs the eval). Reference the traditional algorithm in
`development/past_implementations/study/{00-survey,06-proposed-architecture,07-mvp-proposed-architecture}.md`
(repo A = the paper-faithful `DoubleSparse`).

The reference selector, per decode step:
1. **Reuse the existing offline channel mask** (the `mean|Q·K|` heavy-channel calibration is the
   paper-faithful part and is already done — `/cluster-storage/models/glm51-fp8-channel-mask-loop12.safetensors`).
   Do not re-derive it in Phase A.
2. Materialize a **real per-head `K_label`** by reconstructing K-noPE from the latent in **bf16/fp32**
   (no fp8 dequant-in-register) and gathering the mask channels → `[S, H, r]`.
3. Build `Q_label` (query projected onto the same heavy channels).
4. Score **exactly** in fp32: `Q_label · K_labelᵀ` per head. Test **both** raw-dot and **cosine**
   normalization (the Loop-7 lever — cosine needs the materialized per-head signature this path has,
   which table-free lacks).
5. **Exact full-width `torch.topk`** (no radix approximation, no selector-width bucketing, no bf16
   reduce — reduce scores in fp32).
6. Full attention over the selected indices (the existing FlashMLA sparse decode path is fine — it's
   the *selection*, not the attention, under investigation).

Then measure, on the **same** GSM8K configs already validated this session
(`run_eval --eval-name gsm8k --api completion`, temp 0, **5-shot/200 dense** + **24-shot/150 sparse**,
plus a serial control), three arms on the same server build:
- **DSA** (native indexer) — the target.
- **Naive-DS** (reference selector, raw-dot) and **Naive-DS (cosine)**.
- **Current production DS** (the table-free path) — to confirm the regression reproduces.

**Decision gate (the loop's fork):**
- **Ceiling is BAD** (naive-DS, best of raw/cosine, still far below DSA — e.g. long-context still
  collapses): conclude **H0/H2** — the algorithm (or the mask) doesn't transfer. Recovering accuracy
  is a research problem, not an optimization rollback. Document, optionally probe mask quality
  (recalibrate / vary `label_dim` / per-head vs shared selection), and **stop** — do not chase perf.
- **Ceiling is GOOD** (naive-DS ≈ DSA, especially long-context > 0): conclude **H1** — a perf
  optimization regressed it. Proceed to Phase B.

### Phase B — Bisect the optimization history (only if ceiling is good)

Walk forward from the naive selector toward the current production path, **re-enabling one
optimization at a time**, measuring GSM8K (dense + sparse) at each step, until the accuracy drops.
The first toggle that drops it is the culprit (there may be more than one). Order by suspicion:

1. raw-dot vs cosine scorer (if cosine was the ceiling-maker, this alone may explain it),
2. fp8 absorbed-latent scoring vs materialized bf16 `K_label`,
3. bf16 vs fp32 score-reduce,
4. approximate radix top-k vs exact `torch.topk`,
5. selector-width ladder / W=5120 bucketing,
6. `head_agg` shared-vs-per-head.

Prefer **config/flag toggles** where they already exist (`scorer_norm`, `head_agg`,
`selector_width_buckets`, `score_reduce_dtype`, the `recall_oracle` / `score_capture` diagnostics in
`config.py`) over reverting commits; fall back to `git`-stepping the loop commits
(loop6→loop12, list in `git log -- python/sglang/srt/layers/attention/double_sparsity/`) where no
toggle exists. Reuse `selection_recall_oracle.py` and the score-capture instruments as secondary
diagnostics (recall@2048 and score-flip dumps) to corroborate each GSM8K delta.

---

## Acceptance criteria (rough — gen-plan will formalize as AC-X)

- **AC-1** A reference ("naive") DS selector serves GLM-5.1-FP8 with all perf optimizations off
  (materialized fp32 `K_label`, exact full-width top-k, fp32 reduce, eager OK), DS genuinely active at
  long context (`selected < total`, `dense_fallback == 0`), both raw-dot and cosine selectable.
- **AC-2** GSM8K measured for DSA, naive-DS (raw + cosine), and current production DS on the validated
  configs (5-shot/200 dense, 24-shot/150 sparse, temp 0, completion API), with the production
  regression reproduced (sanity that the harness/build are sound).
- **AC-3** The decision gate is recorded with an explicit numeric threshold for "ceiling good vs bad"
  (e.g. naive-DS long-context within N points of DSA), and the loop branches accordingly.
- **AC-4** (conditional, ceiling good) The optimization that introduces the regression is identified
  by toggle/bisection, with the GSM8K accuracy each optimization costs and the commit(s) responsible.
- **AC-5** A root-cause writeup in `development/loop13/` with the evidence table (per-arm GSM8K +
  recall-oracle corroboration), the verdict (H0/H1/H2), and a recommendation (research vs targeted
  fix) — explicitly *not* a fix.

---

## Constraints (carry forward, do not relitigate)

- **One TP=8 server at a time.** Tear down, wait for all 8 GPUs to return to ~0 MiB before the next
  boot. Track the launched PID; never blanket-kill GPU PIDs or `pkill -f` a parent-matching pattern.
- **Never set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments` for serving** (breaks custom all-reduce
  at TP=8; calibration-only, separate process).
- **Run the dev clone** (`/sgl-workspace/sglang`, default `import sglang`), **never** the v2 clone, and
  **never** set `PYTHONPATH` to a clone. Always launch via the `development/loop13/` scripts whose
  `_env.sh` guard enforces this (see "Execution harness — clone safety"). Hand-rolled
  `python -m sglang.launch_server` is allowed only after the same guard passes.
- **DS+radix:** the dev clone gates it; use `--disable-radix-cache` (output-neutral at temp 0). Keep
  DSA and DS otherwise byte-identical (model, base, dsa backends, page 64, fp8 KV, seed 42).
- **GSM8K harness is settled:** `python -m sglang.test.run_eval --eval-name gsm8k --api completion`
  (completion path bypasses GLM's thinking template and uses leakage-free shot/question split). Do
  not switch to the chat path.
- Perf is irrelevant this loop — do not optimize the reference selector; slow-but-correct is the point.

---

## References

- **Measured regression + harness:** `development/loop12/gsm8k_evidence/`,
  `development/loop12/RUN_AND_EVALUATE.md`, `development/loop12/V2_PERFORMANCE.md`.
- **Traditional algorithm (the ceiling target):**
  `development/past_implementations/study/00-survey.md` (§1 paper concept, §4 vocabulary, §5 the three
  "double" definitions), `06-proposed-architecture.md`, `07-mvp-proposed-architecture.md`,
  `08-current-system-architecture.md` (as-built; §3 int8 table, §5 over-scan + kernel ABI, §6 recall).
- **Optimization history (the bisection list):** `git log --oneline -- python/sglang/srt/layers/attention/double_sparsity/`
  — Loop 6 `84d3410b9` (int8 table), `ece26eb52` (over-scan), `2715b7382` (tie-break); Loop 7
  `599d7cc99`/`e2674f4f4` (cosine scorer, recall 5%→40%); Loop 8 `4e49d8416`/`43709a761` (GLM-5.1 port
  + calibration); Loop 9 `c877d7fa1` (bf16 reduce), `859c8ee2c` (approx radix top-k); Loop 10
  `6c92240b9`/`9956c240e` (selector-width ladder, W=5120); Loop 11 `776f3e613`→`01e3ff238` (table-free
  absorbed-latent, delete TokenLabelTable).
- **Current selector code:** `selection_kernel.py`, `absorbed_latent.py`, `absorbed_latent_kernel.py`,
  `config.py` (the `scorer_norm`/`head_agg`/`selector_width_buckets` knobs + diagnostics),
  `selection_recall_oracle.py` (recall corroboration).

## Open decisions for the user / gen-plan

1. **How naive is naive?** Purest is a stand-alone pure-torch reference selector (most trustworthy,
   slowest); cheaper is reusing the absorbed path in bf16 with exact top-k + cosine. Recommend the
   pure-torch reference for AC-1 so the ceiling is unimpeachable.
2. **Ceiling threshold.** What long-context GSM8K gap from DSA counts as "algorithm transfers"
   (e.g. within 5 points) vs "doesn't transfer"?
3. **Scope of the fork's BAD branch.** If the ceiling is bad, do we spend this loop probing mask
   quality (recalibrate / `label_dim` / per-head selection), or stop at the verdict and open a
   research loop?
4. **Is per-head selection feasible on MLA at all,** or is shared-across-heads a hard constraint we
   must accept (and therefore part of the ceiling, not a bug)?

--- Original Design Draft End ---
