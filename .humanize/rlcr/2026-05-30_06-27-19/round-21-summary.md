# Round 21 Summary — AC-5 full-context evidence rebuilt to the R9 fail-closed standard

## Mainline objective (round contract)
Fix Codex R20 blocking issue 1: the R20 AC-5 verifier stored only a DERIVED `per_req_gen_tps` and re-checked
it, so mutating that array to 100.0 still "passed" the strict TPS axis (tamperable; below the R9 bar). Rebuild
the full-context AC-5 evidence to the R9 standard from the existing raw JSONLs (no re-run).

## What landed (commit `991666b58`, data-only)
1. **Exact committed source** (`ac5_fullctx_arrays.json`): per-request `ttfts_s`, `itl_sum_s`, `output_lens`,
   `input_lens`, `errors_empty`, gen-nonempty count, full 64-hex source SHA256, and the stored headline +
   aggregate means — NOT a stored derived TPS array.
2. **Fail-closed verifier** (`ac5_fullctx_metrics_tool.py --verify`): RECOMPUTES P99 TTFT = p99(ttfts) and
   per-req TPS p50 = p50(output_len/itl_sum) **from the raw committed arrays** (no derived metric); adds
   **aggregate-mean integrity** (sensitive to every element — catches a single-element tamper that a robust
   median misses); asserts the empty-latency class (every ttft>0, itl_sum>0, output_len==512, errors empty,
   len==completed, gen-nonempty==completed, 64-hex SHA); and validates the operating point from **all three**
   `.meta.json` sidecars (int8 / mem0.7 / radix-on / fixture / full context / TP=8 / stats-on).
   **6 tamper tests each exit 1** (single itl_sum, single output_len, single ttft=0, stored TPS p50→100,
   stored P99 TTFT→5000, sidecar disable_radix_cache→True); clean exits 0 PASS. This closes the R20 leak
   (the exact R20 analog — set stored TPS to 100 — now exits 1).
3. **Committed c32/c64 sidecars** (R20 had only c16). **Filled the decode-component breakdown** in
   `ac5_fullctx_attribution.txt`: per-req decode TPS = gen/#running-req = **24.9/19.5/17.3** at batch
   16/32/38 (matching the client arrays exactly) + the DSA FlashMLA+MoE floor reference (AC-7 verified
   46.1/37.0/29.4 → ~21.7 ms step) + the DS-selection delta (R17 microbench).

## Result (numbers unchanged — measured, now exact-recomputable + fail-closed)
| conc | achieved | P99 TTFT | <22s | per-req TPS p50 | ≥30 |
|---:|---:|---:|:--:|---:|:--:|
| 16 | 16.00 | **13.13 s** | ✅ | 24.9 | ✗ |
| 32 | 31.99 | 25.33 s | ✗ | 19.5 | ✗ |
| 64 | 47.03 | 77.90 s | ✗ | 17.3 | ✗ |

conc-16 meets the strict tail-latency SLO (<22s) at full context; per-req TPS misses 30 (the full-context
topk over-scan residual); conc-32/64 are the structural decode-batch ceiling. Directional per DEC-3.

## Files Changed
- `runs/20260530_dsv32_loop6/ac5_fullctx/`: `ac5_fullctx_arrays.json` (exact raw arrays + means), 
  `ac5_fullctx_metrics_tool.py` (recompute-from-raw + mean-integrity + sidecar invariants + fail-closed),
  `ac5_fullctx_attribution.txt` (decode-component section filled), `ac5_fullctx_report.md` (verifier
  description updated), `meta_c32.json` + `meta_c64.json` (new sidecars).
- `.humanize/bitlesson.md` — `BL-20260530-durable-tracked-acceptance-evidence` updated (recompute from RAW
  not a stored derived metric; add mean-integrity for single-element tampers; validate all sidecars);
  goal-tracker (R21 row + task6 note); round-21 contract/summary (gitignored loop state).
- (No production code change this round — data-only evidence rebuild; R17 decode fix + R19 bench fail-closed
  fix stand.)

## Validation
- `ac5_fullctx_metrics_tool.py --verify` → PASS (recomputes P99 TTFT + per-req TPS p50 + means from raw
  committed arrays == stored headline; all 3 sidecars' operating point verified; no empty-latency rows).
- 6 temporary-copy tamper tests each exit 1 (incl. the single-element itl_sum mutation that leaked before the
  mean-integrity check, and the exact R20 leak analog). `git diff --check` clean; commit `991666b58` pushed
  to `jimmy`. GPUs free (data-only round; no server booted).

## Remaining Items
- conc-16 full-context per-req TPS (24.9 < 30): the residual DS-selection topk over-scan. conc-32/64
  structural (DS ≤ DSA; conc-64 unattainable even for DSA). **Gated AC-10.** Cross-node smoke (future-gated),
  DSA conc-64 TPS ~29.4 (queued) unchanged. No ABI-lock change; DS-fair AC-12 gate unchanged.

## Goal Tracker Update Request
### Requested Changes:
1. **Mark the AC-5 full-context EVIDENCE as acceptance-grade** (R9-standard fail-closed verifier + exact raw
   arrays + all sidecars + component breakdown), resolving Codex R20 blocking issue 1. AC-5 stays Active only
   for the two genuine open decisions below.
2. **Owner approval — AC-5 measurement methodology:** approve `num_prompts=64` steady-state (warmup120/
   window300) as the AC-5 methodology instead of the literal `NUM_PROMPTS=320`. The verified cold-flood
   BitLesson (`BL-20260530-cold-flood-not-steady-state-slo`) shows np320 cold-ramps (window) or full-drains
   the queue (fixed-count → P99 TTFT ≈ full 320-request drain, ~300s, misleading), while np64-window is the
   steady-state methodology that reproduced the DSA baseline (R11/R12) and AC-7. Without approval the literal
   np320 produces a methodologically-wrong number.
3. **Owner decision — conc-16 full-context TPS axis:** the research-grade full-context blocked-topk kernel
   (within-block K=2048 under CUDA-graph) to lift conc-16 from 24.9→~30, vs accepting the bounded-context
   op-point (closed-batch 30.3, 64K servability = separate full-context point AC-8/R16). conc-32/64 ≥30 is
   structurally unattainable (DS ≤ DSA) regardless.
### Justification:
This round delivered the evidence fix Codex required (the verifier is now genuinely fail-closed at the R9 bar,
demonstrated against 6 single-field tampers). The two remaining AC-5 items are decisions the owner must make
(methodology + the kernel-vs-rescope), not more measurement — surfacing them per Codex's instruction that
methodology needs explicit owner approval and the bounded-context target is not a rescope until the owner
changes it.

## BitLesson Delta

Action: update
Lesson ID(s): BL-20260530-durable-tracked-acceptance-evidence
Notes: Extended with the R20/R21 finding — a fail-closed verifier must recompute the published metric from the
RAW committed arrays, NOT re-check a stored DERIVED array (R20's derived `per_req_gen_tps` was tamperable to
100.0 and passed; R21 recomputes per-req TPS = output_len/itl_sum from raw). Also: a robust percentile (p50/
median) is insensitive to a single-element tamper, so add an aggregate-MEAN check (sensitive to every element)
alongside the percentile, and validate the operating point from ALL sidecars — demonstrate fail-closure with
single-element tampers, not just whole-array ones. Applied: BL-20260531-bench-empty-stream-failclosed (the
empty-latency class the verifier asserts), BL-20260530-clean-latency-attribution (per-conc queue_duration +
decode-component breakdown), BL-20260530-cold-flood-not-steady-state-slo (the np64 methodology justification).
