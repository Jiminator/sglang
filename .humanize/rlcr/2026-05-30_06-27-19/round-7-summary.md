# Round 7 Summary — AC-5 durable evidence + corrected per-conc attribution

## Mainline objective (round contract)
Close the two AC-5 evidence/attribution blockers from Codex's R6 review **without a re-run** (the R6 client run is real; the local benchmark JSONLs + the full server request-time-stat log are present). This is an evidence/correction round — no production code, the directional verdict is unchanged; it only makes the AC-5 result durable and the attribution honest.

## What landed (commit 51dd009b8)
Three **tracked** files under `runs/20260530_dsv32_loop6/client_slo_int8/`, all recomputable without the gitignored `*.jsonl`:

1. **`ac5_evidence_addendum.txt`** — per conc (16/32/64): completed=320, **errors=0 (all-empty proof)**, achieved concurrency, duration, ISL distribution (min/p50/p99/max ≈ 4274–4295, nominal 4096), OSL=512, **TTFT** min/p50/p90/p99/max, **TPOT** + per-req TPS (1000/median_TPOT = 17.6/11.5/9.3), **ITL**, output throughput, and the radix-on sidecar proof (`disable_radix_cache=False`, `mem_fraction_static=0.7`, `max_total_num_tokens=396096`). Every number in `client_slo_report.md` now recomputes from this file.

2. **`attribution_per_conc.txt`** — reprocessed the **full** server log: 967 rows parsed; **5 invalid `queue_duration<0` rows + 3 HEALTH_CHECK probes filtered with a disclosed drop policy → 959 valid** (the >960 nominal is per-conc warmup requests, explained). Per-conc bucketing by wall-clock run windows (`T0 + cumulative measured durations 829.6/692.7/713.1 s`, since the 3 runs are contiguous with no idle gap to cluster on). **Honest measured-vs-inferred:**
   - MEASURED: `queue_duration` (admission wait) p99 = **10.5 / 22.3 / 99.4 s**; client TTFT p99 = 12.8 / 25.5 / 111.2 s; min TTFT ≈ 1.3 s = uncontended prefill floor.
   - INFERRED: post-admission residual = `TTFT_p99 − queue_p99` = **2.2 / 3.2 / 11.8 s** (prefill + chunked-prefill/decode interleave, NOT pure prefill). Tail-to-tail (p99−p99), not p50−p50 across two distributions whose rows aren't the same request.
   - `forward_duration` (completion-time = prefill + all 512 decode steps) is reported **context-only — never used as a first-token prefill term** (the R6 misuse, corrected).
   - Conclusion: P99 TTFT is admission-wait-dominated at every conc (queue p99 ≫ residual); the queue term grows 10.5→22.3→99.4 s while the prefill floor stays ~1.3 s. Still NOT KV-pool-bound (64×4608=295K < 396K pool) → throughput contention.

3. **`decode_batch_excerpt.txt`** — the TPS root cause, tracked + quantified: steady-state decode batch is **16 / ~32 / ~38** (this **corrects the R6 summary's "#running-req 19-20"** figure), aggregate gen ~270–370 tok/s, so per-req decode TPS = gen/`#running-req` = **17.7 / 11.5 / 9.7 tok/s** — reproducing the client p50 TPS (17.6 / 11.5 / 9.3) almost exactly.

**`client_slo_report.md`** updated: softened "spine validated" → **"directional characterization, not yet validated"**; rewrote the attribution section to the corrected measured-vs-inferred framing; fixed the prefill-floor and decode-batch figures; kept the **strict-SLO miss explicit** (conc-32/64 TTFT 25.5/111.2 s > 22 s; per-req TPS < 30 at every conc) as a **live mainline blocker**; references all three addenda.

## Result (unchanged verdict, now durable + honest)
DIRECTIONAL — accepted progress, explicitly NOT shippable (DEC-3). conc-16 meets strict `<22 s` (12.8 s); conc-32/64 TTFT and all-conc per-req TPS miss the strict SLO and remain an open mainline blocker. The footprint→admission→TTFT spine is *characterized* (not yet validated) by clean, recomputable per-conc evidence; the residual is throughput/decode-batch, not footprint.

## Files Changed
- `runs/20260530_dsv32_loop6/client_slo_int8/`: `ac5_evidence_addendum.txt`, `attribution_per_conc.txt`, `decode_batch_excerpt.txt` (all new, tracked).
- `runs/20260530_dsv32_loop6/client_slo_report.md` (directional-characterization wording + corrected attribution).
- `.humanize/bitlesson.md` (+1 add, 1 update), goal-tracker, round-7 contract/summary (gitignored loop state).

## Validation
- All three evidence files tracked (`git check-ignore` → none ignored); `git diff --check` clean (stripped a trailing EOF blank line); commit 51dd009b8 pushed to `jimmy`.
- Numbers cross-check: addendum per-req TPS (17.6/11.5/9.3, from median TPOT) ≈ decode-batch gen/#running-req (17.7/11.5/9.7); attribution queue p99 (10.5/22.3/99.4) + residual (2.2/3.2/11.8) ≈ client TTFT p99 (12.8/25.5/111.2).
- No re-run, no production code change; the R6 run's `.meta.json` radix-on sidecars are unchanged.

## Remaining Items
- **Open mainline blocker:** strict client SLO still fails (conc-32/64 TTFT > 22 s; per-req TPS < 30 every conc). Characterized (throughput/decode-batch, not footprint); to be solved or characterized at the operating-point level before any strict/shippable claim.
- **AC-6 (task7, next):** DSA-default product property on hardware (DSA-default boot meets SLO unchanged, allocates no DS table; DS opt-in toggles the compact path).
- **AC-7** (3-trial DS+DSA re-sweep at the lifted point), **AC-8** (~70K-token 64K servability probe), **AC-9** (within-budget harness edit from real `usage.prompt_tokens` — DS-fair thresholds UNCHANGED), then gated **AC-10**. No FlashMLA decode-assert changes (AC-3.3).

## BitLesson Delta

Action: add
Lesson ID(s): BL-20260530-clean-latency-attribution, BL-20260530-admission-restore-tps-tradeoff
Notes: Added BL-20260530-clean-latency-attribution capturing the R6->R7 cross-round fix: when attributing a tail latency metric from a server request-time-stat log under continuous batching — parse the full log and disclose total-vs-valid + a filtering policy for impossible rows (queue_duration<0, HEALTH_CHECK) and reconcile the count gap (warmup); state measured (queue_duration, client TTFT, min-TTFT prefill floor) vs inferred (post-admission residual = TTFT_p99-queue_p99, which includes chunked-prefill/decode interleave, not pure prefill); never use a completion-time counter (forward_duration) as a first-token term; compare tail-to-tail (p99-p99) not p50-p50 across two distributions; bucket per-conc by wall-clock run windows when runs are contiguous; attach the decode-batch excerpt if citing batch growth as the TPS root cause. Updated BL-20260530-admission-restore-tps-tradeoff's Validation Evidence to the corrected decode-batch figures (steady-state batch 16/~32/~38, per-req 17.7/11.5/9.7 tok/s; the R6 "#running-req 19-20 -> ~14 tok/s" was imprecise). Applied existing BL-20260530-durable-tracked-acceptance-evidence (embed metrics as tracked .txt, not gitignored .jsonl/.csv; git check-ignore + git diff --check before claiming done) and the push-between-rounds preference (pushed 51dd009b8 to jimmy).
