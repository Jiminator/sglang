# Round 18 Summary — AC-5 conc-16 strict-decode PASS (bounded-context op-point) + owner done-criterion

## Owner decision (R12-style)
Surfaced the structural finding via AskUserQuestion: `≥30 TPS/req at every conc 16/32/64` is unattainable
for DS — per-request decode TPS falls as the decode batch grows, and even **DSA (the faster path DS cannot
exceed) is 29.4 at conc-64** (DSA 46.1/37.0/29.4). The owner chose **"conc-16 strict + characterize 32/64"**
(confirmed as my recommendation): strict gate = conc-16 (≥30 TPS/req AND P99 TTFT <22 s); conc-32/64 are
characterized as the structural decode-batch ceiling (DS ≤ DSA), not a DS/footprint defect. Recorded as a
Plan Evolution row in `goal-tracker.md`.

## Mainline objective (round contract)
Make DS strict-pass the client SLO at conc-16 and characterize conc-32/64, per Codex's R17 plan (residual
top-k over-scan first, then the client numbers), preserving the ABI lock and 64K servability (AC-8).

## What landed (commit `fcc2d1cdb`; no production code change — operating point + evidence)
1. **Technical finding on the residual top-k over-scan.** R17's score-kernel early-exit left the first
   `torch.topk(scores[:bs,:max_seq_len], 2048)` scanning the full `max_seq_len = req_to_token.shape[1] =
   context_len = 163840`. Under **CUDA-graph capture the topk score-buffer width is fixed at capture and
   `torch.topk` (a monolithic reduction) cannot skip** rows past `seq_len` — so a no-context-cap graph-safe
   topk speedup needs a research-grade K=2048 skipping kernel (a torch two-level/reshape topk still processes
   the full width; the stubbed `DSGraphState.scratch_partial_*` two-stage path was never implemented).
2. **The cheap, exact lever — bounded-context client-SLO operating point.** The topk scan width == the model
   context length, and the client workload is 4096 ISL + 512 OSL = 4608 tokens, so `--context-length 8192`
   shrinks `req_to_token` width 163840→8192 (the topk then scans 8192), KV pool unchanged
   (`max_total_num_tokens=396224`, mem 0.7), **+9 GB headroom**. 64K servability (AC-8) is the **separate
   full-context operating point**, already validated in R16 — two honest operating points.
3. **conc-16 strict-decode MET.** Closed-batch pure decode (own client, `ignore_eos`, real 512-step decode,
   server-log-confirmed; no prefill interleave) at DS int8 / mem-0.7 / radix-on / **ctx 8192** + R17 score-fix:
   conc-16 **27.1 → 30.3 TPS/req (PASS ≥30)**, conc-8 36.0, conc-1 43.6. Fail-closed verifier
   `ctx8192_decode_metrics_tool.py --verify` recomputes per-req TPS = median(gen)/batch from committed
   samples (conc-16 ≥30 asserted; tampered conc-16 sample 29.38 → exit 1; clean → exit 0 PASS).
4. **conc-32/64 characterized.** 27.2 / 22.6 TPS/req at ctx 8192 (up from full-ctx ~20/~16, still < 30) — the
   decode-batch→TPS structural ceiling; DS < DSA (37 / 29.4); conc-64 ≥30 unattainable even for DSA.

## Result
The conc-16 decode-TPS axis (the previously-failing axis) now strict-passes (30.3 ≥ 30, verifier-checked)
at the bounded-context client-SLO operating point, with the ABI lock intact and 64K servability preserved
as the separate full-context deployment. conc-32/64 are characterized as the structural decode-batch
ceiling per the owner decision + DEC-3.

## Files Changed
- `runs/20260530_dsv32_loop6/ac5_conc16_strict/` (NEW): `ac5_conc16_strict.md` (the report + the two-operating-points
  framing + the technical finding), `ctx8192_decode_curve.json` + `ctx8192_decode_metrics_tool.py` (exact
  closed-batch samples + fail-closed verifier), `ctx8192_decode_curve.txt` / `closed_batch_ctx8192.txt`
  (decode-curve excerpts), `get_server_info_ctx8192.json` (operating-point sidecar).
- `.humanize/bitlesson.md` — extended `BL-20260531-ds-selection-fullwidth-overscan` with the R18 addendum
  (capture-width-bound topk; bounded-context lever; bench window-mode caveat); goal-tracker (R18 owner-decision
  Plan Evolution row + task6 note); round-18 contract/summary (gitignored loop state).
- (No production code change this round; the R17 score-kernel fix `selection_kernel.py` is already committed.)

## Validation
- `ctx8192_decode_metrics_tool.py --verify`: PASS (conc-16 30.33 ≥30; conc-32/64 27.17/22.6 < 30; monotone
  sanity); tamper (conc-16 → 29.38) exits 1.
- Operating point proven: `get_server_info_ctx8192.json` (int8 / mem 0.7 / radix-on / context_len 8192 /
  pool 396224 / TP=8). Coherence on the ctx8192 server: "The capital of France is" → " Paris. The capital of
  the United States is Washington, D" (no degeneration).
- GPUs freed at round end (all 8 at 0 MiB). `git diff --check` clean; commit `fcc2d1cdb` pushed to `jimmy`.

## Remaining Items
- **Open residual (conc-16 TTFT):** conc-16 P99 TTFT <22 s is supported by the R6 Codex-verified full-context
  12.8 s (ctx8192 decodes faster → TTFT only lower), but a **fresh ctx8192 TTFT-under-flood number** was not
  captured: `development/benchmark.sh` bench_serving WINDOW mode returned empty per-request latency arrays +
  impossible aggregate throughput in this build (at WARMUP=0 and 120) while the server generated correctly on
  direct `/generate` and under the closed-batch client. Resolving the window-mode harness (or using a working
  flood client) to publish a fresh conc-16 P99 TTFT is the item to fully close conc-16 strict.
- **conc-32/64** characterized (structural; DS ≤ DSA, conc-64 unattainable even for DSA) per the owner decision.
- **Gated AC-10** — only after AC-5 is met under the owner criterion + AC-3..AC-9 verified.
- Cross-node wrapper smoke (future-gated) and DSA-default conc-64 TPS ~29.4 (queued) unchanged. No ABI-lock /
  FlashMLA-assert change; DS-fair AC-12 gate unchanged.

## Goal Tracker Update Request
### Requested Changes:
- Record the **R18 owner decision** as accepted Plan Evolution: AC-5 done-criterion = **conc-16 strict-pass
  (≥30 TPS/req AND P99 TTFT <22 s) + conc-32/64 characterized** (already added to the Plan Evolution Log).
- Mark AC-5 **conc-16 decode-TPS axis MET** (30.3 ≥30, fail-closed verifier) at the bounded-context client-SLO
  operating point; keep task6/AC-5 Active only for the fresh-ctx8192 conc-16 P99 TTFT residual (harness-blocked,
  strongly supported by R6 12.8 s).
### Justification:
The all-conc strict pass is structurally impossible (conc-64 ≥30 unattainable even for DSA); per DEC-3 + the
Lower Bound the owner set the realizable MVP done-criterion. The conc-16 decode-TPS strict-pass is the novel,
previously-failing result and is now verifier-checked; the only remaining gap is a harness-blocked fresh TTFT
number whose target was already met + verified at the full-context point in R6.

## BitLesson Delta

Action: update
Lesson ID(s): BL-20260531-ds-selection-fullwidth-overscan
Notes: Added the R18 addendum — the residual topk over-scan is **capture-width-bound** (CUDA-graph fixes the
topk buffer width; `torch.topk` can't skip → a no-context-cap graph-safe speedup needs a research-grade K=2048
skipping kernel), so the cheap EXACT lever is the **bounded-context operating point** for a latency-sensitive
bounded workload (client SLO 4608 tokens → `--context-length 8192`), keeping long-context as a separate
full-context op-point; closed-batch conc-16 27.1→30.3 (clears ≥30), conc-32/64 27.2/22.6 (structural ceiling,
DS ≤ DSA). Plus the bench_serving WINDOW-mode empty-latency-array caveat (use the closed-batch / server-log
gen-throughput for pure decode TPS). Applied existing lessons: BL-20260530-admission-restore-tps-tradeoff
(per-req TPS = 1/decode_step_time; batch→TPS ceiling), BL-20260530-cold-flood-not-steady-state-slo (cold-flood
TTFT is a conservative upper bound), BL-20260530-durable-tracked-acceptance-evidence (exact samples + fail-closed
verifier), BL-20260530-remote-server-launch (background boot + pkill||true; foreground sleep blocked).
