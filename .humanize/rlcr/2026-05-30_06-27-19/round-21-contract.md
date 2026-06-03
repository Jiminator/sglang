# Round 21 Contract

## Mainline Objective (exactly one)
**Rebuild the R20 full-context AC-5 evidence to the R9 fail-closed standard.** Codex R20 found a real
defect: the verifier stored only the *derived* `per_req_gen_tps` (not raw `itls`), so mutating that array
to 100.0 still passed — the strict TPS axis was tamperable, regressing below the R9 bar. Rebuild from the
existing raw JSONLs (`/tmp/ac5r20/results/*.jsonl`, still present — no re-run): commit the exact per-request
source (ttfts, per-request ITL sums + output/input lens + errors, full 64-hex SHA, stored headline) and a
verifier that **recomputes P99 TTFT + per-req TPS p50 from the committed raw arrays** (not a stored derived
value), validates the operating-point invariants from all three sidecars, and **fails closed** (demonstrated
by tamper tests). Fill the empty component-breakdown section and commit all three conc sidecars.

## Target AC(s)
- **AC-5 (task6)** — the done-criterion. `coding` / data-only (rebuild from existing raw artifacts).

## Truly Blocking This Objective
- **The tamperable verifier / lossy committed source** (Codex R20 blocking issue 1). Until the committed
  artifact holds raw arrays the verifier independently recomputes from, the AC-5 numbers are not
  acceptance-grade. This is the round's primary fix.

## Queued / Explicitly Out of Scope This Round
- **The exact full-context blocked-topk kernel** (Codex R20 item 5) — research-grade; the conc-16 full-context
  ≥30 TPS lever. Surfaced as an owner decision (kernel vs explicit bounded-context rescope), NOT this round's
  deliverable (Codex: don't repeat the owner-decision prompt as the main deliverable — the main deliverable is
  the evidence fix).
- **The `NUM_PROMPTS=320` vs `num_prompts=64`-steady-state methodology** — a genuine decision the owner must
  approve (Codex item 3): the verified cold-flood BitLesson shows np320 cold-ramps/full-drains while np64-window
  is the steady-state methodology that reproduced the DSA baseline (R11/R12) and AC-7. Surfaced as a
  Goal-Tracker-Update plan-evolution request; the rebuilt verifier asserts the actual np64-steady-state counts
  and documents the methodology as pending owner approval (no silent 320-by-implication).
- **AC-10** — gated. **Cross-node smoke** — future-gated. **DSA conc-64 TPS ~29.4** — queued.

## Concrete Success Criteria
1. **Exact committed source** per conc (c16/c32/c64): `ttfts_s`, per-request `itl_sum_s` + `output_lens` +
   `input_lens` + `errors` (and a generated-text-nonempty count), full 64-hex source SHA256, and the stored
   headline (`p99_ttft_ms`, `tps_p50`, `achieved`, `completed`). NO stored derived `per_req_gen_tps` as the
   verifier's source.
2. **Fail-closed verifier** that reads ONLY committed files and: recomputes P99 TTFT = p99(ttfts) and per-req
   TPS p50 = p50(output_len/itl_sum) and matches the stored headline at published precision; asserts
   len(ttfts)==len(itl_sum)==len(output_lens)==len(errors)==completed, every output_len==512, every error
   empty, every ttft>0, every itl_sum>0 (no empty-latency rows), 64-hex SHA; and validates ALL THREE sidecars
   prove `enable_double_sparsity`, `signature_dtype=int8`, `mem0.7`, `disable_radix_cache=false`, radix fixture
   set, `context_length=null` (full context), `max_total_num_tokens=396096`, TP=8, request-time-stats on.
3. **Tamper demonstration** (temporary copies): mutate a per-request itl_sum (→ recomputed TPS shifts), an
   output_len, a sidecar radix field, and a stored headline TTFT — EACH exits 1; clean exits 0 PASS.
4. **Component breakdown filled:** `ac5_fullctx_attribution.txt` must contain the actual decode-batch
   `gen throughput / #running-req` lines (DS selection+FlashMLA+MoE) + the measured queue_duration per conc +
   the DSA floor reference — no empty `Decode component` header.
5. All three conc `.meta.json` sidecars committed. GPUs already free (data-only round). Commit + push to
   `jimmy`; goal-tracker + round-21-summary + BitLesson Delta updated; GTU request for the np64 methodology +
   the topk/rescope owner decision.

## Applicable BitLessons (confirm per-task via bitlesson-selector)
- `BL-20260530-durable-tracked-acceptance-evidence` (commit the EXACT source the consumer uses; recompute the
  consumer's value; fail-closed verifier; tolerance ≤ published precision; validate provenance/shape — the
  core lesson this round operationalizes).
- `BL-20260531-bench-empty-stream-failclosed` (the empty-latency class the verifier must assert against).
- `BL-20260530-clean-latency-attribution` (per-conc queue_duration bucketing + measured-vs-inferred for the
  component file).
- `BL-20260530-cold-flood-not-steady-state-slo` (the np64-steady-state methodology justification for the GTU).

## Out-of-bounds reminders
No production decode-path change this round (data-only rebuild). No ABI-lock / FlashMLA-assert change. No
plan-process tokens in code/comments. Do not change the DS-fair AC-12 gate. Must not exit by lying / editing
loop state / cancel.
