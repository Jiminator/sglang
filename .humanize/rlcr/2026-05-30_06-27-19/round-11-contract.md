# Round 11 Contract

## Mainline Objective (exactly one)
**Finish AC-6 properly on hardware:** a DSA-default client-SLO confirmation under the
**proper steady-state methodology** that actually passes the SLO, plus a DS-opt-in
toggle proof at the **locked radix-on operating point** — and an AC-6 doc that claims
only what the artifacts prove. Codex's R10 review verified AC-9 (MET) but rejected the
AC-6 evidence: the WARMUP=0 smoke shows DSA *failing* the SLO (cold-ramp artifact) and
the deferral to AC-7/established-baseline is not accepted; and the DS boot was radix-OFF
(`disable_radix_cache=true`) while DSA was radix-ON, contradicting "differ only by DS
enablement." AC-5's directional verdict + the open strict-SLO blocker stay tracked but
are not this round's objective.

## Target ACs (1)
- **AC-6** (`coding`, hardware) — DSA-default boot meets the SLO unchanged (proper
  methodology, tracked, **P99 TTFT < 22 s AND ≥ 30 TPS at every conc**) and allocates no
  DS table; DS opt-in toggles the compact int8 path **at the locked radix-on point**.

## Blocking Side Issues in Scope (Codex R10 review)
1. **DSA-default SLO under proper methodology.** `benchmark_baseline.sh` against DSA-default
   (mem 0.85, radix-on, no DS), conc 16/32/64, `NUM_PROMPTS=320`, `TRIALS=1`,
   `WARMUP_SECONDS=120`, `MEASUREMENT_WINDOW_S=600`. Track JSONL-derived summaries
   sufficient to recompute completed/errors/P99 TTFT/per-req TPS. AC-6 passes only if
   P99 TTFT < 22.0 s and per-req TPS ≥ 30 at **every** conc; otherwise AC-6 stays failed
   and the miss is recorded honestly.
2. **DS opt-in at the locked radix-on point.** Re-boot DS with `SIGNATURE_DTYPE=int8`,
   `MEM_FRACTION_STATIC=0.7`, `RADIX_FIXTURE_ARTIFACT=runs/20260530_dsv32_loop6/ds_radix_fixture_state_int8.json`.
   Capture `/get_server_info` proving `enable_double_sparsity=true`, `signature_dtype=int8`,
   **`disable_radix_cache=false`**, the fixture path; boot excerpt proving the int8
   `token_label_table` on all 8 ranks.
3. **Honest AC-6 doc.** Rewrite `ac6_optin_dsa_default_product.md` to claim only what the
   artifacts support; the WARMUP=0 smoke is kept as admission-only context (or removed
   from the SLO proof), and the radix-on parity is corrected.

## Queued / Out of Scope (explicitly NOT downgraded)
- **Strict-SLO failure stays the open mainline blocker** (AC-5 conc-32/64 TTFT > 22 s; per-req TPS < 30). Tracked, not this round.
- **AC-7** (3-trial DS+DSA re-sweep), **AC-8** (~70K probe), gated **AC-10** — later rounds. No FlashMLA decode-assert changes (AC-3.3); do not change DS-fair thresholds (AC-9, already MET).

## Round Success Criteria
1. Tracked DSA-default proper-methodology SLO artifact (120/600/320) with per-conc completed/errors/P99 TTFT/per-req TPS; AC-6 verdict stated against `<22 s` AND `≥30 TPS` at every conc (pass → AC-6 met; miss → recorded honestly, AC-6 stays partial).
2. Tracked DS opt-in `/get_server_info` showing `disable_radix_cache=false` + fixture path + `signature_dtype=int8` + the int8 table on 8 ranks (radix-on locked point).
3. `ac6_optin_dsa_default_product.md` rewritten to match the artifacts (radix-on parity; WARMUP=0 demoted to admission-only).
4. `git diff --check` clean; commit + push to `jimmy` after each commit; goal-tracker updated (task7/AC-6); `round-11-summary.md` with BitLesson Delta; servers killed + GPUs freed.

## Out-of-Scope Guards
- No fabrication: if DSA-default genuinely misses the SLO under proper methodology, record it honestly and keep AC-6 partial (do not reclassify a miss as a pass).
- Do not weaken the strict SLO or mark the loop done. No new serve/bench scaffolding (reuse Loop-5 scripts). Kill stale `sglang::router` before booting.
