# Round 22 Contract

## Mainline Objective (exactly one)
**Harden the AC-5 full-context verifier to be fail-closed on the workload + full operating-point identity,
and obtain the two owner decisions the loop needs to converge.** Codex R21 demonstrated a real fail-open
gap: `ac5_fullctx_metrics_tool.py --verify` validates only selected `server_args` flags, so mutating the
sidecar's workload fields (`mode`, `num_prompts`, `isl_total_tokens`, `osl_tokens`) and
`server_args.max_total_num_tokens` to garbage still exits 0 PASS. Fix the verifier so every sidecar proves
the AC-5 workload identity (mode, conc matches the artifact key, ISL 4096 / OSL 512, warmup/window,
`chunked_prefill_size=8192`, `max_total_num_tokens=396096`) with tamper tests. Then surface — directly to
the owner — the two convergence-gating decisions Codex says require explicit approval: (a) the AC-5
measurement methodology (np64-steady-state vs literal `NUM_PROMPTS=320`), and (b) the conc-16 full-context
TPS path (research-grade blocked-topk kernel vs explicit bounded-context rescope).

## Target AC(s)
- **AC-5 (task6)** — the done-criterion. `coding` (verifier hardening, data-only) + owner decisions.

## Truly Blocking This Objective
- **The verifier workload-metadata fail-open** (Codex R21 blocking issue 1): until the verifier proves the
  AC-5 workload identity + full operating point from every sidecar, the evidence is not acceptance-grade.
  Tractable, must-fix this round.
- **The two owner decisions** (Codex R21 blocking issues 2 & 3): the methodology and the kernel-vs-rescope
  cannot be resolved by Claude — Codex explicitly requires owner approval; the loop cannot converge without
  them. Surfacing them to the owner (AskUserQuestion, R12/R18 precedent) is the convergence path.

## Queued / Explicitly Out of Scope This Round
- The research-grade full-context blocked-topk kernel — only pursued if the owner picks the kernel path
  (decision b). Not implemented speculatively this round.
- The literal `NUM_PROMPTS=320` re-run — only if the owner rejects np64 (decision a); the cold-flood
  BitLesson shows np320 is methodologically wrong (full-drain → ~300s P99 TTFT, or cold-ramp).
- **AC-10** — gated. **Cross-node smoke** — future-gated. **DSA conc-64 TPS ~29.4** — queued.

## Concrete Success Criteria
1. **Verifier hardened + fail-closed on workload identity:** `--verify` reads every sidecar and asserts
   `mode=double_sparsity`, sidecar `concurrency` == the artifact key, `isl_total_tokens=4096`,
   `osl_tokens=512`, the recorded warmup/window, `server_args.chunked_prefill_size=8192`, and
   `server_args.max_total_num_tokens=396096` — in addition to the existing flag invariants. The committed
   `num_prompts`/`completed` are asserted to MATCH the artifact (with the methodology recorded) — not silently
   accepted as the literal AC-5 320. Demonstrate the Codex tamper (`mode=baseline`, `num_prompts=320`,
   `isl=1`, `osl=1`, `max_total=1`) now each exit 1; clean exits 0 PASS.
2. **Owner decisions obtained** (AskUserQuestion): (a) np64-steady-state vs literal np320 methodology;
   (b) conc-16 full-context TPS — blocked-topk kernel vs bounded-context rescope. Record the answers as a
   Goal-Tracker Plan-Evolution row and set the next round's direction accordingly.
3. GPUs already free (data-only round). Commit + push to `jimmy`; goal-tracker + round-22-summary +
   BitLesson Delta updated.

## Applicable BitLessons (confirm per-task via bitlesson-selector)
- `BL-20260530-durable-tracked-acceptance-evidence` (validate ALL provenance/identity the verifier relies on,
  not a subset; recompute-from-raw; fail-closed with tamper tests — the lesson this round extends to workload
  identity).
- `BL-20260530-cold-flood-not-steady-state-slo` (the np64-vs-np320 methodology justification for decision a).
- `BL-20260531-ds-selection-fullwidth-overscan` (the conc-16 full-context TPS lever for decision b).

## Out-of-bounds reminders
No production decode-path change this round unless the owner picks the kernel path. No ABI-lock / FlashMLA
-assert change. No plan-process tokens in code/comments. Do not change the DS-fair AC-12 gate. Must not exit
by lying / editing loop state / cancel.
