# Round 13 Contract

## Mainline Objective (exactly one)
**AC-7: the 3-trial DS+DSA directional re-sweep at the lifted operating point** (DS int8 @
mem 0.7 radix-on, DSA-default @ mem 0.85 radix-on), conc 16/32/64, `num_prompts=64`,
`WARMUP_SECONDS=120`, `MEASUREMENT_WINDOW_S=600`, **TRIALS=3** both sides — then refresh
`ac11_resweep.md` / `ac11_analysis.md` via `benchmark_compare.py --ac11`, showing DS
achieved-concurrency now tracks nominal (vs Loop-5's queue-starved 14.5/24.6/35.7) with no
hidden achieved-concurrency deficit. **Gated by** the cross-node wrapper host-targeting smoke
(Codex R12's deferred success-criterion). AC-5's directional verdict + the open DS strict-SLO
blocker stay tracked, not this round's objective.

## Target ACs (1)
- **AC-7** (`coding`, hardware-run, soft per DEC-9) — AC-11 directional re-sweep at the lifted point.

## Blocking Side Issues in Scope (Codex R12 review)
1. **Cross-node wrapper host-targeting smoke (blocking prereq for AC-7).** Before trusting any
   AC-7 artifact: boot the remote server, run `benchmark_baseline.sh`/`benchmark.sh` with
   `HOST=<remote>`, 1 conc / 1 trial / short window, and capture both the `bench_serving`
   readiness banner naming `http://<remote>:<port>` and the matching `.meta.json`
   server-info sidecar from the same host. If they disagree, stop and fix the wrapper.
2. **Per-side methodology consistency.** Use the Loop-5 ac11 methodology (`num_prompts=64`,
   warmup 120 / window 600, TRIALS=3) — `num_prompts=64` is the steady-state methodology (R11
   finding; `=320` is cold-ramp). Record per-side `mem_fraction_static` (DS 0.7, DSA 0.85),
   radix-on both, and the achieved concurrency per conc.

## Queued / Out of Scope (NOT downgraded)
- **AC-5 DS strict-SLO miss** stays the open mainline blocker (the AC-5 remediation is a later round, per Codex's plan, after AC-7 data is in hand).
- **DSA-default conc-64 TPS ~29.4** stays Queued (pre-existing, not DS).
- **AC-8** (~70K probe), gated **AC-10** — later rounds. No FlashMLA decode-assert changes; DS-fair thresholds unchanged.

## Round Success Criteria
1. Wrapper host smoke passes (readiness banner + sidecar both name the intended remote host); recorded as a tracked artifact.
2. 3-trial DS+DSA sweep at the lifted point completes (radix-on both, num_prompts=64, 120/600, conc 16/32/64), JSONLs + `.meta.json` sidecars captured; DS achieved-concurrency tracks nominal (no hidden deficit).
3. `benchmark_compare.py --ac11` run; `ac11_resweep.md` / `ac11_analysis.md` refreshed under `runs/20260530_dsv32_loop6/` with the lifted-point DS-vs-DSA TPS/TTFT + achieved-conc summary; the DSA conc-64 TPS ~29.4 (Queued) noted, not hidden.
4. `git diff --check` clean; commit + push after each commit; goal-tracker updated (task8/AC-7); `round-13-summary.md` with BitLesson Delta; servers killed + GPUs freed.

## Out-of-Scope Guards
- No fabrication; if DS achieved-concurrency or SLO numbers fall short, record honestly (AC-7 is soft / may be characterized per DEC-9).
- Reuse Loop-5 scripts (now `--host`-fixed); no new scaffolding. Kill stale `sglang::router` before booting. Push between commits to survive pre-emption.
