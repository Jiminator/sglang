# Round 10 Contract

## Mainline Objective
**Run the AC-11 directional comparator (task13): a 3-trial radix-on DSA+DS sweep at conc
16/32/64 (120s warmup, 600s window), then `benchmark_compare.py --ac11`**, with #F handled
honestly — the report must record DS's ACHIEVED (effective) concurrency alongside the nominal
value so a queue-dominated admission at `mem_fraction_static=0.6` is visible, not hidden. A
directional miss (DS TPS < 95% DSA, or DS P99 TTFT > 1.10× DSA) is recorded as an AC-11 failure
+ follow-up (DEC-7), not a build-break.

## Target ACs (≤ 2)
- **AC-11** — 3 DS trials + 3 DSA trials per concurrency (16/32/64), 120s warmup, 600s window,
  radix-on parity on BOTH sides, comparator emits a TPS/TTFT pass-or-fail summary; each JSONL
  duration ≥ 600s with valid sidecars; effective-vs-nominal concurrency recorded.

## Blocking issues in scope
- **#F — DS KV-pool / effective concurrency at mem 0.6.** At the radix-on operating point the
  GSP workload's ~55% shared prefix is cached once and reused, which should raise the
  admittable concurrency vs the radix-off smoke; but DS may still be admission-bound at conc
  64. Handle by ACCOUNTING (Codex option 3, the honest, non-OOM-risking choice): capture each
  run's achieved concurrency (`bench_serving` `concurrency` / `max_concurrent_requests`, and
  spot DS server `#running-req`/`token usage`) and surface effective-vs-nominal in the AC-11
  report. Do NOT raise DS `mem_fraction_static` (0.7 OOMs during generation) or shrink the
  immutable conc set; do NOT publish TTFT that hides queue domination.

## Queued / explicitly out of scope this round
- **task14 AC-12** (NIAH 4K/16K/64K + MMLU 5-shot) — next round. The Round-9 long-context
  recall finding (top_k-bounded) is AC-12 evidence, handled there.
- **task15 evidence bundle** — after AC-12.
- AC-10 label-capture artifact provenance note (server_args null / stale SHA) — fold into task15.
- Stale `calibrate.py` operator recipe docstring.

## Round success criteria
1. AC-11 sweep collected on 8x H200: DSA (3 trials × conc 16/32/64, radix-on) and DS (3 trials ×
   conc 16/32/64, radix-on via the fixtures-passed artifact), 120s warmup, 600s window. Each
   JSONL `duration` ≥ 600s; `.meta.json` sidecars valid (seed, commit_sha, chunked_prefill_size,
   workload, server_args with the locked Option-B fields + matching radix-on). Use a NUM_PROMPTS
   that keeps epochs reasonable while still satisfying the duration/window floors.
2. `benchmark_compare.py --ac11 --ac11-baseline-results <DSA trials> --ac11-ds-results <DS
   trials>` runs to a verdict (exit 0 = all gates pass, 3 = directional miss, 2 = input refusal);
   the comparator must NOT refuse on radix mismatch (both radix-on). Save the markdown + JSON.
3. The AC-11 report explicitly records DS effective vs nominal concurrency (#F accounting); a
   directional miss is documented as an AC-11 failure + follow-up per DEC-7, with the
   queue/admission caveat, rather than hidden.
4. Artifacts saved under `runs/20260528_dsv32_mvp/` (JSONLs gitignored; sidecars + comparator
   reports + server_info committed). Commit + push each step. Goal tracker updated (task13/AC-11
   status, #F accounting); `round-10-summary.md` with a BitLesson Delta. No immutable-section
   changes.

## Known risks / notes
- This is a long sweep (~3 trials × 3 conc × 2 sides at 600s windows). Run in the background with
  long monitors; pick `NUM_PROMPTS` so one epoch is ~100-300s (the time-window loop runs full
  epochs of NUM_PROMPTS — too-large a value makes warmup/window enormous, see
  BL-20260529-dsv32-bench-smoke-sizing).
- Both sides must be radix-on: DS via `RADIX_FIXTURE_ARTIFACT`; DSA is radix-on by default (do
  NOT set `DISABLE_RADIX_CACHE`). The comparator refuses a `disable_radix_cache` mismatch.
- The benchmark hard guard refuses a JSONL whose observed `duration` < 600s — ensure the window
  is genuinely met.
- Operational: don't kill the pre-existing port-30000 router; free port; verify `nvidia-smi`
  clear before each boot; standalone `pkill`/`commit`/`push`.
