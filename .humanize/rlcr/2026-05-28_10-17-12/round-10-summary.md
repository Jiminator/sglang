# Round 10 Summary

## Work Completed
Ran the AC-11 directional comparator (task13) — a 3-trial radix-on DSA+DS sweep — and handled
#F honestly by surfacing effective-vs-nominal concurrency in the comparator/report. The
directional targets are MISSED (DS TTFT admission-bound), recorded as an AC-11 failure +
follow-up per DEC-7 (not a build-break).

- **AC-11 sweep on 8x H200.** DSA (radix-on default, mem 0.85) and DS (radix-on via the
  fixtures-passed artifact, no env override, mem 0.6), 3 trials × conc 16/32/64, 120s warmup /
  600s window, NUM_PROMPTS=64. 18 JSONLs, each `duration ≥ 602s`; sidecars carry the locked
  Option-B fields with matching `disable_radix_cache=false`, TP=8, fp8, page 64,
  `chunked_prefill_size=8192`.
- **Comparator verdict — directional MISS (DEC-7).** `benchmark_compare.py --ac11` exit 3.
  DS TPS: conc16 0.726 FAIL, conc32 0.900 FAIL, **conc64 1.146 pass** (DS faster). DS P99 TTFT
  FAIL at all conc (57.7/132.9/292.0s vs DSA 0.73/1.37/2.04s) — DS is admission/queue-bound.
- **#F handled (Codex option 3 — account, don't hide).** The comparator now:
  - treats `double_sparsity_radix_fixture_artifact` as a DS-only field;
  - treats `random_seed` (per-boot telemetry) and `mem_fraction_static` (DS reserves a
    TokenLabelTable → 0.6 vs DSA 0.85, an unavoidable asymmetry, NOT a locked Option-B field)
    as recorded-not-matched — while all locked Option-B fields stay strictly matched (a real
    mismatch still refuses, exit 2);
  - emits an **effective-vs-nominal concurrency** table (DS achieved 14.5/24.6/35.7 =
    91%/77%/56% of nominal; DSA ~100%) so the TTFT gap is shown to be partly admission-bound.
  Radix-on lifted DS effective concurrency far above the radix-off smoke's ~2.
- **Follow-up filed** (`ac11_analysis.md`): lift DS effective concurrency (TokenLabelTable
  footprint / KV budget) and re-sweep; profile DS admission at conc 64. Not blocking the
  recorded AC-11 artifact.

## Files Changed
- `development/benchmark_compare.py` — DS-only artifact key; `_AC11_IGNORED_SERVER_ARG_KEYS`
  (`random_seed`, `mem_fraction_static`); `RunMetrics.achieved_concurrency`; effective-vs-nominal
  table in the AC-11 report.
- `test/registered/unit/development/test_ac11_comparator.py` — removed `mem_fraction_static`
  from the refuse-cases; added a regression that the ignored fields don't refuse while real
  Option-B mismatches still do.
- `runs/20260528_dsv32_mvp/` — `mvp_compare_ac11.{md,json}`, `ac11_analysis.md`,
  `ac11_results/*.meta.json` (18 sidecars), `ac11_{dsa,ds}_server_info.json`.
- Commit `a24bc469c`. Pushed. (Raw `*.jsonl` gitignored; metrics preserved in sidecars +
  comparator JSON.)

## Validation
- `pytest test/registered/unit/development/test_ac11_comparator.py
  test/registered/unit/layers/attention/test_double_sparsity_unit.py
  test/registered/unit/manual/test_dsv32_quality_smoke_sequential.py
  test/registered/unit/development/test_option_b_scripts.py -q` → **359 passed** (28 subtests).
- AC-11 hardware: 18 JSONLs ≥602s, radix-on parity both sides; comparator ran to a directional
  verdict with the #F effective-concurrency table.

## Remaining Items
- **task14 AC-12:** NIAH 4K/16K/64K + MMLU 5-shot via `test_double_sparsity_v32.py`. The
  Round-9 long-context recall finding (top_k-bounded; `BL-...-longcontext-needle-recall-vs-topk`)
  is the key risk for NIAH 64K — distinguish a sparse-recall limit from a regression.
- **task15 evidence bundle:** assemble AC-Q/AC-10/AC-1b/AC-11/AC-12 artifacts, raw-JSONL
  locations, sidecars, server args, CUDA-graph status, mask provenance, comparator reports, and
  the AC-10 label-capture provenance note.
- Queued cleanup: stale `calibrate.py` operator recipe docstring.

## BitLesson Delta
- Action: add
- Lesson ID(s): BL-20260529-ds-vs-dsa-memfraction-admission-asymmetry
- Notes: Added the lesson that DS and DSA cannot share `mem_fraction_static` (DS reserves a
  TokenLabelTable → 0.6 vs DSA 0.85), so an apples-to-apples serving comparator must treat
  mem-fraction (and per-boot random_seed) as recorded-not-matched while keeping the locked
  Option-B fields strict, and must surface achieved concurrency — because a DS TTFT miss at
  high nominal concurrency is admission-bound (mem-0.6 KV pool), not per-token latency (DS TPS
  was competitive-to-better). Records the comparator changes + the DEC-7 directional-miss
  handling.

## Goal Tracker Update Request

### Requested Changes:
- Confirm task13 / AC-11 as **EXECUTED with a recorded directional MISS** per DEC-7 (comparator
  ran, verdict + follow-up published, #F effective-concurrency surfaced). It is not a green
  pass — DS P99 TTFT misses the 1.10× target (admission-bound) and DS TPS misses at conc 16/32.
- Confirm #F as **accounted** (effective-vs-nominal in the AC-11 comparator/report), with the
  TokenLabelTable/KV-budget follow-up filed in `ac11_analysis.md`.

### Justification:
Per the immutable AC-11 (DEC-7), DS TPS-within-5% and P99-TTFT-≤1.10× are DIRECTIONAL targets;
a miss is recorded as an AC-11 failure + follow-up, not a build-break. The comparator ran to a
verdict and the report distinguishes effective from nominal concurrency, satisfying the "do not
hide queue-dominated admission" requirement Codex set for #F. The loop4-compatible MVP narrative
should state AC-11 as "comparator complete; directional TTFT target missed (admission-bound at
mem 0.6), follow-up filed." I did not alter the immutable AC or the directional thresholds.
