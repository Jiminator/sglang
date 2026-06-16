# loop11b queue — single source of truth

Finish loop 11's unstarted M4 verdict milestone on a fresh 8×H200 node, then productionize DS UX.
Honest-verdict posture: a FAIL on the throughput SLO is a complete, reportable result. The plan of
record is `development/loop11b/plan.md`; this queue is the live status ledger (committed every round).

## Op-point (this node)

| fact | value |
|------|-------|
| hardware | 8× NVIDIA H200, 143771 MiB/GPU, all idle at kickoff (fresh node, different physical box than loop 11) |
| model | GLM-5.1-FP8 snapshot `…/models--zai-org--GLM-5.1-FP8/snapshots/f396cf805182f4ca10fa675e1a99815b3ca384db` (present ✓) |
| serving | TP=8, page 64, kv fp8_e4m3, custom-all-reduce ON, flashmla_kv both phases, CUDA graph ON, mem 0.8 (DS) / 0.85 (DSA), max_running_requests=64, cuda_graph_max_bs=64 |
| mask | `/models/glm51-fp8-channel-mask-s256.safetensors` is GONE (/models does not exist) → REGEN MANDATORY (task4) |
| mask out-path (decision) | regen target: `/cluster-storage/models/glm51-fp8-channel-mask-s256.safetensors` (durable shared storage; survives node release). DEC-1 content-hash makes path non-authorizing. Commit a copy under `development/loop11b/artifacts/` if small enough (AC-8 reproducibility). |
| workload | gsp 4096/512, ~55% prefix, seeds {16:213, 32:431, 64:31234}, server seed 20260607 |
| hard rule | NEVER set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments` for serving (breaks custom-all-reduce IPC at GLM TP=8). One TP=8 server at a time. |

## Milestone map

| milestone | tasks | gate | status |
|-----------|-------|------|--------|
| M-A op-point re-establishment | task1–task6 | mask regen+provenance, DEC-1 validator change, radix-on minted+authorized, bs cap ≥64 @0.8, AC-7 clean | IN PROGRESS (Round 0) |
| M-B M4 close | task7 (tax guard), task8 (locked sweep), task9 (headline) | AC-4 ratio ≤1.10 @bs64; AC-2 P99<22s; AC-3 p50≥30 (absolute); both op-points | pending |
| M-C productionize | task10 | runbook + Cat-A/B UX fixes; loop8 warning reconciled; no ABI change | pending |
| close-out | task11 | results.md regenerated, evidence preflight, push | pending |

## Task ledger

| id | description | targeted quantity | expected effect | lossiness posture | compat vs landed loop-11 | status |
|----|-------------|-------------------|-----------------|-------------------|--------------------------|--------|
| task1 | populate this queue.md kickoff ledger | — (AC-8 discipline) | single source of truth exists | none | additive | **in progress** (Round 0) |
| task2 | repoint serve-script MODEL_PATH/CHANNEL_MASK_PATH → GLM-5.1-FP8 | model_path matches fixture boot; mask path exists | correctness precondition + UX-A fix | none (default-only) | additive; no flag/ABI change | pending |
| task3 | pre-sweep methodology review (codex, analyze) | comparator-gap list resolved into sweep design | sweep is fair + honest before it runs | none (planning) | informs task8 | pending |
| task4 | regenerate GLM-5.1 channel mask + provenance.json | mask `content_sha256` + full-file `sha256`; recall-comparability vs frozen baseline | query-side mask restored; serves + clears AC-5 | recall-gated ±0.5pp | re-mint expected (full-file SHA non-reproducible: embedded created_at) | **DONE** — ld32 mask `35155ac4…` (loop8 DEC-3 recipe fp8_e4m3/label-dim 32); serves; recall MATCHES baseline (off L4096 58.045%=base, L16384 36.36% +0.32pp, overall 64.80%, all ≤±0.5pp); provenance committed. Superseded ld16 a4be98c4 (−5.2pp). |
| task5 | DEC-1 validator content-hash change + mint radix-on | content-hash fixture authorizes no-override boot; /server_info locked keys | radix-on earned for this boot; mask portable across path/timestamp | value-equiv re-verified on mint (DEC-12) | shared surface → AC-7 regression same round; legacy/label schemas stay rejected | DEC-1 code+tests DONE (`5ac86f5cf`); **GATE A/B/C ALL PASS** (recall on-vs-off equiv ≤±0.5pp; cross-rank identity + no-dense-fallback; edge boundary −0.38/partial@2752 −0.01/evict 0.0pp, nearfull +1.57pp out-of-contract). **Fixture MINTED** `35155ac4` content-hash; no-override boot **AUTHORIZED live** (validator PASSED, real radix hit, /server_info fixture set). DEC-1 altpath portability verifying. |
| task6 | capacity + AC-7 reconfirm | derived decode-bs cap ≥64 @0.8; conc-64 peak ≥61; DSA token_capacity un-regressed | M4 unblocked; AC-7 clean | none | refs 504640 (DS) / 410560 (DSA) are references, not hard gates | **DS-side DONE** — token_capacity **504640** (=loop11 ref), eff_max_running_req_per_dp 64, capture OK (153s), no table, serves; DSA AC-7 pending boot |
| task7 | per-step tax guard @ bs64 (loop-11 task8) | bs64 DS/DSA window ratio ≤1.10; bs30 ≤380k µs (GRAPH) | loop-10 per-step win not traded for capacity | none (measurement); q4/q5 exact-only if pulled | conditional reducer is exact (no recall change) | pending |
| task8 | locked DS-vs-DSA sweep (loop-11 task9) | conc 16/32/64 × 2 trials/seed; DS p50 decode-TPS, P99 TTFT; per-trial reuse + dense_fallback_total | the verdict numbers | dense_fallback_total==0 + sparse-selection proof per trial | both op-points (prod-envelope + same-memory); comparator floor→2 | pending |
| task9 | headline DS-vs-DSA report | one table on SLOS.md SLOs + honest verdict | the loop's reason to exist | n/a | retires/rewrites the loop8 throughput warning | pending |
| task10 | production UX pass (no ABI) | runbook + Cat-A/B fixes; loop8 warning reconciled | DS enablement is a short documented path | n/a | DEC-5: no flag rename / JSON-schema change | pending |
| task11 | close-out + evidence preflight + push | all artifacts exist+tracked+claims match POST-commit | reviewer reproduces verdict from committed artifacts | n/a | rewrite-over-append results.md | pending |

## Kickoff ideas / side issues (append-only; no silent deletions)

Discovered by the task3 codex methodology review (queued — none block M-A; all bind to task8/M-B).
Full review preserved at `development/loop11b/runs/20260616_ma/task3_codex_methodology_review.md`.

- SI-1 (task8, AC-9): `bench_serving.py` does NOT dump per-request `cached_tokens` / `cached_tokens_details`
  (they exist in `meta_info`, tokenizer_manager.py:1758, but RequestFuncOutput drops them, bench_serving.py:96/1901).
  → add a request-metadata sidecar so prefix-reuse (~55%) is captured from request metadata, not inferred from GSP shape.
- SI-2 (task8, AC-5): no per-request/aggregate `total_tokens` for the no-op proof; comparator reads
  `selected_tokens_mean`/`dense_fallback_total`/`total_tokens_mean` (benchmark_compare.py:255) and DS emits
  `selected_tokens`/`sparsity_rate`/`dense_fallback` (metrics.py) → add `total_tokens` or derive (recorded formula).
- SI-3 (task8, AC-3/DEC-6): `--ac11` currently GATES the DS/DSA ratio and exits nonzero on ratio failure
  (benchmark_compare.py:1323) — conflicts with DEC-6 (judge DS on the ABSOLUTE SLO; ratio is competitive-position
  reporting). → make the ratio report-only (keep the absolute 30/22 gate) AND lower `AC11_MIN_TRIALS` 3→2 (bc.py:310;
  refusal checks use only that constant, bc.py:1199; `_median` handles even counts, bc.py:319).
- SI-4 (task8, DEC-4): `--ac11` reports medians only (`_median_metrics`, bc.py:326); DEC-4 needs min/median/max
  for decode-TPS, P99 TTFT, achieved concurrency, throughput.
- SI-5 (task8, AC-3): aggregate `output_throughput`/`total_throughput` emitted by bench_serving (bench_serving.py:1848)
  but not carried by the comparator → report (NOT gate).
- SI-6 (task8, AC-9): `_bench_meta_writer.py` sidecar lacks boot-order + GPU thermal/clock → extend for the run-order ledger.
- SI-7 (task8+task10): `benchmark.sh`/`benchmark_baseline.sh` default `TRIALS=3` + "independent trials" wording
  → 2 repeated stability trials + reworded (the wording fix is also a task10 UX-A item).

## Sweep methodology (task8/M-B design — distilled from task3 codex review)

- **Run order:** one TP=8 server at a time. Per op-point, paired adjacent boots, alternate side by trial:
  `c{16,32,64} t1: DSA→DS`, `t2: DS→DSA`; same per-conc seed (213/431/31234). Block-scheduled (all DSA then all DS)
  → LABEL unpaired, diagnostic-only. Log per boot: op_point_id, pair_id, logical_trial_id, side, boot_sequence,
  exact cmd/env (prove no `expandable_segments`), `/server_info` snapshot, GPU thermal/clock at 4 points, log+JSONL paths.
- **Two op-points, SEPARATE comparator invocations + artifact trees** (do not mix in one `--ac11`):
  `production_envelope` (DS 0.8 / DSA 0.85), `same_memory_080` (both 0.8). Comparator ignores cross-side mem
  (bc.py:527) but enforces within-side constant (bc.py:900). Both radix-ON (refuses mismatch, bc.py:286).
- **Recall comparability:** boot DS recall-oracle (eager), run
  `loop7/niah_oracle_sweep.py --lengths 1024 4096 16384 --num 20 --decode-steps 4`, gate with
  `loop9/oracle_recall_summary.py --baseline loop9/runs/20260610_m0/recall_baseline.json --max-delta-pp 0.5`
  (fails on length-set / per-length sample-count mismatch; baseline = 6240 samples/length × 20 trials). If served-fp8
  is not comparable, define + record a fresh served-fp8 baseline.
- **Admission cap:** if DS achieved-conc < nominal (bc.py:1017), LABEL the row admission-capped, judge DS vs the
  absolute SLO, treat lower-conc reruns as diagnostics only.

## Decisions in force (owner-resolved; see plan.md "Pending User Decisions")

- DEC-1: validator pins mask by tensor `content_sha256` + config fingerprint (replaces full-file SHA + path).
- DEC-2: publish BOTH production-envelope (DS0.8/DSA0.85) AND same-memory (both 0.8) comparisons.
- DEC-3: judge per-request median decode-TPS; aggregate total-tokens/s reported, not gated.
- DEC-4: two trials/conc at the same per-conc seed (repeated measurements); comparator `--ac11` floor→2.
- DEC-5: Category A + B UX only; no CLI flag rename / `--double-sparsity-config` JSON-schema change.
- DEC-6: DS judged on the absolute SLO (30 TPS / P99 TTFT < 22 s) regardless of DSA; DS/DSA ratios reported.
- DEC-12 (loop-11, carried): radix-on authorized under the production-representative-reuse edge contract;
  near-full reuse (+1.57pp) is out-of-contract value-affecting, recorded but not a gate input.

## Frozen references (NEVER re-run)

- `development/loop9/runs/20260610_m0/recall_baseline.json` — recall@2048 gate reference (survives node move).
- loop-11 frozen serving ladder — DIRECTIONAL reference only; the locked sweep re-measures DSA on this node.
