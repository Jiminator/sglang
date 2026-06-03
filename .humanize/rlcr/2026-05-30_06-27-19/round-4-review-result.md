# Round 4 Review Result

Mainline Progress Verdict: ADVANCED

ACs: 3/10 met, 2 partial, 5 not met, 0 deferred | Forgotten items: 0 | Unjustified deferrals: 0

Read first as required: `development/loop6/refined_plan_v1.md`. Also read the Pensieve review pipeline and taste-review guidance, Round 0-3 summaries/reviews, `goal-tracker.md`, commit `75e68053f`, the AC-4 artifacts under `runs/20260530_dsv32_loop6/memfraction_sweep_int8*`, `development/CLIENT_SLOS.md`, `development/roadmap.md`, and the relevant docs for V3.2/DSA backend behavior (`docs/basic_usage/deepseek_v32.md`, `docs/advanced_features/attention_backend.md`, `docs/references/environment_variables.md`).

## Part 1 — Goal Tracker Audit

| AC | Status | Evidence if met | Blocker if not met | Justification if deferred |
|----|--------|-----------------|--------------------|---------------------------|
| AC-1 | MET | `runs/20260530_dsv32_loop6/ds_on_v32_decision.md`; verified in R0/R3 history. | - | - |
| AC-2 | MET | `runs/20260530_dsv32_loop6/footprint_feasibility.md`; R1 revision makes int8 same-`label_dim` the binding lever and names `top-k overlap@2048 >= 0.99`. | - | - |
| AC-3 | MET | Commits `84d3410b9`, `e85cd2564`, `5d8e47fb3`, `8a05b1688`; unit/GPU tests, scale-sidecar consumers, decode microbench, launcher `SIGNATURE_DTYPE`, and real-mask NIAH non-regression all verified by R3 review. | - | - |
| AC-4 | PARTIAL | The sweep strongly advances AC-4: `max_total_num_tokens` rises 53056 -> 396096 -> 739200 attempted; 0.7 boots with `dtype=torch.int8`, table 6.48 GB, KV 17.73 GB, post-graph headroom 17.56 GB; local ignored NVML CSV shows a plateau. | Durable acceptance artifact is incomplete: committed artifacts lack torch `memory_reserved`/`memory_allocated` residual for the successful boots, explicit accounting for `written` / score scratch / FlashMLA metadata, and tracked stress-run proof for the 97/97 + 30K `/generate` no-OOM claim. | - |
| AC-5 | NOT MET | - | Full `NUM_PROMPTS=320` client SLO benchmark at conc 16/32/64 with radix-on proof, strict `<22.0` TTFT, `>=30` TPS/req, trial rule, and admission/prefill attribution has not run. | - |
| AC-6 | PARTIAL | Dev checks exist: compact flag parsing, fp16 default/int8 opt-in, DSA-default/no-table unit regression. | Hardware product proof remains pending: DSA-default boot/no DS table and DSA-default SLO unchanged; DS opt-in proof at the lifted point. | - |
| AC-7 | NOT MET | - | AC-11 DS+DSA 3-trial re-sweep at the lifted point has not run. | - |
| AC-8 | NOT MET | - | ~70K-token lifted-mem servability probe has not run; Round 4's 30K long request does not satisfy AC-8. | - |
| AC-9 | NOT MET | - | `test/manual/test_double_sparsity_v32.py` still needs real `usage.prompt_tokens` budget assertion plus live re-run. | - |
| AC-10 | NOT MET | - | Correctly gated behind full Tier-1 spine (AC-3 through AC-9). | - |

Forgotten items detection:
- No original-plan task is absent from Active, Completed, or Deferred.
- One completion claim is not verified: Round 4 summary/tracker claimed AC-4 done, but the durable evidence is not enough for full AC-4 verification.

Deferred items audit:
- Tracker has no explicit deferred items. This is correct. AC-10 is gated/pending, not deferred. DEC-5 multi-node work remains downstream roadmap scope and does not contradict the single-node TP=8 Loop 6 goal.

Goal completion summary:
```text
Acceptance Criteria: 3/10 met (0 deferred)
Active Tasks: 7 remaining (task5 partial + task6-task11)
Estimated remaining rounds: 5+ (AC-4 evidence addendum, AC-5, AC-6/AC-9, AC-7/AC-8, gated AC-10 or explicit later-loop disposition)
Critical blockers: AC-4 evidence addendum before AC-5; AC-5 benchmark/attribution is still the headline gate
```

## Part 2 — Mainline Drift Audit

The current round's objective was clear and singular: validate the compact int8 table's mem-fraction lift and no-generation-OOM behavior. It served the original footprint -> admission -> SLO spine and did not drift into side work.

The result advances the mainline: the 0.7 operating point admits 396096 tokens, about 3.5x the AC-2 conc-64 target, and the 0.8 failure is a boot-time cuda-graph-capture ceiling, not evidence that the table lever failed. Per the plan's "not `mem_fraction_static=0.8` as a number" framing and "0.7 acceptable as a conservative first step" language, no page-level/two-stage escalation is triggered by this result.

```text
Mainline Progress Verdict: ADVANCED
Blocking Side Issues: 1
Queued Side Issues: 1
```

Blocking side issue:
- AC-4 evidence is not durable/complete enough to verify task5 and gate AC-5.

Queued side issue:
- The committed AC-4 text artifacts have trailing whitespace, causing `git diff --check 8a05b1688..75e68053f` to fail. This is branch hygiene, not an AC blocker.

## Part 3 — Implementation Review

1. **AC-4 cannot be marked verified from the committed artifacts yet.**

   Evidence:
   - `runs/20260530_dsv32_loop6/memfraction_sweep_int8/mf_0.7.txt` records table/KV/avail/NVML and `/get_server_info`, but does not record `torch.cuda.memory_reserved()` / `memory_allocated()` residual for the successful run.
   - AC-4 requires full HBM budget logging including NVML plus torch reserved/allocated residual, and named components including `written`, score scratch, FlashMLA metadata, CUDA-graph pool, and headroom. The report gives the major components but not the required residual breakdown.
   - The OOM entry in `mf_0.8.txt` includes PyTorch allocated/reserved numbers only because the exception printed them; the successful 0.6/0.7 boots do not.

   Required action: add a tracked AC-4 addendum under `runs/20260530_dsv32_loop6/` with per-fraction torch reserved/allocated numbers and either explicit named components or a clearly labeled residual bucket for `written`/scratch/FlashMLA metadata.

2. **The no-generation-OOM / no-monotonic-growth proof is mostly summarized, not durably auditable.**

   Evidence:
   - `memfraction_sweep_int8.md` says the stress completed 97/97 requests plus one ~30K request with no OOM.
   - `git ls-files runs/20260530_dsv32_loop6/memfraction_sweep_int8` does not include `nvml_timeseries_0.7.csv`.
   - `git status --ignored runs/20260530_dsv32_loop6/memfraction_sweep_int8` reports `!! .../nvml_timeseries_0.7.csv`; `.gitignore` ignores `*.csv`.
   - The local ignored CSV has 29 samples and does plateau at 1,049,104 MiB total used / 11,947 MiB min free, so the claim is plausible now, but it is not part of the committed evidence bundle.
   - There is no tracked request/result log or post-stress server-log excerpt showing the 97/97 + 30K `/generate` result and no generation-time OOM.

   Required action: track or embed the NVML time series as text, and add a request/result plus server-log excerpt for the sustained stress. No rerun is required if the logs still exist and can be copied into a tracked artifact.

3. **No code regression found in Round 4.**

   Round 4 changed only acceptance artifacts. The prior code-review blockers remain resolved: `serve_double_sparsity.sh` exposes/logs `SIGNATURE_DTYPE`, the int8 boot excerpts show `dtype=torch.int8 scales=float16`, and the Tier-1 ABI lock remains intact (`python/sglang/srt/layers/attention/dsa_backend.py` still asserts `indices.shape[-1] == self.dsa_index_topk`). The V3.2 docs confirm DSA is the native sparse backend, and `development/CLIENT_SLOS.md` confirms the client target that AC-5 still needs to measure.

## Part 4 — Goal Tracker Update

Updated `.humanize/rlcr/2026-05-30_06-27-19/goal-tracker.md` mutable section:
- Plan version moved to Round 4 Review.
- Added an R4-review plan-evolution row rejecting AC-4 as verified until the evidence addendum lands.
- Left task5 active as `partial (R4) — evidence addendum required`.
- Added a blocking side issue for the incomplete AC-4 artifact.
- Left AC-5/task6 gated on task5; did not move AC-4 to Completed and Verified.

## Part 5 — Progress Stagnation Check

Development is not stagnating. The repeated issues from R1/R2 (premature deferral, missing scale-sidecar consumers, launcher unable to select int8) were addressed in later rounds. R4 produced meaningful hardware evidence and moved the mainline from AC-3 into AC-4. The current blocker is an evidence-packaging/acceptance-completeness gap, not a circular technical failure.

## Action Items

Mainline Gaps:
1. Add a tracked AC-4 evidence addendum with complete HBM residual accounting and durable sustained-generate proof.
2. After task5 is verified, run AC-5 exactly as planned: `NUM_PROMPTS=320`, conc 16/32/64, radix-on proof, strict `<22.0` TTFT, `>=30` TPS/req, all-trials/worst-trial disclosure, and measured admission-wait vs prefill-compute attribution.

Blocking Side Issues:
1. AC-4 evidence addendum blocks task5 verification and therefore blocks starting AC-5.

Queued Side Issues:
1. Clean trailing whitespace in `runs/20260530_dsv32_loop6/memfraction_sweep_int8/mf_0.6.txt`, `mf_0.7.txt`, and `mf_0.8.txt` so `git diff --check` passes before final branch hygiene.

Validation Performed:
- Inspected commit `75e68053f` and all committed AC-4 artifact files.
- Verified the local ignored NVML time series plateaus, but confirmed it is not tracked because `*.csv` is ignored.
- Verified `git diff --check 8a05b1688..75e68053f` fails only on trailing whitespace in AC-4 text artifacts.
- Did not rerun hardware; this review is artifact/code-history based.

NOT COMPLETE
