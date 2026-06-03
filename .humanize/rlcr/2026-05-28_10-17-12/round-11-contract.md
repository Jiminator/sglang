# Round 11 Contract

## Mainline Objective (exactly one)
Close TIER 2 by **executing the AC-12 hard quality gate (task14) on real hardware**
and recording an honest pass/fail, then **assembling the task15 evidence bundle**.
AC-12 is the last unrun acceptance criterion; task15 is its dependent assembly step.
Single coherent objective: *finish the loop4-compatible MVP gate and publish the
complete evidence bundle*.

## Target ACs
- **AC-12** (primary): NIAH 4K/16K/64K + MMLU 5-shot via `test/manual/test_double_sparsity_v32.py`.
  HARD gate — NIAH within 5 pp of DSA at each length AND MMLU 5-shot within 1.0 pp of DSA.
- task15 evidence bundle targets AC-0/AC-4/AC-1/AC-1.1/AC-1b/AC-6/AC-8/9/AC-10/AC-11/AC-12/AC-Q
  (assembly only — no new AC).

## Execution Plan (per Codex Round-10 directive)
1. **DS server on node 0** (local — holds the calibrated mask `/models/dsv32-fp8-channel-mask.safetensors`
   + radix fixture `runs/20260528_dsv32_mvp/ds_radix_fixture_state.json`). `serve_double_sparsity.sh`
   with `RADIX_FIXTURE_ARTIFACT=...`, TP=8, fp8 KV, page 64, chunked-prefill 8192, Option-B graph flags.
   Reached by the harness via `localhost`.
2. **DSA baseline server on node 1** (remote via `rx devbox run --rank 1`; node 1 has the repo +
   `/cluster-storage` model but NOT the node-0-local mask/fixture, which DSA does not need).
   `serve_native_nsa.sh`, radix-on (default), mem 0.85. Must bind `--host 0.0.0.0` so node 0 can
   reach it — `serve_native_nsa.sh` currently has no host knob, so add one (see Blocking #B1).
3. Capture `/get_server_info` from both into `runs/20260528_dsv32_mvp/ac12_{ds,dsa}_server_info.json`;
   preserve both boot logs.
4. MMLU data pre-staged out-of-repo at `/root/ac12_mmlu_data/{dev,test}` (57+57 subjects) →
   `AC12_MMLU_DATA_DIR=/root/ac12_mmlu_data`. MMLU must NOT skip (Codex directive #3).
5. Run `PYTHONPATH=python DS_BASE_URL=http://localhost:<p> DSA_BASE_URL=http://10.220.51.5:<p>
   AC12_NIAH_NUM_PROMPTS=20 AC12_MMLU_NUM_EXAMPLES=200 python -m pytest
   test/manual/test_double_sparsity_v32.py -v`.
6. Copy every `development/results/ac12_*.json` into `runs/20260528_dsv32_mvp/ac12_results/`, record
   exact env/command, summarize pass/fail in `runs/20260528_dsv32_mvp/ac12_analysis.md`.
7. **Honesty rule (Codex):** if NIAH 64K (or 16K) fails because of the known top_k-bounded recall
   limit (BL-...-ds-longcontext-needle-recall-vs-topk), publish a **HARD AC-12 failure with
   evidence — do NOT reclassify as directional.** AC-12 is hard pass/fail (DEC-7 directional
   handling applies only to AC-11, not AC-12). If AC-12 fails, the bundle states the
   loop4-compatible MVP is **not** complete (a smoke milestone + recorded quality gap).
8. task15: write `runs/20260528_dsv32_mvp/evidence_bundle.md` (AC-by-AC table + artifact paths,
   raw-JSONL locations, mask provenance/SHA, server args/server_info, CUDA-graph + chunked-prefill
   status, radix fixture, AC-10 label-capture provenance note; AC-11 stated as
   "executed; directional TTFT/TPS target missed; #F admission caveat + follow-up filed").

## Blocking Side Issues (truly block the mainline)
- **#B1: `serve_native_nsa.sh` has no host-binding knob.** DSA on node 1 must bind `0.0.0.0`
  for the node-0 harness to reach it; the launcher only forwards the default 127.0.0.1.
  Resolution: add a `HOST` env knob (default 127.0.0.1) to `serve_native_nsa.sh` (and, for
  symmetry, `serve_double_sparsity.sh`), keep the locked Option-B flags inside the script, and
  keep `test/registered/unit/development/test_option_b_scripts.py` green. Minimal enabling change.

## Queued (explicitly OUT of scope this round)
- **Comparator per-side `mem_fraction_static` validation hole** (Codex queued #1): keep it ignored
  cross-side (DSA 0.85 vs DS 0.6) but compare it within each side. Fix only when the comparator is
  next touched — AC-12 does not touch the comparator. Documented, not done this round.
- **AC-11 directional-miss performance follow-up** (TokenLabelTable/KV-budget, conc-64 admission
  profile): performance work, not next-round mainline (Codex queued #2). Filed in `ac11_analysis.md`.
- **Stale `calibrate.py` operator-recipe docstring**: cleanup; fold into task15 hygiene if cheap,
  else remains queued.

## Success Criteria
1. Both servers boot at the locked Option-B point (DS radix-on via fixture artifact, no env
   override; DSA radix-on); `/get_server_info` captured for both; boot logs preserved.
2. All four AC-12 tests execute (no silent skip); `ac12_niah_{4096,16384,65536}.json` +
   `ac12_mmlu_5shot.json` artifacts produced and copied to `runs/.../ac12_results/`.
3. Honest verdict recorded in `ac12_analysis.md` — each gate's DS vs DSA delta vs threshold;
   any failure published as a hard AC-12 failure with evidence (no directional reclassification).
4. `evidence_bundle.md` assembled (task15), stating loop4-MVP complete only if AC-12 fully passes;
   otherwise stating it is a smoke milestone + recorded AC-12 quality gap.
5. CPU regression suite still green after the launcher knob change.
6. Goal-tracker mutable section updated; commit (NO AI authorship) + push.

## Out-of-Scope Confirmation
This round changes no immutable AC and no AC-12 threshold. It does not re-run AC-11, does not
tune DS performance, and does not modify the comparator.
