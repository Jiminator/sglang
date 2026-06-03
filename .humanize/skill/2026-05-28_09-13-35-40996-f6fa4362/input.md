# Ask Codex Input

## Question

You are doing a FIRST-PASS planning critique of a design draft for the SGLang repository (you are running inside the repo at its root; you may read files to ground your analysis).

# Repository context
- SGLang is a high-performance LLM serving engine. The active branch is `dev/double-sparsity-standalone`.
- This draft is "Loop 5": ship a demonstrable Double Sparsity (DS) MVP on a 2-node 8x H200 cluster serving DeepSeek-V3.2 (FP8). DS is an attention KV-sparsity scheme; DSA (native NSA) is the comparison baseline.
- Verified facts about current repo state (already confirmed by reading the code):
  - DS code lives in `python/sglang/srt/layers/attention/double_sparsity/` (calibrate.py, channel_mask.py, radix_fixture_capture.py, validator.py, token_label_write.py, etc.).
  - The DSA backend with the production write hook is `python/sglang/srt/layers/attention/dsa_backend.py`.
  - `_write_token_labels` is defined at dsa_backend.py:1501 with signature `(self, layer, cache_loc, k)` — it does NOT take `forward_batch`. Inside it (lines ~1572-1593) the env-gated M3-B radix capture branch references a local `forward_batch` that does not exist; the resulting NameError is swallowed by a `try/except Exception` that sets `is_extend=False`, so `_ds_radix_publish_extend_snapshot(...)` is NEVER called. record_write() still runs, but the extend snapshot never publishes. Call sites at dsa_backend.py:1664 (extend), :1863 (decode), :2387 (TRT-LLM) do not pass forward_batch.
  - The calibration artifact `/models/dsv32-fp8-channel-mask.safetensors` does NOT exist on disk (confirmed). The draft calls this the single root blocker.
  - `development/serve_double_sparsity.sh` passes `--disable-radix-cache` (line 69), required today by the DS validator's DEC-2 guard; an "AC-10-FIXTURE-MARKER" comment (line ~89) marks where to remove it for radix-on.
  - `development/benchmark.sh` defaults are `TRIALS=3`, `WARMUP_SECONDS=120`, `MEASUREMENT_WINDOW_S=600`, `CONCURRENCIES="16 32 64"`, and it has a HARD GUARD (line ~93) that refuses to publish a JSONL artifact if observed duration < MEASUREMENT_WINDOW_S. Output files are named `${MODE}_gsp_isl4096_osl512_c${C}_t${TRIAL}.jsonl`.
  - `development/benchmark_baseline.sh`, `development/benchmark_compare.py` (--baseline/--ds/--output), `test/manual/test_dsv32_quality_smoke.py`, and `test/manual/test_double_sparsity_v32.py` all exist.

# Your job
Critique this draft's assumptions, identify missing requirements, and propose stronger plan directions. Be concrete and repo-grounded. Pay special attention to any internal inconsistencies between the draft's narrative (e.g. "single-trial smoke, ~30 min") and what the scripts actually do by default (e.g. TRIALS=3, 600s measurement window with a hard publish guard).

# Output format (use these exact section headers)
CORE_RISKS:
- highest-risk assumptions and potential failure modes
MISSING_REQUIREMENTS:
- likely omitted requirements or edge cases
TECHNICAL_GAPS:
- feasibility or architecture gaps
ALTERNATIVE_DIRECTIONS:
- viable alternatives with tradeoffs
QUESTIONS_FOR_USER:
- questions that need explicit human decisions
CANDIDATE_CRITERIA:
- candidate acceptance criteria suggestions (each with a clear pass/fail condition)

# Raw draft content
---
# Loop 5 Draft — Double Sparsity MVP on H200

## Objective
Get a demonstrable Double Sparsity (DS) MVP running end-to-end on the 2-node H200 cluster as fast as possible, without confusing a hardware smoke milestone with the loop4-complete MVP.

Two deliverables:
1. Smoke MVP: DS-on DeepSeek-V3.2 (FP8) serves real requests on H200, produces non-trivial DS selection, has one DS benchmark JSON + one DSA benchmark JSON, and passes the paired quality smoke.
2. Loop4-compatible MVP: the smoke milestone plus loop4 requirements: TP=8, FP8 KV, page size 64, CUDA graphs represented, chunked prefill probed, radix cache enabled for the final run, DSA baseline captured with matching knobs, AC-11 comparator run, and AC-12 full quality gate run.
If AC-10 radix, AC-11 comparator, or AC-12 full quality are missing, the result is a useful smoke milestone, not the minimal viable working version requested by loop4.

## Why a new loop
Loop 4 built deep code-tier scaffolding but never executed against hardware. The critical artifact /models/dsv32-fp8-channel-mask.safetensors does not exist on disk. Generating it unblocks every DS-on AC.

## Hardware
- Node 0 (local): 8x H200, hostname h200-10-220-51-16, verified 8 GPUs x 143GB free.
- Node 1 (remote): 8x H200, hostname h200-10-220-51-5, access via `rx devbox run double-sparsity --rank 1 -- <cmd>`.
- DSv3.2 FP8 weights: /cluster-storage/models/deepseek-ai/DeepSeek-V3.2.
- Ports: workers 30001, router 30000, prometheus 29000.

## MVP scope — IN
0. Close the Round 38 AC-10 producer bug before claiming radix-on. `_write_token_labels` does not accept `forward_batch`, but the capture branch references it and hides the failure. Fix: update signature to accept `forward_batch: Optional[ForwardBatch] = None`; pass live forward_batch at extend, decode, TRT-LLM call sites; keep token-label writes first and publish radix capture only when forward_batch present and mode is extend; add producer-side regression; verify /generate exposes non-empty meta_info["double_sparsity_radix_capture"] when SGLANG_DS_RADIX_FIXTURE_CAPTURE=1.
1. Generate the channel mask (task-ac4-hwrun). Single GPU, --tp 1, ~15-30 min. Output /models/dsv32-fp8-channel-mask.safetensors. Validate shape=[L,H,16], dtype=fp8_e4m3, head_dim=128, page_size=64, label_dim=16.
2. DS boot smoke (task-ac1-hwtest). Launch serve_double_sparsity.sh on local 8x H200 TP=8 with the new mask; one /generate; confirm text returned and token-label table populates via SGLANG_DS_RADIX_FIXTURE_CAPTURE=1 -> meta_info with non-empty per_token_slot_sha and per_layer_written_all_true=True.
3. DSA + DS benchmark pair (task-ac8-server + task-ac9-baseline). DSA baseline: benchmark_baseline.sh Option B flags conc 16/32/64. DS run: benchmark.sh same operating point. Radix-off DS allowed only as smoke. Final loop4 run must close AC-10 and run both with radix on. Single trial allowed only for smoke; final comparable-performance run uses AC-11 shape: conc 16/32/64, 3 trials, 120s warmup, 600s measurement, median comparison.
4. Quality smoke (task-ac8-quality). Boot both servers on different ports; test_dsv32_quality_smoke.py compares DS-on vs DSA on 20 deterministic prompts. Gates: prefix-match >=0.80, ROUGE-L >=0.85, NIAH-mini 4/5.

Smoke MVP = one DS bench JSON, one DSA bench JSON, one quality smoke artifact, side by side.

## Smoke-only items NOT enough for loop4 MVP
- AC-10 radix-cache flip (run M3-B fixtures, prove producer capture, flip guard, remove --disable-radix-cache, final comparator radix on).
- AC-11 directional comparator (3-trial DSA+DS sweep conc 16/32/64, 120s warmup, 600s measurement, medians, DS TPS within 5% of DSA, DS P99 TTFT no worse than 1.10x DSA).
- AC-6 CUDA-graph capture validation (record whether CUDA graphs enabled; clear exception if capture cannot be used).
- AC-1b chunked-prefill probe (run and record; if it passes keep default, if it fails disable on both DS and DSA and file follow-up).
- AC-12 full NIAH 4K/16K/64K + MMLU 5-shot.

## Acceptance evidence
A single directory /sgl-workspace/sglang/runs/<date>_dsv32_mvp/ with: calibrate.log + mask validation output; serve_*.log for DS and DSA; branch+commit SHA; full server args from /get_server_info; knob evidence (TP, kv_cache_dtype=fp8_e4m3, page_size=64, CUDA graph status, radix status, chunked-prefill, DS config path, mask hash, overlap/piecewise status); six bench JSONLs native_nsa_*c{16,32,64}_t1.jsonl and double_sparsity_*c{16,32,64}_t1.jsonl plus .meta.json sidecars; mvp_compare.md; dsv32_quality_smoke_*.json. Final loop4 evidence when claiming complete: radix-on launch evidence; AC-11 3-trial artifacts + pass/fail; AC-12 NIAH+MMLU artifacts + pass/fail.

## Risks
1. Calibrate OOMs at TP=1 -> bump to TP=2 --gpus 0,1.
2. DS server fails validator DEC-2 guard -> launcher already passes --disable-radix-cache so guard accepts; read validator error verbatim if it fails on mask hash/page size.
3. bench_serving crashes on DS selection -> would mean production _write_token_labels hook buggy on hardware; boot smoke catches this.
4. Quality smoke prefix-match << 0.80 -> bad DS labels or prompts too short/sensitive.
5. TPS gap > 5% -> acceptable for smoke, not for loop4 MVP unless reported as AC-11 failure.

## Loop-runner notes
- Single mainline objective per round: the next concrete command from the critical path. No multi-day fixture refactors.
- Each round produces ARTIFACTS in runs/<date>_dsv32_mvp/, not just code changes.
- Existing loop-4 code stays as-is unless a specific bench failure mode requires patching. The Round 38 AC-10 producer bug is the one known exception that must be patched before a radix-on/default-cookbook parity claim.
---
End of draft. Produce your critique now.

## Configuration

- Model: gpt-5.5
- Effort: xhigh
- Timeout: 3600s
- Timestamp: 2026-05-28_09-13-35
- Tool: codex
