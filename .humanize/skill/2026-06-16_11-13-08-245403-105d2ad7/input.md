# Ask Codex Input

## Question

You are doing a SECOND-PASS reasonability review of a CANDIDATE IMPLEMENTATION PLAN (not the draft).
Do NOT write code. Decide whether the plan is reasonable and converged, or needs changes.

## Read these
- The candidate plan: `development/loop11b/.plan_candidate_v1.md` — READ IT IN FULL.
- The original draft: `development/loop11b/draft.md`.
- Your own first-pass critique was already incorporated; it is at
  `development/loop11b/.codex_pass1_out.md`. Verify the plan actually addressed it.
- Ground truth files if you need them: `python/sglang/srt/layers/attention/double_sparsity/validator.py`,
  `channel_mask.py`, `calibrate.py`; `development/benchmark_compare.py`, `benchmark.sh`,
  `development/serve_double_sparsity.sh`, `development/SLOS.md`,
  `development/serve_double_sparsity_radix_fixture.json`, `python/sglang/srt/server_args.py`.

## Context (already established + verified in-repo)
- This is loop 11b: finish loop 11 (closed M0–M3) by re-establishing the serving op-point on a fresh
  8×H200 node, running task8 (per-step tax guard) + task9 (locked AC-11 sweep) to HARD verdicts vs
  native DSA, delivering the DS-vs-DSA SLO comparison per SLOS.md, and productionizing DS UX.
- Verified: the radix fixture hashes the FULL mask file (which embeds a fresh `created_at`) and pins
  `channel_mask_path` with exact fail-closed match → mask SHA cannot be reproduced → re-mint is the
  default. A tensor-only `compute_content_sha256` exists (channel_mask.py:101-115).
- Verified: `benchmark_compare.py --ac11` gates per-request decode-TPS p50 (≥0.95×), P99 TTFT
  (≤1.10×), and the DS-column absolute 30 TPS / 22 s; it does NOT gate aggregate throughput and only
  OBSERVES `dense_fallback_total`; it IGNORES mem_fraction cross-side (DS 0.8 vs DSA 0.85 sanctioned)
  but ENFORCES radix-state match; `benchmark.sh` reuses the same per-conc seed across all 3 trials.
- Verified UX surface: ~10 safe doc/default fixes (Cat A), 2 CLI-help-text edits (Cat B,
  server_args.py:6090/6103), zero required ABI changes (Cat C).

## Your job
Challenge the candidate plan hard. Is each AC well-defined, verifiable, and fairly scoped? Are the
task breakdown, milestones, path boundaries, and the 6 pending decisions (DEC-1..DEC-6) correct and
complete? Did the plan correctly resolve your pass-1 points? Is anything over- or under-specified, or
internally inconsistent? Are there NEW risks the plan still misses (e.g. calibration cost/OOM on the
fresh node, the GLM calibration corpus availability, server_info field names, the absorbed-latent
recall baseline being GLM-5.1/fp16 vs the served fp8 config, AC-9 pairing feasibility with one TP=8
server at a time)? Keep it tight and decision-oriented.

## Required output format (use these exact headers)
AGREE:
- (points accepted as reasonable)
DISAGREE:
- (points considered unreasonable, with why)
REQUIRED_CHANGES:
- (must-fix items before convergence — be specific about the AC/task/DEC to change and how)
OPTIONAL_IMPROVEMENTS:
- (non-blocking improvements)
UNRESOLVED:
- (opposite opinions that need an explicit human decision — map to DEC-N where possible)

## Configuration

- Model: gpt-5.5
- Effort: xhigh
- Timeout: 3600s
- Timestamp: 2026-06-16_11-13-08
- Tool: codex
