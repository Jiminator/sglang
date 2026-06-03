# Ask Codex Input

## Question

You are doing a SECOND-PASS reasonability review of a candidate implementation plan for the SGLang repo (you are running inside the repo; read files to ground judgments). A first Codex pass already produced findings that this candidate incorporates. Your job now: judge whether the candidate plan is REASONABLE and converged, or whether required changes remain.

Context recap (verified by reading the code):
- calibrate.py:268 loads the full model on one cuda device (device_map={"":"cuda"}); --tp is informational. V3.2 671B in bf16 ~1.3TB does not fit on 1 (or 8) H200. This is captured as DEC-1.
- _write_token_labels (dsa_backend.py:1501) references undefined forward_batch in the capture branch; fix is AC-0.
- benchmark_compare.py enforces disable_radix_cache match (refuses radix mismatch) — captured in AC-11 and DEC-4.
- serve scripts are single-node TP=8, MODEL_PATH defaults to HF id (DEC-2, DEC-6).
- benchmark.sh defaults TRIALS=3/120s/600s with a hard publish guard (DEC-3, AC-8/9).

Output format (use these EXACT headers):
AGREE:
- points accepted as reasonable
DISAGREE:
- points considered unreasonable and why
REQUIRED_CHANGES:
- must-fix items before convergence (be specific; if none, write "none")
OPTIONAL_IMPROVEMENTS:
- non-blocking improvements
UNRESOLVED:
- opposite opinions that genuinely need a human decision (not just questions already listed as DEC-*)

Focus on: (a) Are the two tiers and their AC partition correct and non-overlapping? (b) Are any ACs missing positive/negative tests that matter? (c) Is anything in the candidate factually wrong about the repo? (d) Are the DEC-* the right set, or is one of them actually resolvable from the code without asking the user? Keep it tight.

--- CANDIDATE PLAN v1 ---
# Loop 5: Double Sparsity MVP on H200 — Candidate Plan v1

## Goal
Ship a demonstrable Double Sparsity (DS) MVP on DeepSeek-V3.2 (FP8) on the H200 cluster, in two explicit tiers:
- TIER 1 SMOKE MVP: DS-on serves real requests, produces *genuinely sparse* (non-trivial) selection, paired DS+DSA benchmark artifacts at a clearly-labeled smoke shape, and a passing paired quality smoke.
- TIER 2 LOOP4-COMPATIBLE MVP: smoke + radix-on final run (AC-10), AC-11 3-trial directional comparator, AC-12 full quality gate, CUDA-graph status recorded (AC-6), chunked-prefill probed (AC-1b).
Prerequisites: fix the Round-38 AC-10 producer bug (AC-0); resolve the calibration feasibility blocker BEFORE claiming mask generation (AC-4).

## Acceptance Criteria (draft AC numbering preserved for code-marker traceability)

AC-0 — Producer-bug fix. `_write_token_labels` accepts `forward_batch`; extend/decode/TRT-LLM call sites pass it; token-label write stays first; radix capture publishes the extend snapshot only when forward_batch present and mode is extend; producer-side regression added.
  POS: capture-enabled `/generate` returns non-empty meta_info["double_sparsity_radix_capture"] with per_token_slot_sha set and per_layer_written_all_true=True, no capture error. POS: pytest producer regression passes.
  NEG: with capture disabled, meta_info has no radix_capture key. NEG: a decode-only forward must NOT publish/overwrite the extend snapshot.

AC-4 — Channel mask generated + validated. Output /models/dsv32-fp8-channel-mask.safetensors readable by every DS process; SHA recorded.
  POS: load_channel_mask() succeeds; metadata dtype="fp8_e4m3", page_size=64, label_dim=16, head_dim=128; channel_selection int32 [L,H,16]; content SHA recorded.
  NEG: load rejects a mask with wrong dtype/page_size/out-of-range channel index. NEG: a synthetic/placeholder mask (all-zero or random) must be rejected by validation as not calibration-derived.
  BLOCKER NOTE: calibrate.py loads the full model on ONE cuda device in bf16/fp16 (device_map={"":"cuda"}, --tp informational). V3.2 671B ≈ 1.3TB bf16 — infeasible on one or even 8 H200 (1.14TB). Mask generation as documented cannot run; resolution required (DEC-1).

AC-1 — DS boot smoke. serve_double_sparsity.sh boots on TP=8 with MODEL_PATH=/cluster-storage path + the new mask; /generate returns text; token-label table populates via production hook.
  POS: /get_server_info shows DS enabled, TP=8, kv_cache_dtype=fp8_e4m3, page_size=64, radix setting; /generate returns non-empty text.
  NEG: a missing/invalid mask makes the validator reject boot with a verbatim error (not a silent dense fallback).

AC-1.1 — Non-trivial DS selection. A prompt longer than top_k proves genuine sparsity.
  POS: meta_info["double_sparsity"] shows selected_tokens < total_tokens and dense_fallback==0 on a long prompt.
  NEG: an all-tokens-selected result on a long prompt fails (would mean DS is effectively dense).

AC-1b — Chunked-prefill probe, recorded BEFORE the benchmark sweep. If it fails, disable chunked-prefill on BOTH DS and DSA and file follow-up.
  POS: probe result recorded in run dir; if disabled, both DS and DSA sidecars show chunked_prefill_size=-1. NEG: benchmarks collected under mismatched chunked-prefill settings between DS and DSA.

AC-6 — CUDA-graph status recorded. Final bundle records capture/replay success OR a clearly-recorded exception explaining why capture cannot be used. POS: run dir has explicit cuda_graph status. NEG: only `disable_cuda_graph=False` recorded with no capture/replay evidence is insufficient.

AC-8 / AC-9 — DS + DSA benchmark artifacts. Smoke = explicit TRIALS=1 (+ shortened MEASUREMENT_WINDOW_S, clearly labeled non-AC-11). Each JSONL meets its configured duration window; .meta.json sidecars present.
  POS: configured-count JSONLs produced; each duration >= configured window; sidecars valid. NEG: benchmark.sh hard guard refuses to publish any JSONL whose duration < MEASUREMENT_WINDOW_S.

AC-10 — Radix-cache flip (TIER 2). Both fixtures pass (label capture + FP8 scale stability), guard flipped, --disable-radix-cache removed, DS boots radix-on WITHOUT an env override.
  POS: DS server boots radix-on; both fixtures pass; final comparator runs with radix on. NEG: radix-on boot still requiring SGLANG_DS_RADIX_OVERRIDE fails AC-10.

AC-11 — Directional comparator (TIER 2). 3-trial DSA+DS sweep conc 16/32/64, 120s warmup, 600s measurement, medians; DS TPS within 5% of DSA; DS P99 TTFT <= 1.10x DSA; radix settings MATCH (comparator refuses mismatch).
  POS: 9 DSA + 9 DS JSONLs each duration>=600, sidecars valid, radix match, comparator exits 0, TPS/TTFT gates pass. NEG: comparator refuses to publish when disable_radix_cache differs between sides.

AC-12 — Full quality gate (TIER 2). NIAH 4K/16K/64K + MMLU 5-shot via test_double_sparsity_v32.py. POS: all gates pass. NEG: any gate below threshold fails.

Quality smoke (TIER 1): test_dsv32_quality_smoke.py compares DS-on vs DSA on 20 deterministic prompts. Gates prefix-match>=0.80, ROUGE-L>=0.85, NIAH-mini 4/5. Requires two servers reachable; on a single 8-GPU node two TP=8 servers cannot co-reside (topology decision DEC-2).

## Path Boundaries
UPPER: both tiers complete — AC-0/4/1/1.1/1b/6/8/9 + AC-10/11/12 — with full evidence bundle in runs/<date>_dsv32_mvp/.
LOWER: TIER 1 smoke only — AC-0, AC-4 (real calibrated mask), AC-1+AC-1.1, one labeled-smoke DS bench, one matching DSA bench, one quality-smoke artifact, side-by-side. Explicitly labeled "smoke milestone, not loop4 MVP".
ALLOWED: single-node-sequential OR two-node serving (DEC-2); smoke window may be shortened if labeled non-AC-11 (DEC-3). CANNOT: claim loop4 MVP while AC-10/11/12 missing; publish a comparator across mismatched radix settings; use a synthetic mask.

## Key dependencies / sequence
M1 Unblock: AC-0 producer fix -> resolve DEC-1 calibration feasibility -> AC-4 mask. M2 Smoke: AC-1 boot -> AC-1.1 sparsity -> AC-1b probe -> AC-8/AC-9 smoke bench -> quality smoke -> smoke comparator (radix-off both sides). M3 Loop4: AC-10 radix flip -> AC-11 3-trial -> AC-6 cuda-graph evidence -> AC-12 quality gate.

## Open decisions for the user (consolidated)
DEC-1 Calibration feasibility: calibrate.py cannot load V3.2 on one GPU in bf16. Are code changes to the calibration path (FP8/sharded/CPU-offload/layerwise streaming) in scope, or is there an alternate mask source / pre-existing mask?
DEC-2 Topology: single-node TP=8 sequential (DS then DSA, with stored DSA references for quality) OR true 2-node/16-GPU (DS node0 + DSA node1 simultaneously)? Scripts are single-node today.
DEC-3 Smoke bench shape: explicit TRIALS=1 + shortened MEASUREMENT_WINDOW_S for smoke, or run the full default shape?
DEC-4 Smoke comparator radix: compare radix-off DS vs radix-off DSA for the smoke (requires DSA baseline also radix-off), or defer the comparator until AC-10 is closed?
DEC-5 AC-10 flip mechanism: how to wire record_radix_fixture_passed before validate_double_sparsity — CLI flag, launcher init module, or env override? Codex wants final radix-on boot without an env override.
DEC-6 MODEL_PATH: both serve scripts default to HF id deepseek-ai/DeepSeek-V3.2; must MODEL_PATH be pinned to /cluster-storage/models/deepseek-ai/DeepSeek-V3.2 on every node?
Quantitative metrics to confirm hard-vs-directional: prefix-match>=0.80, ROUGE-L>=0.85, NIAH-mini 4/5, DS TPS within 5%, DS P99 TTFT <=1.10x.

## Configuration

- Model: gpt-5.5
- Effort: xhigh
- Timeout: 3600s
- Timestamp: 2026-05-28_09-21-04
- Tool: codex
