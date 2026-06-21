# Ask Codex Input

## Question

You are doing a SECOND-PASS reasonability review of a candidate implementation plan for a curation/clean-port task in the SGLang repo. You reviewed an earlier draft (first pass). Now challenge the CANDIDATE PLAN below. It has already been corrected against verified code evidence (file:line) gathered by read-only investigation — do not relitigate facts that are stated as verified; instead attack the PLAN'S REASONING, completeness, ordering, and acceptance criteria.

Output STRICTLY in this format with these exact headers:
AGREE:
DISAGREE:
REQUIRED_CHANGES:
OPTIONAL_IMPROVEMENTS:
UNRESOLVED:

(AGREE = points accepted as reasonable. DISAGREE = points you consider unreasonable, with why. REQUIRED_CHANGES = must-fix before convergence. OPTIONAL_IMPROVEMENTS = non-blocking. UNRESOLVED = opposite opinions needing a human decision.)

=== CONTEXT (verified) ===
Task: extract the MINIMAL correct table-free Double Sparsity (DS) runtime from a messy dev branch (/sgl-workspace/sglang @ dev/double-sparsity-standalone) onto a fresh branch off clean main in a SECOND clone (/sgl-workspace/double-sparisty-v2/sglang @ main 105e095e0, zero DS refs today). Plus ONE perf eval. Prove parity: DS ~26.9 TPS / ~25.1s P99 TTFT at conc 64 — regression gate, NOT a new 30-TPS SLO. Method = ADDITIVE MINIMAL CLOSURE (copy new files, hand re-apply DS hunks onto current main, a file ships only if its removal breaks `import sglang` / server boot with DS / the conc-64 run). Exclude .pensieve/.humanize/development/SLOS.md and all dev scaffolding. Push only to fork Jiminator/sglang.

Verified by code inspection (treat as FACT):
- validator.py is RUNTIME (must ship): __init__.py:28 exports validate_double_sparsity; server_args.py:7215-7223 calls apply_radix_fixture_artifact + validate_double_sparsity unconditionally at DS startup. It validates config, loads/validates channel mask (safetensors SHA), gates unsupported modes (disagg, hierarchical cache, non-graph-safe selectors), and mutates server_args.
- calibrate.py is DEV-ONLY (zero runtime refs) -> droppable cleanly.
- Dropping dev modules needs ~900 lines of pruning: clean-delete radix_fixture_capture.py + calibrate.py; guard-removal for score_capture (selection_kernel.py:1186-1199), selection_capture (model_runner.py:3256-3263 + dsa_backend attr reads), latent_capture (deepseek_v2.py:2381-2392); ENTANGLED oracle_artifact_sink + selection_recall_oracle inside _maybe_record_recall_oracle() in selection_kernel.py (~125 lines, called from hot path ~line 1210) -> must delete the diagnostic function + its param from absorbed_topk_select() signature + update callers.
- DS-active signal is per-request meta_info["double_sparsity"] = {sparsity_rate, selected_tokens, total_tokens, dense_fallback, optional error_class/error_message}. Transport: deepseek_v2.py _publish_ds_request_summary -> metrics.py meta_info_for_request -> forward_batch.ds_per_request_summary -> model_runner.py:3240-3250 -> LogitsProcessorOutput.per_request_summary (logits_processor.py:111 field) -> batch_result_processor.py -> Req.per_request_summary (schedule_batch.py) -> tokenizer_manager.py:1799-1806 -> response. metrics.py meta_info_for_request/record_error/mark_channel_mask_valid are LIVE; record_selection is DEAD (prunable). output_streamer/detokenizer/multi_tokenizer do NOT carry DS code. So the only EXTRA modified file the draft missed is logits_processor.py (one dataclass field).
- v2 main DRIFT: cuda_graph_runner.py was REFACTORED AWAY -> logic now in model_executor/runner/{base,decode,prefill}_cuda_graph_runner.py + runner_backend/{base,full,breakable,tc_piecewise}_cuda_graph_backend.py (#23906, #28081). DS CUDA-graph hunks must be re-targeted. deepseek_v2.py source is +740 lines vs target (heavy DS forward integration). Abort/finisher API STABLE: set_finish_with_abort + update_finish_state both exist on target (schedule_batch.py:1539, 1403); loop-11b fix carries cleanly.
- Test floor: KEEP test_lifted_budget_decode.py as-is (zero scaffolding); EXTRACT slim runtime subset from test_double_sparsity_unit.py (it imports calibrate/selection_capture/radix_fixture_capture/validator-radix) into a new runtime-only test; DROP test_oracle_sink_and_force, test_selection_recall_oracle, test_ac12_helpers, test_ac11_comparator, all test/manual dsv32, test/registered/unit/development/*.
- Radix-cache-ON is FAIL-CLOSED in validator.apply_radix_fixture_artifact: needs a config-bound radix fixture STATE FILE (which lived under development/, now excluded) OR env SGLANG_DS_RADIX_OVERRIDE=1. So AC "radix cache on" collides with excluding development/.
- channel_mask.py validates safetensors metadata + content SHA; NO production GLM-5.1 mask artifact is committed in the repo.

=== CANDIDATE PLAN v1 ===
GOAL: Produce branch double-sparsity-v2 (or next free name) on Jiminator/sglang off current main, containing only the minimal DS runtime + one perf eval + minimal feature tests, reproducing loop-11b conc-64 parity, with DS provably active.

ACCEPTANCE CRITERIA (each with positive/negative tests):
- AC-1 Branch & diff hygiene: branch exists on fork off current main; `git diff main...branch` touches only DS runtime + one perf script + minimal feature tests (+ approved artifacts). NEG: any .pensieve/.humanize/development/SLOS.md/oracle/capture/calibration/comparator/test_dsv32_*/test/registered/unit/development/* present -> fail.
- AC-2 Exclusions absent: rg over branch finds zero excluded paths/files. POS: rg returns empty. NEG: a stray oracle/capture/calibrate import remains -> fail.
- AC-3 Import & prune closure: `import sglang` + DS package import clean; NO shipped runtime file imports a dropped module (oracle_artifact_sink, selection_recall_oracle, *_capture, calibrate, radix_fixture_capture). POS: grep finds no such import. NEG: leftover _maybe_record_recall_oracle reference -> fail.
- AC-4 validator ships & gates: validate_double_sparsity + apply_radix_fixture_artifact present and called at startup; negative gates fire for missing/corrupt mask, SHA mismatch, hierarchical-cache+DS, disagg+DS, non-graph-safe selector without --disable-cuda-graph. POS: each bad config rejected with clear error. NEG: bad mask silently accepted -> fail.
- AC-5 Server boot DS active: GLM-5.1-FP8 TP=8 dsa backend glm4_moe, FP8 KV, page 64, CUDA graphs ON, radix cache ON, NO expandable allocator. DS genuinely active: a decode response has meta_info["double_sparsity"] with selected_tokens>0, total_tokens>selected_tokens, dense_fallback==0. NEG: meta absent or total==selected (dense fallback) -> fail.
- AC-6 Abort path: injected DS per-request error drives set_finish_with_abort + update_finish_state in the same scheduler step (no reliance on pre-#25725 check_finished). POS: request finishes with abort state. NEG: hang/wrong finisher -> fail.
- AC-7 Perf parity: conc-64 GSP (sys 2253 / q 1843 -> ISL 4096 ~55% prefix, OSL 512, range-ratio 1.0), one trial, --backend sglang, fixed prompt count + seed; emits decode TPS (p50) + P99 TTFT within a written parity band around 26.9 TPS / 25.1s. POS: within band. NEG: outside band or perf script can't emit p50 decode TPS -> fail.
- AC-8 No dead code: every shipped DS module/symbol referenced from serve path or perf script (record_selection pruned). POS: dead-code sweep clean. NEG: unreferenced symbol remains -> fail.

PATH BOUNDARIES: upper = full DS runtime incl validator + lifted_budget + metadata signal + extracted slim tests + one perf script; lower = same minus lifted_budget IF parity run uses enable_lifted_budget_decode=false AND closure still holds. Fixed choices (deterministic): additive minimal closure (not merge/rebase); stock bench_serving stays unless DEC-3 picks B; push only to fork.

MILESTONES: M1 branch+blank-base; M2 copy new runtime files + prune dropped-module references; M3 re-apply modified-upstream hunks onto current main (re-target cuda_graph hunks to runner/runner_backend; add logits_processor field; carry abort fix); M4 import+prune closure green (import sglang, no dropped imports); M5 server boots DS-active on GLM-5.1-FP8; M6 perf eval reproduces parity; M7 dead-code sweep + extracted tests pass; M8 push to fork.

OPEN DECISIONS carried to user:
- DEC-1 calibration tooling: drop calibrate.py (runtime-only). Lean drop (verified clean). Couples to DEC-4.
- DEC-2 DS-active signal: keep meta_info["double_sparsity"] + startup bind logs (already runtime). Lean keep (resolved by evidence; metrics.py ships, record_selection pruned).
- DEC-3 perf-eval fidelity: A stock bench_serving with derived p50 decode TPS (from --output-details ITLs) vs B port window flags. Lean A.
- DEC-4 channel mask provenance: commit a sanitized GLM-5.1 mask artifact vs documented external path vs keep calibrate to regenerate. UNRESOLVED — blocks the eval.
- DEC-5 test floor: keep lifted_budget test + extract slim config/selector/validator test; drop rest. Lean resolved.
- DEC-6 (NEW) radix-fixture provenance / radix-on acceptance: radix-on needs a fixture state file that lived under development/. Options: (a) ship a sanitized fixture artifact + path, (b) document + use SGLANG_DS_RADIX_OVERRIDE for the parity run, (c) relax AC-5 to radix-off with documented delta. UNRESOLVED.

=== YOUR JOB ===
Challenge this plan. Is the closure/prune ordering right (e.g. should pruning happen before or after hunk re-application)? Is the parity band quantifiable enough? Are AC-5 / AC-6 testable as written? Did we miss a modified file or a closure hazard (e.g. sgl-kernel/triton build deps, deep_gemm, flash_mla)? Is collapsing DEC-2/DEC-5 to "resolved" justified, or should they stay user-facing? Are DEC-4 and DEC-6 the real blockers, or is one of them avoidable? Be concrete and terse.

## Configuration

- Model: gpt-5.5
- Effort: xhigh
- Timeout: 3600s
- Timestamp: 2026-06-18_11-06-18
- Tool: codex
