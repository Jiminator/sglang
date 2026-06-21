CORE_RISKS:
- Minimal closure is not just file selection. `double_sparsity/__init__.py` imports `validator.py`; `server_args.py` calls `validate_double_sparsity`; dropping `validator.py` breaks runtime.
- Dropping oracle/capture files requires code pruning: `selection_kernel.py` imports `oracle_artifact_sink` on an eager path, and `model_runner.py` imports `selection_capture` whenever DS is enabled.
- Radix cache ON currently depends on `development/serve_double_sparsity_radix_fixture.json`, which the draft excludes. `validator.py` rejects radix unless a matching artifact is provided or `SGLANG_DS_RADIX_OVERRIDE=1`.
- Channel mask provenance can block the eval. `channel_mask.py` validates safetensors metadata and content SHA; no production mask artifact is in the repo, and dropping `calibrate.py` removes regeneration tooling.
- “DS genuinely active” depends on metadata plumbing the draft omits: `layers/logits_processor.py`, `managers/scheduler_components/output_streamer.py`, `managers/detokenizer_manager.py`, and `managers/multi_tokenizer_mixin.py`.
- v2 main has drift: `server_args.py` structure moved, and `model_executor/cuda_graph_runner.py` appears absent/renamed in the clean clone. Hand hunks are high risk.
- Stock `bench_serving.py` lacks dev-branch steady-state/window flags and decode TPS p50 output. The perf script must define how TPS is derived.
- “Closure check after each addition” is likely impractical on shared 8xH200: server boot plus conc-64 perf is expensive and noisy.

MISSING_REQUIREMENTS:
- Explicit import/dependency closure: top-level imports, config-gated imports, CUDA graph paths, and metadata transport files.
- Explicit DS-active proof: not just `--enable-double-sparsity`; require bound selectors, no placeholder, no dense fallback, and request-level selected/total token evidence.
- Quantified parity band: define prompt count, seed, accepted TPS/TTFT range, repeat count or one-shot tolerance.
- Runtime artifact policy for radix fixture and channel mask, including SHA/fingerprint validation.
- Abort-path regression guard: prove DS error handling calls `req.update_finish_state()` and does not rely on older `check_finished` behavior.
- Config behavior for removed diagnostics: decide whether `recall_oracle`, `selection_capture`, `latent_capture`, and `score_capture` are rejected, ignored, or supported.
- Test floor must be extracted, not copied. `test_double_sparsity_unit.py` imports calibrate/capture/oracle/development helpers.
- Negative absence checks for excluded paths/files: `.pensieve`, `.humanize`, `development`, `SLOS.md`, oracle/capture/calibration/comparator/manual dsv32 tests.

TECHNICAL_GAPS:
- `validator.py` is runtime, despite the name. It handles config validation, mask load, radix fixture validation, unsupported-mode gates, and server arg mutation.
- `metrics.py` is mixed: `meta_info_for_request` and `record_error` are live; `record_selection` appears unused. It is not by itself a reliable active signal.
- DS memory savings require coordinated changes in `pool_configurator.py`, `model_runner_kv_cache_mixin.py`, `memory_pool.py`, and `memory_pool_host.py`; partial port can boot but allocate wrong DSA sidecars.
- DeepSeek/GLM path changes are subtle: `forward_mla.py`, `forward_mha.py`, and `deepseek_v2.py` must land together to avoid stale DSA index behavior or missing `q_nope_for_ds`.
- CUDA graph state changes span `dsa_backend.py` and current-main graph runner equivalents; locate the renamed file before planning hunks.
- Build/runtime deps need a gate: Triton kernels, `sgl_kernel.flash_mla`, custom all-reduce, `safetensors`, and optional `deep_gemm`.
- GLM-5.1-FP8 must be the actual boot target; importing DeepSeek classes alone is insufficient because GLM reuses the path through `glm4_moe`.

ALTERNATIVE_DIRECTIONS:
- Generate an automated import-closure report from the DS allowlist and fail on imports of dropped modules.
- First do a runtime-preserving squash export into v2, validate, then prune diagnostics in separate commits. Larger interim diff, lower behavior risk.
- Keep `validator.py`; prune oracle/capture/calibrate config paths explicitly. Smaller runtime risk than moving validation code now.
- Keep stock `bench_serving.py` and add one perf wrapper that uses `--output-details` to compute p50 decode TPS from ITLs. Lowest diff, weaker fidelity.
- Port only decode TPS output from dev `bench_serving.py`, not warmup/window flags. Slightly larger diff, clearer metric.
- Keep perf launching manual/product-script only; make pytest cover parser, config gates, radix fixture validation, abort behavior, and metadata transport.

QUESTIONS_FOR_USER:
- Should `calibrate.py` be dropped entirely? If yes, what is the authoritative GLM-5.1-FP8 channel mask path and expected content SHA?
- Where should the radix fixture live if `development/` is excluded, and may it contain the current absolute model path fingerprint?
- Is radix cache ON mandatory for acceptance, or can the first clean branch use radix-off while documenting the delta?
- Should the DS-active signal be response `meta_info["double_sparsity"]`, startup/log metrics, Prometheus-style metrics, or a combination?
- Perf fidelity: stock wrapper with derived TPS, or port dev timing fields/window flags into `bench_serving.py`?
- Test floor: extract a new small DS runtime test file, or curate the existing large `test_double_sparsity_unit.py`?
- Should diagnostic config keys be rejected in shipping configs, or retained with their implementation files?
- Is `lifted_budget.py` required for v2 if the parity run uses `enable_lifted_budget_decode=false`?

CANDIDATE_CRITERIA:
- Branch is from v2 `origin/main`; diff contains only allowlisted runtime, one perf script, minimal tests, and explicitly approved artifacts.
- `rg` confirms excluded paths and oracle/capture/calibrate/comparator/manual dsv32 files are absent unless explicitly approved.
- Import gate passes: `import sglang`, `import sglang.srt.layers.attention.dsa_backend`, and DS package imports succeed.
- No runtime file imports a dropped DS module.
- Config negative gates fail clearly for missing/corrupt mask, hash mismatch, HiSparse+DS, hierarchical cache+DS, and radix-on without fixture.
- Radix positive gate accepts the approved fixture only when model path, TP, page size, KV dtype, selector mode, and mask SHA match.
- Server boot gate uses GLM-5.1-FP8 TP=8, DSA backend, FP8 KV, page 64, CUDA graphs, radix cache, and no expandable CUDA allocator.
- Activity gate proves bound selectors and request metadata with `selected_tokens > 0`, `total_tokens > selected_tokens`, `dense_fallback == 0`.
- Abort gate injects a DS per-request error and verifies `update_finish_state()` produces the abort finish state in the same scheduler step.
- Perf gate runs conc-64 GSP 2253+1843/512 with fixed prompt count and seed, emits p50 decode TPS plus P99 TTFT, and compares to the written parity band around ~26.9 TPS / ~25.1s.
