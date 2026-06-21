# Ask Codex Input

## Question

You are doing the INVENTORY/closure analysis for a curation clean-port of the table-free Double Sparsity (DS) runtime in SGLang. You are in the DEV clone /sgl-workspace/sglang on branch dev/double-sparsity-standalone (you may read any file). The DS feature is DONE/validated; we are extracting the MINIMAL runtime onto a fresh branch off clean main in a SEPARATE clone. A file ships ONLY if removing it breaks `import sglang`, DS server boot, or the conc-64 perf run. Dev scaffolding (.pensieve/.humanize/development/SLOS.md, oracle/capture/recall/comparator/manual dsv32) does NOT ship.

I have already computed the DS diff (dev HEAD vs base a77449f86 = main merged 2026-05-25). Validate my allowlist and adjudicate the newly-found files. Be terse and concrete (file:line). Use repo reads to verify.

=== double_sparsity/ package (22 files) — my keep/drop ===
KEEP (16): __init__.py, config.py, absorbed_latent.py, absorbed_latent_kernel.py, selection_kernel.py, topk_kernel.py, selector.py, channel_mask.py, cuda_graph.py, page_table_adapter.py, lifted_budget.py, error_containment.py, metrics.py (prune dead record_selection), validator.py (strip radix-fixture gate), calibrate.py (ships per decision).
DROP (6): oracle_artifact_sink.py, selection_recall_oracle.py, radix_fixture_capture.py, score_capture.py, selection_capture.py, latent_capture.py.

=== modified-upstream code files in the DS diff ===
Plan already accounted for: dsa_backend.py, server_args.py, deepseek_v2.py, forward_mla.py, forward_mha.py, model_runner.py, model_runner_kv_cache_mixin.py, pool_configurator.py, cuda_graph_runner.py (REFACTORED AWAY on current main -> retarget to model_executor/runner/ + runner_backend/), memory_pool.py, memory_pool_host.py, schedule_batch.py, scheduler.py, io_struct.py, tokenizer_manager.py, batch_result_processor.py, logits_processor.py (one per_request_summary dataclass field).

NEWLY FOUND by my inventory (the plan/draft MISSED these) — adjudicate ship vs drop with closure reasoning:
1. managers/scheduler_components/output_streamer.py — adds per_request_summary dict field + merge logic + passes per_request_summary into output. Carries the per-request DS meta to the stream.
2. managers/multi_tokenizer_mixin.py — _extract_per_request_summary_by_index + threads per_request_summary per child output.
3. managers/detokenizer_manager.py — forwards per_request_summary=getattr(recv_obj,...).
4. distributed/device_communicators/custom_all_reduce_v2.py — adds override_algo param (fixed FP summation order). ONLY caller: selection_kernel.py:198 self.base_ca.custom_all_reduce(inp, override_algo=self.pinned_algo) — a SHIPPED file.
5. layers/attention/dsa/dequant_k_cache.py — adds alloc-free dequantize_k_cache_paged_out (refactors existing dequantize_k_cache_paged into a thin wrapper). Callers of _out: lifted_budget.py:302,305; referenced cuda_graph.py:132, selection_kernel.py:110, validator.py:109. The original dequantize_k_cache_paged also used by dsa_backend.py:1796 + forward_mha.py:541 (those exist on main already).
6. bench_serving.py — has DS-meta capture hunks. The decision is bench_serving STAYS STOCK (DS-meta is eval instrumentation) -> DROP its DS hunks.

QUESTIONS:
A. Is the 16/6 package split correct? Any KEEP file that is actually dev-only, or any DROP file that something shipped imports?
B. For each of the 6 newly-found files: ship or drop, and is it closure-required (import/boot/conc-64) or only eval/optional? In particular: are files 1-3 (per_request_summary transport) REQUIRED for meta_info["double_sparsity"] to reach the client decode response, or is the chain complete without them? Is file 5 needed only if lifted_budget ships?
C. Any OTHER modified-upstream file or transitive dependency I am still missing for closure (e.g. a shipped DS file importing something only added on the dev branch outside the files above)? Check imports of the KEEP set against current main.
D. Kernel/build deps: do shipped DS modules need triton / sgl_kernel.flash_mla (flash_mla_sparse_fwd) / deep_gemm / custom kernels that exist on current main? Name any that would be a NEW build dependency.

Output strictly:
ALLOWLIST_CONFIRMED: <yes/no + any correction>
NEWFILE_VERDICTS: <one line each for files 1-6: SHIP/DROP + closure-required? + reason>
MISSING_CLOSURE: <any other file/dep I'm missing, or NONE>
KERNEL_DEPS: <list or NONE-NEW>

## Configuration

- Model: gpt-5.5
- Effort: xhigh
- Timeout: 3600s
- Timestamp: 2026-06-18_13-27-06
- Tool: codex
