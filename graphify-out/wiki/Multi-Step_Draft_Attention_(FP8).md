# Multi-Step Draft Attention (FP8)

> 496 nodes

## Key Concepts

- **CaptureHiddenMode** (216 connections) — `python/sglang/srt/model_executor/forward_batch_info.py`
- **EagleDraftInput** (174 connections) — `python/sglang/srt/speculative/eagle_info.py`
- **EagleVerifyInput** (170 connections) — `python/sglang/srt/speculative/eagle_info.py`
- **ForwardContext** (120 connections) — `python/sglang/srt/model_executor/forward_context.py`
- **DraftBackendFactory** (110 connections) — `python/sglang/srt/speculative/draft_utils.py`
- **CudaGraphRunner** (108 connections) — `python/sglang/srt/model_executor/cuda_graph_runner.py`
- **TritonAttnBackend** (87 connections) — `python/sglang/srt/layers/attention/triton_backend.py`
- **EAGLEWorker** (80 connections) — `python/sglang/srt/speculative/eagle_worker.py`
- **EAGLEWorkerV2** (71 connections) — `python/sglang/srt/speculative/eagle_worker_v2.py`
- **EAGLEDraftCudaGraphRunner** (64 connections) — `python/sglang/srt/speculative/eagle_draft_cuda_graph_runner.py`
- **EagleDraftExtendInput** (57 connections) — `python/sglang/srt/speculative/eagle_info.py`
- **EAGLEDraftExtendCudaGraphRunner** (53 connections) — `python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py`
- **AdaptiveController** (52 connections) — `python/sglang/srt/speculative/adaptive_runtime_state.py`
- **TRTLLMMLABackend** (50 connections) — `python/sglang/srt/layers/attention/trtllm_mla_backend.py`
- **BaseSpecWorker** (49 connections) — `python/sglang/srt/speculative/base_spec_worker.py`
- **EagleDraftWorker** (49 connections) — `python/sglang/srt/speculative/eagle_worker_v2.py`
- **TRTLLMHAAttnBackend** (47 connections) — `python/sglang/srt/layers/attention/trtllm_mha_backend.py`
- **SpecInputType** (46 connections) — `python/sglang/srt/speculative/spec_info.py`
- **log_info_on_rank0()** (45 connections) — `python/sglang/srt/utils/common.py`
- **DeepEPCudaGraphRunnerAdapter** (44 connections) — `python/sglang/srt/model_executor/cuda_graph_runner.py`
- **EagleVerifyOutput** (44 connections) — `python/sglang/srt/speculative/eagle_info.py`
- **EAGLEDraftNpuGraphRunner** (41 connections) — `python/sglang/srt/hardware_backend/npu/graph_runner/eagle_draft_npu_graph_runner.py`
- **ScheduleBatch** (38 connections) — `python/sglang/srt/speculative/eagle_worker_v2.py`
- **SpecRuntimeState** (37 connections) — `python/sglang/srt/speculative/adaptive_runtime_state.py`
- **TreeMaskMode** (36 connections) — `python/sglang/srt/speculative/eagle_utils.py`
- *... and 471 more nodes in this community*

## Relationships

- [[CLI Arg Parsing & Deprecation]] (335 shared connections)
- [[Hybrid Attention Backend]] (187 shared connections)
- [[Aiter Attention Backend]] (182 shared connections)
- [[Breakable CUDA Graph (TBO)]] (145 shared connections)
- [[Vision-Language Model Configs]] (113 shared connections)
- [[Community 62]] (85 shared connections)
- [[Anthropic/OpenAI API Entrypoints]] (83 shared connections)
- [[Grammar Manager & HiCache Clear]] (58 shared connections)
- [[Model Configs & Pooler]] (53 shared connections)
- [[Pipeline Parallel & Custom Allreduce]] (50 shared connections)
- [[Batch-Overlap Operations]] (24 shared connections)
- [[Disaggregation Bootstrap & Decode]] (22 shared connections)

## Source Files

- `python/sglang/srt/distributed/parallel_state.py`
- `python/sglang/srt/hardware_backend/npu/graph_runner/eagle_draft_extend_npu_graph_runner.py`
- `python/sglang/srt/hardware_backend/npu/graph_runner/eagle_draft_npu_graph_runner.py`
- `python/sglang/srt/kv_canary/runner/canary_manager.py`
- `python/sglang/srt/layers/attention/aiter_backend.py`
- `python/sglang/srt/layers/attention/tokenspeed_mla_backend.py`
- `python/sglang/srt/layers/attention/triton_backend.py`
- `python/sglang/srt/layers/attention/trtllm_mha_backend.py`
- `python/sglang/srt/layers/attention/trtllm_mla_backend.py`
- `python/sglang/srt/layers/moe/utils.py`
- `python/sglang/srt/model_executor/cpu_graph_runner.py`
- `python/sglang/srt/model_executor/cuda_graph_runner.py`
- `python/sglang/srt/model_executor/forward_batch_info.py`
- `python/sglang/srt/model_executor/forward_context.py`
- `python/sglang/srt/model_executor/input_buffers.py`
- `python/sglang/srt/speculative/adaptive_runtime_state.py`
- `python/sglang/srt/speculative/adaptive_spec_params.py`
- `python/sglang/srt/speculative/base_spec_worker.py`
- `python/sglang/srt/speculative/dflash_info.py`
- `python/sglang/srt/speculative/draft_utils.py`

## Audit Trail

- EXTRACTED: 1499 (30%)
- INFERRED: 3428 (70%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*