# Community 275

> 24 nodes

## Key Concepts

- **server_args.py** (30 connections) — `python/sglang/srt/server_args.py`
- **add_load_format_choices()** (1 connections) — `python/sglang/srt/server_args.py`
- **add_quantization_method_choices()** (1 connections) — `python/sglang/srt/server_args.py`
- **add_attention_backend_choices()** (1 connections) — `python/sglang/srt/server_args.py`
- **add_deterministic_attention_backend_choices()** (1 connections) — `python/sglang/srt/server_args.py`
- **add_radix_supported_deterministic_attention_backend_choices()** (1 connections) — `python/sglang/srt/server_args.py`
- **add_disagg_transfer_backend_choices()** (1 connections) — `python/sglang/srt/server_args.py`
- **add_grammar_backend_choices()** (1 connections) — `python/sglang/srt/server_args.py`
- **add_moe_runner_backend_choices()** (1 connections) — `python/sglang/srt/server_args.py`
- **add_fp8_gemm_runner_backend_choices()** (1 connections) — `python/sglang/srt/server_args.py`
- **add_fp4_gemm_runner_backend_choices()** (1 connections) — `python/sglang/srt/server_args.py`
- **add_radix_eviction_policy_choices()** (1 connections) — `python/sglang/srt/server_args.py`
- **add_rl_on_policy_target_choices()** (1 connections) — `python/sglang/srt/server_args.py`
- **add_linear_attn_kernel_backend_choices()** (1 connections) — `python/sglang/srt/server_args.py`
- **# TODO: this list should likely contain only methods that support online quantiz** (1 connections) — `python/sglang/srt/server_args.py`
- **# FIXME: hack to reduce ITL when decode bs is small** (1 connections) — `python/sglang/srt/server_args.py`
- **# TODO: when extra_buffer is more verified, we can set the default path based on** (1 connections) — `python/sglang/srt/server_args.py`
- **# FIXME: https://github.com/sgl-project/sglang/pull/7367 is not compatible with** (1 connections) — `python/sglang/srt/server_args.py`
- **# FIXME: https://github.com/sgl-project/sglang/pull/7367 is not compatible with** (1 connections) — `python/sglang/srt/server_args.py`
- **# TODO: currently, it is only supported in the single node scenario. https://git** (1 connections) — `python/sglang/srt/server_args.py`
- **# TODO: adaptive spec currently requires topk=1, so each runtime state** (1 connections) — `python/sglang/srt/server_args.py`
- **# NOTE: CUDA Green Context may encounter potential issues with CudaGraph on torc** (1 connections) — `python/sglang/srt/server_args.py`
- **# TODO: Also validate grpc_port != metrics_http_port and grpc_port != nccl_port** (1 connections) — `python/sglang/srt/server_args.py`
- **# NOTE: This is a global variable to hold the server args for scheduler.** (1 connections) — `python/sglang/srt/server_args.py`

## Relationships

- [[CLI Arg Parsing & Deprecation]] (4 shared connections)
- [[Community 309]] (1 shared connections)
- [[Context-Parallel Attention]] (1 shared connections)
- [[Anthropic/OpenAI API Entrypoints]] (1 shared connections)

## Source Files

- `python/sglang/srt/server_args.py`

## Audit Trail

- EXTRACTED: 53 (100%)
- INFERRED: 0 (0%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*