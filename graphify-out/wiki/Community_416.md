# Community 416

> 15 nodes

## Key Concepts

- **lora_moe_runners.py** (8 connections) — `python/sglang/srt/lora/lora_moe_runners.py`
- **build_lora_hooks()** (8 connections) — `python/sglang/srt/lora/lora_moe_runners.py`
- **_compute_lora_alignment()** (7 connections) — `python/sglang/srt/lora/lora_moe_runners.py`
- **_add_lora_gate_up_delta()** (6 connections) — `python/sglang/srt/lora/lora_moe_runners.py`
- **_naive_moe_lora_align_block_size()** (5 connections) — `python/sglang/srt/lora/lora_moe_runners.py`
- **Tensor** (5 connections) — `python/sglang/srt/lora/lora_moe_runners.py`
- **_add_lora_down_delta()** (5 connections) — `python/sglang/srt/lora/lora_moe_runners.py`
- **_get_moe_lora_block_config()** (4 connections) — `python/sglang/srt/lora/lora_moe_runners.py`
- **device** (1 connections) — `python/sglang/srt/lora/lora_moe_runners.py`
- **Compute rank-aware block sizes for MoE LoRA kernels.      Shrink: output dim is** (1 connections) — `python/sglang/srt/lora/lora_moe_runners.py`
- **Construct LoRA token-expert alignment on CPU for small batches.      When the nu** (1 connections) — `python/sglang/srt/lora/lora_moe_runners.py`
- **Compute LoRA alignment tensors for the non-virtual-expert (classic) path.      R** (1 connections) — `python/sglang/srt/lora/lora_moe_runners.py`
- **Add LoRA gate_up delta to intermediate_cache in-place.** (1 connections) — `python/sglang/srt/lora/lora_moe_runners.py`
- **Add LoRA down delta to intermediate_cache in-place.** (1 connections) — `python/sglang/srt/lora/lora_moe_runners.py`
- **Build LoRA hook closures for injection into any MoE runner.      Computes token_** (1 connections) — `python/sglang/srt/lora/lora_moe_runners.py`

## Relationships

- [[DeepSeek MLA Attention & MoE]] (5 shared connections)
- [[Qwen3 / Kimi Model Configs]] (3 shared connections)
- [[MoE Dispatch/Combine (Cutlass)]] (2 shared connections)
- [[NCCL Symmetric Memory]] (1 shared connections)

## Source Files

- `python/sglang/srt/lora/lora_moe_runners.py`

## Audit Trail

- EXTRACTED: 51 (93%)
- INFERRED: 4 (7%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*