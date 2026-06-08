# Community 271

> 24 nodes

## Key Concepts

- **Module** (7 connections) — `python/sglang/srt/layers/moe/kt_ep_wrapper.py`
- **KTConfig** (6 connections) — `python/sglang/srt/layers/moe/kt_ep_wrapper.py`
- **.apply()** (6 connections) — `python/sglang/srt/layers/moe/kt_ep_wrapper.py`
- **.__init__()** (5 connections) — `python/sglang/srt/layers/moe/kt_ep_wrapper.py`
- **kt_ep_wrapper.py** (4 connections) — `python/sglang/srt/layers/moe/kt_ep_wrapper.py`
- **create_kt_config_from_server_args()** (4 connections) — `python/sglang/srt/layers/moe/kt_ep_wrapper.py`
- **mask_cpu_expert_ids()** (4 connections) — `python/sglang/srt/layers/moe/kt_ep_wrapper.py`
- **Tensor** (4 connections) — `python/sglang/srt/layers/moe/kt_ep_wrapper.py`
- **.create_weights()** (4 connections) — `python/sglang/srt/layers/moe/kt_ep_wrapper.py`
- **.process_weights_after_loading()** (4 connections) — `python/sglang/srt/layers/moe/kt_ep_wrapper.py`
- **.submit()** (4 connections) — `python/sglang/srt/layers/moe/kt_ep_wrapper.py`
- **.sync()** (4 connections) — `python/sglang/srt/layers/moe/kt_ep_wrapper.py`
- **dtype** (3 connections) — `python/sglang/srt/layers/moe/kt_ep_wrapper.py`
- **.create_moe_runner()** (3 connections) — `python/sglang/srt/layers/moe/kt_ep_wrapper.py`
- **Configuration for KTransformers heterogeneous computing CPU part.      Args:** (1 connections) — `python/sglang/srt/layers/moe/kt_ep_wrapper.py`
- **Create KTConfig from ServerArgs if KT is configured.      Args:         server_a** (1 connections) — `python/sglang/srt/layers/moe/kt_ep_wrapper.py`
- **Mask CPU expert IDs by setting them to -1.      This function masks expert IDs t** (1 connections) — `python/sglang/srt/layers/moe/kt_ep_wrapper.py`
- **Initialize the KT EP wrapper.          Args:             gpu_method: The quantiz** (1 connections) — `python/sglang/srt/layers/moe/kt_ep_wrapper.py`
- **Create weights for both GPU and CPU experts.          Args:             layer: T** (1 connections) — `python/sglang/srt/layers/moe/kt_ep_wrapper.py`
- **Process weights after loading from checkpoint.          Args:             layer:** (1 connections) — `python/sglang/srt/layers/moe/kt_ep_wrapper.py`
- **Create MoE runner for computation.          Args:             layer: The MoE lay** (1 connections) — `python/sglang/srt/layers/moe/kt_ep_wrapper.py`
- **Submit CPU expert computation asynchronously (non-blocking).          This metho** (1 connections) — `python/sglang/srt/layers/moe/kt_ep_wrapper.py`
- **Synchronize and retrieve CPU expert computation results.          This method wa** (1 connections) — `python/sglang/srt/layers/moe/kt_ep_wrapper.py`
- **Execute hybrid CPU+GPU MoE forward pass with parallelism.          This is the m** (1 connections) — `python/sglang/srt/layers/moe/kt_ep_wrapper.py`

## Relationships

- [[DeepSeek MLA Attention & MoE]] (8 shared connections)
- [[Compressed-Tensors Quant Linear]] (5 shared connections)
- [[CLI Arg Parsing & Deprecation]] (4 shared connections)
- [[MoE Two-Batch Overlap Dispatch]] (1 shared connections)
- [[Context-Parallel Attention]] (1 shared connections)
- [[Hybrid Attention Backend]] (1 shared connections)

## Source Files

- `python/sglang/srt/layers/moe/kt_ep_wrapper.py`

## Audit Trail

- EXTRACTED: 61 (85%)
- INFERRED: 11 (15%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*