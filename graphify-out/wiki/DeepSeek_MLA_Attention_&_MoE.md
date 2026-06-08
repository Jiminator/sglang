# DeepSeek MLA Attention & MoE

> 1563 nodes

## Key Concepts

- **LogitsProcessor** (1537 connections) — `python/sglang/srt/layers/logits_processor.py`
- **RadixAttention** (1441 connections) — `python/sglang/srt/layers/radix_attention.py`
- **ParallelLMHead** (1438 connections) — `python/sglang/srt/layers/vocab_parallel_embedding.py`
- **VocabParallelEmbedding** (1324 connections) — `python/sglang/srt/layers/vocab_parallel_embedding.py`
- **RMSNorm** (1210 connections) — `python/sglang/srt/layers/layernorm.py`
- **PretrainedConfig** (1119 connections) — `python/sglang/srt/models/transformers.py`
- **QKVParallelLinear** (1082 connections) — `python/sglang/srt/layers/linear.py`
- **PPProxyTensors** (937 connections) — `python/sglang/srt/model_executor/forward_batch_info.py`
- **MergedColumnParallelLinear** (854 connections) — `python/sglang/srt/layers/linear.py`
- **ReplicatedLinear** (779 connections) — `python/sglang/srt/layers/linear.py`
- **SiluAndMul** (725 connections) — `python/sglang/srt/layers/activation.py`
- **LayerCommunicator** (438 connections) — `python/sglang/srt/layers/communicator.py`
- **LayerScatterModes** (408 connections) — `python/sglang/srt/layers/communicator.py`
- **ModelConfigForExpertLocation** (393 connections) — `python/sglang/srt/eplb/expert_location.py`
- **ExpertLocationDispatchInfo** (215 connections) — `python/sglang/srt/eplb/expert_location_dispatch.py`
- **DeepseekV2AttentionMLA** (149 connections) — `python/sglang/srt/models/deepseek_v2.py`
- **DeepseekV2ForCausalLM** (121 connections) — `python/sglang/srt/models/deepseek_v2.py`
- **DeepEPMoE** (92 connections) — `python/sglang/srt/layers/moe/ep_moe/layer.py`
- **DeepseekV3ForCausalLM** (88 connections) — `python/sglang/srt/models/deepseek_v2.py`
- **DeepseekV2MLP** (87 connections) — `python/sglang/srt/models/deepseek_v2.py`
- **KTEPWrapperMethod** (77 connections) — `python/sglang/srt/layers/moe/kt_ep_wrapper.py`
- **get_global_expert_distribution_recorder()** (56 connections) — `python/sglang/srt/eplb/expert_distribution.py`
- **BailingMoEForCausalLM** (56 connections) — `python/sglang/srt/models/bailing_moe.py`
- **SboFlags** (55 connections) — `python/sglang/srt/batch_overlap/single_batch_overlap.py`
- **tensor_model_parallel_all_reduce()** (55 connections) — `python/sglang/srt/distributed/communication_op.py`
- *... and 1538 more nodes in this community*

## Relationships

- [[Vision-Language Model Configs]] (2659 shared connections)
- [[Model Configs & Pooler]] (2120 shared connections)
- [[Context-Parallel Attention]] (1317 shared connections)
- [[Qwen3 / Kimi Model Configs]] (583 shared connections)
- [[Mamba2 / Hybrid Linear Attention]] (410 shared connections)
- [[Community 37]] (375 shared connections)
- [[Community 46]] (349 shared connections)
- [[Activation Functions & Gemma]] (342 shared connections)
- [[Llama / GPT-OSS Model Layers]] (335 shared connections)
- [[Community 34]] (240 shared connections)
- [[Community 59]] (202 shared connections)
- [[Community 31]] (170 shared connections)

## Source Files

- `python/sglang/srt/batch_overlap/single_batch_overlap.py`
- `python/sglang/srt/configs/deepseek_ocr.py`
- `python/sglang/srt/configs/laguna.py`
- `python/sglang/srt/distributed/communication_op.py`
- `python/sglang/srt/distributed/device_communicators/custom_all_reduce_v2.py`
- `python/sglang/srt/eplb/expert_distribution.py`
- `python/sglang/srt/eplb/expert_location.py`
- `python/sglang/srt/eplb/expert_location_dispatch.py`
- `python/sglang/srt/layers/activation.py`
- `python/sglang/srt/layers/amx_utils.py`
- `python/sglang/srt/layers/communicator.py`
- `python/sglang/srt/layers/communicator_dsa_cp.py`
- `python/sglang/srt/layers/layernorm.py`
- `python/sglang/srt/layers/linear.py`
- `python/sglang/srt/layers/logits_processor.py`
- `python/sglang/srt/layers/moe/ep_moe/layer.py`
- `python/sglang/srt/layers/moe/kt_ep_wrapper.py`
- `python/sglang/srt/layers/moe/utils.py`
- `python/sglang/srt/layers/quantization/mxfp4_flashinfer_trtllm_moe.py`
- `python/sglang/srt/layers/radix_attention.py`

## Audit Trail

- EXTRACTED: 5722 (19%)
- INFERRED: 24688 (81%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*