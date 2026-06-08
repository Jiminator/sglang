# Context-Parallel Attention

> 463 nodes

## Key Concepts

- **add_prefix()** (605 connections) — `python/sglang/srt/utils/common.py`
- **get_global_server_args()** (275 connections) — `python/sglang/srt/server_args.py`
- **get_tensor_model_parallel_world_size()** (195 connections) — `python/sglang/srt/distributed/parallel_state.py`
- **get_pp_group()** (118 connections) — `python/sglang/srt/distributed/parallel_state.py`
- **get_attention_tp_size()** (114 connections) — `python/sglang/srt/layers/dp_attention.py`
- **get_rope()** (95 connections) — `python/sglang/srt/layers/rotary_embedding/factory.py`
- **get_tensor_model_parallel_rank()** (78 connections) — `python/sglang/srt/distributed/parallel_state.py`
- **get_attention_tp_rank()** (66 connections) — `python/sglang/srt/layers/dp_attention.py`
- **make_layers()** (66 connections) — `python/sglang/srt/utils/common.py`
- **is_dp_attention_enabled()** (56 connections) — `python/sglang/srt/layers/dp_attention.py`
- **TopK** (50 connections) — `python/sglang/srt/layers/moe/topk.py`
- **DeepSeekV4Config** (46 connections) — `python/sglang/srt/configs/deepseek_v4.py`
- **Tensor** (40 connections) — `python/sglang/srt/models/sarvam_moe.py`
- **ForwardBatch** (36 connections) — `python/sglang/srt/models/sarvam_moe.py`
- **Tensor** (34 connections) — `python/sglang/srt/models/deepseek_v4.py`
- **SarvamMoEMLAAttention** (34 connections) — `python/sglang/srt/models/sarvam_moe.py`
- **get_moe_expert_parallel_world_size()** (33 connections) — `python/sglang/srt/distributed/parallel_state.py`
- **get_attention_cp_size()** (30 connections) — `python/sglang/srt/layers/dp_attention.py`
- **get_rope_config()** (29 connections) — `python/sglang/srt/utils/hf_transformers/common.py`
- **DeepseekV4ForCausalLM** (28 connections) — `python/sglang/srt/models/deepseek_v4.py`
- **QuantizationConfig** (28 connections) — `python/sglang/srt/models/moss_vl.py`
- **C4Indexer** (27 connections) — `python/sglang/srt/layers/attention/dsv4/indexer.py`
- **is_dsa_enable_prefill_cp()** (26 connections) — `python/sglang/srt/layers/attention/dsa/utils.py`
- **MQALayer** (26 connections) — `python/sglang/srt/models/deepseek_v4.py`
- **DeepseekV4DecoderLayer** (25 connections) — `python/sglang/srt/models/deepseek_v4.py`
- *... and 438 more nodes in this community*

## Relationships

- [[DeepSeek MLA Attention & MoE]] (1317 shared connections)
- [[Vision-Language Model Configs]] (311 shared connections)
- [[Model Configs & Pooler]] (244 shared connections)
- [[Qwen3 / Kimi Model Configs]] (74 shared connections)
- [[Community 37]] (54 shared connections)
- [[NCCL Symmetric Memory]] (53 shared connections)
- [[Activation Functions & Gemma]] (44 shared connections)
- [[Llama / GPT-OSS Model Layers]] (41 shared connections)
- [[Community 115]] (40 shared connections)
- [[Community 59]] (38 shared connections)
- [[Community 46]] (37 shared connections)
- [[Mamba2 / Hybrid Linear Attention]] (36 shared connections)

## Source Files

- `python/sglang/srt/configs/deepseek_v4.py`
- `python/sglang/srt/distributed/parallel_state.py`
- `python/sglang/srt/distributed/utils.py`
- `python/sglang/srt/eplb/expert_location_dispatch.py`
- `python/sglang/srt/layers/attention/deepseek_v4_backend.py`
- `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`
- `python/sglang/srt/layers/attention/dsa/utils.py`
- `python/sglang/srt/layers/attention/dsv4/compressor.py`
- `python/sglang/srt/layers/attention/dsv4/indexer.py`
- `python/sglang/srt/layers/attention/linear/lightning_backend.py`
- `python/sglang/srt/layers/attention/mamba/mamba.py`
- `python/sglang/srt/layers/attention/mamba/mixer2_rms_norm_gated.py`
- `python/sglang/srt/layers/communicator.py`
- `python/sglang/srt/layers/dp_attention.py`
- `python/sglang/srt/layers/logits_processor.py`
- `python/sglang/srt/layers/moe/topk.py`
- `python/sglang/srt/layers/n_gram_embedding.py`
- `python/sglang/srt/layers/rotary_embedding/factory.py`
- `python/sglang/srt/layers/utils/cp_utils.py`
- `python/sglang/srt/layers/vocab_parallel_embedding.py`

## Audit Trail

- EXTRACTED: 1924 (34%)
- INFERRED: 3663 (66%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*