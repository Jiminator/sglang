# Vision-Language Model Configs

> 1674 nodes

## Key Concepts

- **ForwardBatch** (2914 connections) — `python/sglang/srt/model_executor/forward_batch_info.py`
- **QuantizationConfig** (2376 connections) — `python/sglang/srt/layers/quantization/base_config.py`
- **ColumnParallelLinear** (860 connections) — `python/sglang/srt/layers/linear.py`
- **MultimodalDataItem** (848 connections) — `python/sglang/srt/managers/schedule_batch.py`
- **MultimodalInputs** (837 connections) — `python/sglang/srt/managers/schedule_batch.py`
- **Modality** (615 connections) — `python/sglang/srt/managers/schedule_batch.py`
- **MultiModalityDataPaddingPatternMultimodalTokens** (541 connections) — `python/sglang/srt/managers/mm_utils.py`
- **VisionAttention** (495 connections) — `python/sglang/srt/layers/attention/vision.py`
- **Conv2dLayer** (226 connections) — `python/sglang/srt/layers/conv.py`
- **Conv3dLayer** (92 connections) — `python/sglang/srt/layers/conv.py`
- **QuickGELU** (86 connections) — `python/sglang/srt/layers/activation.py`
- **PretrainedConfig** (84 connections)
- **SiglipVisionModel** (65 connections) — `python/sglang/srt/models/siglip.py`
- **ModelSlimConfig** (55 connections) — `python/sglang/srt/layers/quantization/modelslim/modelslim.py`
- **ViTCudaGraphRunner** (53 connections) — `python/sglang/srt/multimodal/vit_cuda_graph_runner.py`
- **general_mm_embed_routine()** (49 connections) — `python/sglang/srt/managers/mm_utils.py`
- **Qwen3Model** (48 connections) — `python/sglang/srt/models/qwen3.py`
- **Qwen3VLMoeVisionModel** (45 connections) — `python/sglang/srt/models/qwen3_vl.py`
- **Step3VLConfig** (42 connections) — `python/sglang/srt/configs/step3_vl.py`
- **deepseek_janus_pro.py** (42 connections) — `python/sglang/srt/models/deepseek_janus_pro.py`
- **SiglipVisionConfig** (41 connections) — `python/sglang/srt/models/siglip.py`
- **Tensor** (41 connections) — `python/sglang/srt/models/step3_vl.py`
- **Gemma3nTextModel** (40 connections) — `python/sglang/srt/models/gemma3n_causal.py`
- **Tensor** (40 connections) — `python/sglang/srt/models/qwen3_vl.py`
- **Ernie4_5_ForCausalLM** (39 connections) — `python/sglang/srt/models/ernie4.py`
- *... and 1649 more nodes in this community*

## Relationships

- [[DeepSeek MLA Attention & MoE]] (2659 shared connections)
- [[Model Configs & Pooler]] (1359 shared connections)
- [[Llama / GPT-OSS Model Layers]] (844 shared connections)
- [[Context-Parallel Attention]] (311 shared connections)
- [[Community 31]] (180 shared connections)
- [[Qwen3 / Kimi Model Configs]] (178 shared connections)
- [[Compressed-Tensors Quant Linear]] (173 shared connections)
- [[Community 34]] (171 shared connections)
- [[Community 46]] (161 shared connections)
- [[Community 37]] (158 shared connections)
- [[Community 59]] (148 shared connections)
- [[Grammar Manager & HiCache Clear]] (148 shared connections)

## Source Files

- `python/sglang/srt/configs/deepseekvl2.py`
- `python/sglang/srt/configs/dots_vlm.py`
- `python/sglang/srt/configs/janus_pro.py`
- `python/sglang/srt/configs/jet_vlm.py`
- `python/sglang/srt/configs/kimi_k25.py`
- `python/sglang/srt/configs/kimi_vl.py`
- `python/sglang/srt/configs/kimi_vl_moonvit.py`
- `python/sglang/srt/configs/nano_nemotron_vl.py`
- `python/sglang/srt/configs/points_v15_chat.py`
- `python/sglang/srt/configs/qwen3_asr.py`
- `python/sglang/srt/configs/qwen3_vl.py`
- `python/sglang/srt/configs/step3_vl.py`
- `python/sglang/srt/configs/step3p5.py`
- `python/sglang/srt/configs/step3p7.py`
- `python/sglang/srt/hardware_backend/npu/graph_runner/vit_npu_graph_runner.py`
- `python/sglang/srt/layers/activation.py`
- `python/sglang/srt/layers/attention/dsa/dsa_indexer.py`
- `python/sglang/srt/layers/attention/dsv4/compressor.py`
- `python/sglang/srt/layers/attention/mamba/mamba.py`
- `python/sglang/srt/layers/attention/vision.py`

## Audit Trail

- EXTRACTED: 5252 (23%)
- INFERRED: 17214 (77%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*