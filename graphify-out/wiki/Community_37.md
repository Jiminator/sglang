# Community 37

> 160 nodes

## Key Concepts

- **ScatterMode** (92 connections) — `python/sglang/srt/layers/communicator.py`
- **MiMoV2ForCausalLM** (65 connections) — `python/sglang/srt/models/mimo_v2.py`
- **MiMoV2Attention** (43 connections) — `python/sglang/srt/models/mimo_v2.py`
- **Tensor** (42 connections) — `python/sglang/srt/models/mimo_v2.py`
- **MiMoV2MoE** (41 connections) — `python/sglang/srt/models/mimo_v2.py`
- **MiMoV2MLP** (39 connections) — `python/sglang/srt/models/mimo_v2.py`
- **ForwardBatch** (35 connections) — `python/sglang/srt/models/mimo_v2.py`
- **MiMoV2DecoderLayer** (34 connections) — `python/sglang/srt/models/mimo_v2.py`
- **MiMoVisionTransformer** (34 connections) — `python/sglang/srt/models/mimo_vl.py`
- **MiMoV2Model** (33 connections) — `python/sglang/srt/models/mimo_v2.py`
- **Qwen2_5_VLMLP** (33 connections) — `python/sglang/srt/models/qwen2_5_vl.py`
- **Qwen2_5_VisionPatchMerger** (33 connections) — `python/sglang/srt/models/qwen2_5_vl.py`
- **QuantizationConfig** (32 connections) — `python/sglang/srt/models/mimo_v2.py`
- **MoEGate** (30 connections) — `python/sglang/srt/models/mimo_v2.py`
- **MiMoV2Config** (30 connections) — `python/sglang/srt/models/mimo_v2.py`
- **MiMoAudioEncoder** (29 connections) — `python/sglang/srt/models/mimo_audio.py`
- **MultimodalDataItem** (29 connections) — `python/sglang/srt/models/mimo_v2.py`
- **Embedding** (28 connections) — `python/sglang/srt/models/mimo_v2.py`
- **PPProxyTensors** (28 connections) — `python/sglang/srt/models/mimo_v2.py`
- **MiMoV2FlashForCausalLM** (28 connections) — `python/sglang/srt/models/mimo_v2.py`
- **MiMoVLVisionConfig** (26 connections) — `python/sglang/srt/models/mimo_vl.py`
- **CommunicateWithAllReduceAndLayerNormFn** (21 connections) — `python/sglang/srt/layers/communicator.py`
- **MiMoAudioEncoderConfig** (21 connections) — `python/sglang/srt/models/mimo_audio.py`
- **communicator.py** (20 connections) — `python/sglang/srt/layers/communicator.py`
- **MiMoV2MTP** (19 connections) — `python/sglang/srt/models/mimo_v2_nextn.py`
- *... and 135 more nodes in this community*

## Relationships

- [[DeepSeek MLA Attention & MoE]] (375 shared connections)
- [[Vision-Language Model Configs]] (158 shared connections)
- [[Context-Parallel Attention]] (54 shared connections)
- [[Batch-Overlap Operations]] (39 shared connections)
- [[Model Configs & Pooler]] (23 shared connections)
- [[NCCL Symmetric Memory]] (15 shared connections)
- [[Community 71]] (11 shared connections)
- [[Community 49]] (5 shared connections)
- [[MoE Two-Batch Overlap Dispatch]] (4 shared connections)
- [[Qwen3 / Kimi Model Configs]] (4 shared connections)
- [[Community 490]] (4 shared connections)
- [[CLI Arg Parsing & Deprecation]] (3 shared connections)

## Source Files

- `python/sglang/srt/configs/model_config.py`
- `python/sglang/srt/layers/communicator.py`
- `python/sglang/srt/layers/communicator_dsa_cp.py`
- `python/sglang/srt/layers/utils/cp_utils.py`
- `python/sglang/srt/models/mimo_audio.py`
- `python/sglang/srt/models/mimo_v2.py`
- `python/sglang/srt/models/mimo_v2_nextn.py`
- `python/sglang/srt/models/mimo_vl.py`
- `python/sglang/srt/models/qwen2_5_vl.py`

## Audit Trail

- EXTRACTED: 570 (37%)
- INFERRED: 962 (63%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*