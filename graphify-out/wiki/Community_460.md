# Community 460

> 13 nodes

## Key Concepts

- **apply_module_patch()** (6 connections) — `python/sglang/srt/utils/common.py`
- **glm46v_processor.py** (4 connections) — `python/sglang/srt/hardware_backend/npu/modules/glm46v_processor.py`
- **qwen_vl_processor.py** (4 connections) — `python/sglang/srt/hardware_backend/npu/modules/qwen_vl_processor.py`
- **npu_apply_glm46v_image_preprocess_patch()** (3 connections) — `python/sglang/srt/hardware_backend/npu/modules/glm46v_processor.py`
- **npu_apply_qwen_image_preprocess_patch()** (3 connections) — `python/sglang/srt/hardware_backend/npu/modules/qwen_vl_processor.py`
- **transform_patches_to_flatten()** (2 connections) — `python/sglang/srt/hardware_backend/npu/modules/qwen_vl_processor.py`
- **parse_module_path()** (2 connections) — `python/sglang/srt/utils/common.py`
- **npu_wrapper_glm46v_preprocess()** (1 connections) — `python/sglang/srt/hardware_backend/npu/modules/glm46v_processor.py`
- **npu_wrapper_glm46v_video_preprocess()** (1 connections) — `python/sglang/srt/hardware_backend/npu/modules/glm46v_processor.py`
- **NPU patch for GLM-4.6V image and video preprocessing.  The GLM-4.6V image proces** (1 connections) — `python/sglang/srt/hardware_backend/npu/modules/glm46v_processor.py`
- **Tensor** (1 connections) — `python/sglang/srt/hardware_backend/npu/modules/qwen_vl_processor.py`
- **npu_wrapper_preprocess()** (1 connections) — `python/sglang/srt/hardware_backend/npu/modules/qwen_vl_processor.py`
- **npu_wrapper_video_preprocess()** (1 connections) — `python/sglang/srt/hardware_backend/npu/modules/qwen_vl_processor.py`

## Relationships

- [[Community 102]] (2 shared connections)
- [[Community 42]] (2 shared connections)
- [[Compressed-Tensors Quant Linear]] (1 shared connections)
- [[Community 118]] (1 shared connections)

## Source Files

- `python/sglang/srt/hardware_backend/npu/modules/glm46v_processor.py`
- `python/sglang/srt/hardware_backend/npu/modules/qwen_vl_processor.py`
- `python/sglang/srt/utils/common.py`

## Audit Trail

- EXTRACTED: 22 (73%)
- INFERRED: 8 (27%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*