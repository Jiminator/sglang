# Community 229

> 29 nodes

## Key Concepts

- **.__init__()** (8 connections) — `python/sglang/srt/hardware_backend/mlx/model_runner.py`
- **._get_attn_config()** (8 connections) — `python/sglang/srt/hardware_backend/mlx/model_runner.py`
- **find_attention_layers()** (7 connections) — `python/sglang/srt/hardware_backend/mlx/kv_cache/model_patching.py`
- **patch_model_attention()** (6 connections) — `python/sglang/srt/hardware_backend/mlx/kv_cache/model_patching.py`
- **._attention_kv_config_for_layer()** (6 connections) — `python/sglang/srt/hardware_backend/mlx/model_runner.py`
- **model_patching.py** (5 connections) — `python/sglang/srt/hardware_backend/mlx/kv_cache/model_patching.py`
- **_find_attention_attr()** (5 connections) — `python/sglang/srt/hardware_backend/mlx/kv_cache/model_patching.py`
- **Any** (5 connections) — `python/sglang/srt/hardware_backend/mlx/kv_cache/model_patching.py`
- **._build_aot_kernels()** (5 connections) — `python/sglang/srt/hardware_backend/mlx/model_runner.py`
- **get_num_layers()** (4 connections) — `python/sglang/srt/hardware_backend/mlx/kv_cache/model_patching.py`
- **Dtype** (4 connections) — `python/sglang/srt/hardware_backend/mlx/model_runner.py`
- **._compute_pool_size()** (4 connections) — `python/sglang/srt/hardware_backend/mlx/model_runner.py`
- **.init_cache_pools()** (4 connections) — `python/sglang/srt/hardware_backend/mlx/model_runner.py`
- **._load_model()** (3 connections) — `python/sglang/srt/hardware_backend/mlx/model_runner.py`
- **._attention_module_for_layer()** (3 connections) — `python/sglang/srt/hardware_backend/mlx/model_runner.py`
- **MlxAOTKernelSet** (3 connections) — `python/sglang/srt/hardware_backend/mlx/model_runner.py`
- **ReqToTokenPool** (3 connections) — `python/sglang/srt/hardware_backend/mlx/model_runner.py`
- **._extract_model_components()** (3 connections) — `python/sglang/srt/hardware_backend/mlx/model_runner.py`
- **Model introspection and attention patching.** (1 connections) — `python/sglang/srt/hardware_backend/mlx/kv_cache/model_patching.py`
- **Return the direct child name that satisfies the attention contract.** (1 connections) — `python/sglang/srt/hardware_backend/mlx/kv_cache/model_patching.py`
- **Find transformer layers and per-layer attention attribute names.** (1 connections) — `python/sglang/srt/hardware_backend/mlx/kv_cache/model_patching.py`
- **Install MLXAttentionWrapper on all attention layers (idempotent).      The wrapp** (1 connections) — `python/sglang/srt/hardware_backend/mlx/kv_cache/model_patching.py`
- **Return the number of transformer layers.** (1 connections) — `python/sglang/srt/hardware_backend/mlx/kv_cache/model_patching.py`
- **Load model using mlx_lm. If ``self._quantization`` requests a preset         (e.** (1 connections) — `python/sglang/srt/hardware_backend/mlx/model_runner.py`
- **Return the uniform attention KV config used by the shared MLX pool.** (1 connections) — `python/sglang/srt/hardware_backend/mlx/model_runner.py`
- *... and 4 more nodes in this community*

## Relationships

- [[Community 87]] (10 shared connections)
- [[Community 133]] (5 shared connections)
- [[Community 47]] (3 shared connections)
- [[Disaggregation Bootstrap & Decode]] (3 shared connections)
- [[Community 528]] (2 shared connections)

## Source Files

- `python/sglang/srt/hardware_backend/mlx/kv_cache/model_patching.py`
- `python/sglang/srt/hardware_backend/mlx/model_runner.py`

## Audit Trail

- EXTRACTED: 80 (82%)
- INFERRED: 17 (18%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*