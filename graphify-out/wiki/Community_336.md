# Community 336

> 20 nodes

## Key Concepts

- **tensor_bridge.py** (9 connections) — `python/sglang/srt/utils/tensor_bridge.py`
- **mlx_to_torch()** (8 connections) — `python/sglang/srt/utils/tensor_bridge.py`
- **_get_tensor_size_bytes()** (5 connections) — `python/sglang/srt/utils/tensor_bridge.py`
- **array** (5 connections) — `python/sglang/srt/utils/tensor_bridge.py`
- **_is_safe_for_mps()** (5 connections) — `python/sglang/srt/utils/tensor_bridge.py`
- **get_torch_device()** (4 connections) — `python/sglang/srt/utils/tensor_bridge.py`
- **torch_to_mlx()** (4 connections) — `python/sglang/srt/utils/tensor_bridge.py`
- **sync_mlx()** (3 connections) — `python/sglang/srt/utils/tensor_bridge.py`
- **is_mlx_available()** (2 connections) — `python/sglang/srt/utils/tensor_bridge.py`
- **device** (2 connections) — `python/sglang/srt/utils/tensor_bridge.py`
- **Tensor** (2 connections) — `python/sglang/srt/utils/tensor_bridge.py`
- **sync_torch()** (2 connections) — `python/sglang/srt/utils/tensor_bridge.py`
- **Return True when the ``mlx`` package can be imported.** (1 connections) — `python/sglang/srt/utils/tensor_bridge.py`
- **Get the PyTorch device for Metal/MPS.      Returns:         torch.device for MPS** (1 connections) — `python/sglang/srt/utils/tensor_bridge.py`
- **Calculate the size of an MLX array in bytes.      Args:         array: MLX array** (1 connections) — `python/sglang/srt/utils/tensor_bridge.py`
- **Check if an array is safe to transfer to MPS without hitting size limits.      M** (1 connections) — `python/sglang/srt/utils/tensor_bridge.py`
- **Convert PyTorch tensor to MLX array.      Uses numpy as an intermediate to enabl** (1 connections) — `python/sglang/srt/utils/tensor_bridge.py`
- **Convert MLX array to PyTorch tensor.      Uses numpy as an intermediate to enabl** (1 connections) — `python/sglang/srt/utils/tensor_bridge.py`
- **Synchronize MLX operations.      Call this before converting MLX arrays to ensur** (1 connections) — `python/sglang/srt/utils/tensor_bridge.py`
- **Synchronize PyTorch MPS operations.      Call this before converting PyTorch ten** (1 connections) — `python/sglang/srt/utils/tensor_bridge.py`

## Relationships

- [[Community 32]] (1 shared connections)

## Source Files

- `python/sglang/srt/utils/tensor_bridge.py`

## Audit Trail

- EXTRACTED: 59 (100%)
- INFERRED: 0 (0%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*