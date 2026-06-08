# Community 392

> 16 nodes

## Key Concepts

- **ci_validate_and_cleanup_local_snapshot()** (9 connections) — `python/sglang/srt/model_loader/ci_weight_validation.py`
- **_validate_sharded_model()** (8 connections) — `python/sglang/srt/model_loader/ci_weight_validation.py`
- **_validate_weights_after_download()** (7 connections) — `python/sglang/srt/model_loader/ci_weight_validation.py`
- **_validate_safetensors_file()** (6 connections) — `python/sglang/srt/model_loader/ci_weight_validation.py`
- **_validate_pytorch_bin_file()** (6 connections) — `python/sglang/srt/model_loader/ci_weight_validation.py`
- **_cleanup_corrupted_files_selective()** (5 connections) — `python/sglang/srt/model_loader/ci_weight_validation.py`
- **ci_validate_and_clean_hf_cache()** (5 connections) — `python/sglang/srt/model_loader/ci_weight_validation.py`
- **_cleanup_corrupted_model_cache()** (3 connections) — `python/sglang/srt/model_loader/ci_weight_validation.py`
- **Validate that a safetensors file is readable and not corrupted.      Args:** (1 connections) — `python/sglang/srt/model_loader/ci_weight_validation.py`
- **Validate that a PyTorch .bin file is readable and not corrupted.      This catch** (1 connections) — `python/sglang/srt/model_loader/ci_weight_validation.py`
- **Validate that all model shards are present and not corrupted.      Args:** (1 connections) — `python/sglang/srt/model_loader/ci_weight_validation.py`
- **Selectively remove corrupted files and their blobs to force re-download.      Th** (1 connections) — `python/sglang/srt/model_loader/ci_weight_validation.py`
- **Remove entire corrupted model cache directory to force a clean re-download.** (1 connections) — `python/sglang/srt/model_loader/ci_weight_validation.py`
- **CI-specific validation and cleanup for local model snapshots.      This function** (1 connections) — `python/sglang/srt/model_loader/ci_weight_validation.py`
- **Validate downloaded weight files to catch corruption early.      This function v** (1 connections) — `python/sglang/srt/model_loader/ci_weight_validation.py`
- **Validate and clean corrupted safetensors files in HF cache before loading.** (1 connections) — `python/sglang/srt/model_loader/ci_weight_validation.py`

## Relationships

- [[Community 391]] (8 shared connections)
- [[Multi-Step Draft Attention (FP8)]] (2 shared connections)
- [[Community 532]] (2 shared connections)
- [[Community 1644]] (1 shared connections)
- [[Community 531]] (1 shared connections)
- [[Community 9661]] (1 shared connections)

## Source Files

- `python/sglang/srt/model_loader/ci_weight_validation.py`

## Audit Trail

- EXTRACTED: 54 (95%)
- INFERRED: 3 (5%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*