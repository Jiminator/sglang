# Community 532

> 10 nodes

## Key Concepts

- **ci_download_with_validation_and_retry()** (9 connections) — `python/sglang/srt/model_loader/ci_weight_validation.py`
- **_find_local_hf_snapshot_dir_unlocked()** (8 connections) — `python/sglang/srt/model_loader/weight_utils.py`
- **_get_lock_file_path()** (3 connections) — `python/sglang/srt/model_loader/ci_weight_validation.py`
- **_cleanup_incomplete_blobs()** (3 connections) — `python/sglang/srt/model_loader/ci_weight_validation.py`
- **_check_index_files_exist()** (3 connections) — `python/sglang/srt/model_loader/weight_utils.py`
- **Generate a unique lock file path for download coordination.      In CI environme** (1 connections) — `python/sglang/srt/model_loader/ci_weight_validation.py`
- **Remove stale .incomplete files from the model's blobs directory.      This is li** (1 connections) — `python/sglang/srt/model_loader/ci_weight_validation.py`
- **CI-specific download with validation and automatic retry on corruption.      Thi** (1 connections) — `python/sglang/srt/model_loader/ci_weight_validation.py`
- **Check if all files listed in safetensors index files actually exist on disk.** (1 connections) — `python/sglang/srt/model_loader/weight_utils.py`
- **Find local HF snapshot directory without locking.      IMPORTANT: Caller MUST ho** (1 connections) — `python/sglang/srt/model_loader/weight_utils.py`

## Relationships

- [[Community 391]] (3 shared connections)
- [[Community 392]] (2 shared connections)
- [[Multi-Step Draft Attention (FP8)]] (2 shared connections)
- [[Community 35]] (2 shared connections)
- [[Weight Loading & EPLB]] (2 shared connections)
- [[Community 47]] (1 shared connections)
- [[Community 42]] (1 shared connections)

## Source Files

- `python/sglang/srt/model_loader/ci_weight_validation.py`
- `python/sglang/srt/model_loader/weight_utils.py`

## Audit Trail

- EXTRACTED: 23 (74%)
- INFERRED: 8 (26%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*