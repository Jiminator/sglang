# Community 372

> 17 nodes

## Key Concepts

- **Tensor** (10 connections) — `python/sglang/srt/sampling/penaltylib/orchestrator.py`
- **.apply()** (7 connections) — `python/sglang/srt/sampling/penaltylib/orchestrator.py`
- **.accumulate_additive_penalties()** (5 connections) — `python/sglang/srt/sampling/penaltylib/orchestrator.py`
- **.accumulate_scaling_penalties()** (5 connections) — `python/sglang/srt/sampling/penaltylib/orchestrator.py`
- **.apply()** (5 connections) — `python/sglang/srt/sampling/penaltylib/orchestrator.py`
- **.cumulate_output_tokens()** (4 connections) — `python/sglang/srt/sampling/penaltylib/orchestrator.py`
- **.cumulate_output_tokens()** (4 connections) — `python/sglang/srt/sampling/penaltylib/orchestrator.py`
- **.get_scaling_penalties()** (4 connections) — `python/sglang/srt/sampling/penaltylib/orchestrator.py`
- **.filter()** (3 connections) — `python/sglang/srt/sampling/penaltylib/orchestrator.py`
- **Feed the output tokens to the penalizers.          Args:             output_ids** (1 connections) — `python/sglang/srt/sampling/penaltylib/orchestrator.py`
- **Apply all penalizers to the logits in-place.          Args:             logits:** (1 connections) — `python/sglang/srt/sampling/penaltylib/orchestrator.py`
- **Apply only additive (non-multiplicative) penalizers.** (1 connections) — `python/sglang/srt/sampling/penaltylib/orchestrator.py`
- **Accumulate all multiplicative penalty tensors into one, or None if none active.** (1 connections) — `python/sglang/srt/sampling/penaltylib/orchestrator.py`
- **Cumulate the output tokens.         Orchestrator will call this function to feed** (1 connections) — `python/sglang/srt/sampling/penaltylib/orchestrator.py`
- **Apply the penalizer to the logits.         Penalizers can modify the logits in-p** (1 connections) — `python/sglang/srt/sampling/penaltylib/orchestrator.py`
- **Return the accumulated scaling penalty tensor for multiplicative penalizers.** (1 connections) — `python/sglang/srt/sampling/penaltylib/orchestrator.py`
- **Filter the penalizer (tensors or underlying data) based on the indices to keep i** (1 connections) — `python/sglang/srt/sampling/penaltylib/orchestrator.py`

## Relationships

- [[Community 349]] (8 shared connections)
- [[CLI Arg Parsing & Deprecation]] (1 shared connections)
- [[Community 877]] (1 shared connections)
- [[Community 440]] (1 shared connections)

## Source Files

- `python/sglang/srt/sampling/penaltylib/orchestrator.py`

## Audit Trail

- EXTRACTED: 53 (96%)
- INFERRED: 2 (4%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*