# Community 337

> 20 nodes

## Key Concepts

- **adaptive_spec_params.py** (9 connections) — `python/sglang/srt/speculative/adaptive_spec_params.py`
- **AdaptiveStepSlot** (8 connections) — `python/sglang/srt/speculative/adaptive_spec_params.py`
- **resolve_candidate_steps_from_config()** (6 connections) — `python/sglang/srt/speculative/adaptive_spec_params.py`
- **.update()** (5 connections) — `python/sglang/srt/speculative/adaptive_spec_params.py`
- **_load_adaptive_config()** (4 connections) — `python/sglang/srt/speculative/adaptive_spec_params.py`
- **validate_adaptive_initial_steps()** (4 connections) — `python/sglang/srt/speculative/adaptive_spec_params.py`
- **._recompute_params()** (4 connections) — `python/sglang/srt/speculative/adaptive_spec_params.py`
- **.__init__()** (4 connections) — `python/sglang/srt/speculative/adaptive_spec_params.py`
- **.max_speculative_num_draft_tokens()** (3 connections) — `python/sglang/srt/server_args.py`
- **Return the maximum draft-token count speculative decoding may use.** (1 connections) — `python/sglang/srt/server_args.py`
- **.__init__()** (1 connections) — `python/sglang/srt/speculative/adaptive_spec_params.py`
- **Adaptive speculative decoding parameters.  Adjusts speculative_num_steps at runt** (1 connections) — `python/sglang/srt/speculative/adaptive_spec_params.py`
- **Load and validate adaptive config.      Uses ``DEFAULT_ADAPTIVE_CONFIG`` when *c** (1 connections) — `python/sglang/srt/speculative/adaptive_spec_params.py`
- **Union of every BS slot's candidate steps; sizes the runtime buffers.** (1 connections) — `python/sglang/srt/speculative/adaptive_spec_params.py`
- **Require the initial step to be a candidate of some BS slot.** (1 connections) — `python/sglang/srt/speculative/adaptive_spec_params.py`
- **Tracks acceptance rate via EMA and adapts num_steps accordingly.      The core i** (1 connections) — `python/sglang/srt/speculative/adaptive_spec_params.py`
- **Update EMA with observed accept lengths. Returns True if params changed.** (1 connections) — `python/sglang/srt/speculative/adaptive_spec_params.py`
- **Recompute steps from EMA. Returns True if params changed.** (1 connections) — `python/sglang/srt/speculative/adaptive_spec_params.py`
- **# TODO: add step=0 (nospec fallback) for BS>=8 once supported.** (1 connections) — `python/sglang/srt/speculative/adaptive_spec_params.py`
- **# TODO: Consider limiting step changes to avoid overshooting.** (1 connections) — `python/sglang/srt/speculative/adaptive_spec_params.py`

## Relationships

- [[Multi-Step Draft Attention (FP8)]] (6 shared connections)
- [[CLI Arg Parsing & Deprecation]] (2 shared connections)
- [[Community 407]] (2 shared connections)

## Source Files

- `python/sglang/srt/server_args.py`
- `python/sglang/srt/speculative/adaptive_spec_params.py`

## Audit Trail

- EXTRACTED: 52 (90%)
- INFERRED: 6 (10%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*