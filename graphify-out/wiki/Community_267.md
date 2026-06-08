# Community 267

> 24 nodes

## Key Concepts

- **PoolsideV1Detector** (25 connections) — `python/sglang/srt/function_call/poolside_v1_detector.py`
- **.parse_streaming_increment()** (11 connections) — `python/sglang/srt/function_call/poolside_v1_detector.py`
- **ToolCallItem** (7 connections) — `python/sglang/srt/function_call/poolside_v1_detector.py`
- **Tool** (7 connections) — `python/sglang/srt/function_call/poolside_v1_detector.py`
- **.detect_and_parse()** (7 connections) — `python/sglang/srt/function_call/poolside_v1_detector.py`
- **._convert_param_value()** (6 connections) — `python/sglang/srt/function_call/poolside_v1_detector.py`
- **StreamingParseResult** (6 connections) — `python/sglang/srt/function_call/poolside_v1_detector.py`
- **._close_current_call()** (5 connections) — `python/sglang/srt/function_call/poolside_v1_detector.py`
- **Any** (5 connections) — `python/sglang/srt/function_call/poolside_v1_detector.py`
- **._reset_call_state()** (4 connections) — `python/sglang/srt/function_call/poolside_v1_detector.py`
- **._get_param_schema()** (4 connections) — `python/sglang/srt/function_call/poolside_v1_detector.py`
- **._find_name_boundary()** (4 connections) — `python/sglang/srt/function_call/poolside_v1_detector.py`
- **._consume_arg_key()** (3 connections) — `python/sglang/srt/function_call/poolside_v1_detector.py`
- **._is_partial_tag()** (3 connections) — `python/sglang/srt/function_call/poolside_v1_detector.py`
- **.__init__()** (1 connections) — `python/sglang/srt/function_call/poolside_v1_detector.py`
- **.has_tool_call()** (1 connections) — `python/sglang/srt/function_call/poolside_v1_detector.py`
- **.supports_structural_tag()** (1 connections) — `python/sglang/srt/function_call/poolside_v1_detector.py`
- **Detector for poolside Laguna-XS.2 (poolside_v1 series) tool-call wire format.** (1 connections) — `python/sglang/srt/function_call/poolside_v1_detector.py`
- **Reset per-call FSM scratch fields. Called when entering a new         <tool_call** (1 connections) — `python/sglang/srt/function_call/poolside_v1_detector.py`
- **Consume `<arg_key>K</arg_key>`, set `current_pending_key` to K.         Returns** (1 connections) — `python/sglang/srt/function_call/poolside_v1_detector.py`
- **Emit the closing `}` (or `{}` for zero-arg) for the active call,         advance** (1 connections) — `python/sglang/srt/function_call/poolside_v1_detector.py`
- **Coerce a raw arg_value string per schema; fall back to raw on failure.** (1 connections) — `python/sglang/srt/function_call/poolside_v1_detector.py`
- **Earliest of `\\n`, `<arg_key>`, `</tool_call>`. -1 if none.** (1 connections) — `python/sglang/srt/function_call/poolside_v1_detector.py`
- **True if slice_ is a strict prefix of any known tag — i.e. more         bytes mig** (1 connections) — `python/sglang/srt/function_call/poolside_v1_detector.py`

## Relationships

- [[Community 40]] (21 shared connections)
- [[Community 58]] (5 shared connections)
- [[Anthropic/OpenAI API Entrypoints]] (1 shared connections)
- [[Community 167]] (1 shared connections)
- [[Vision-Language Model Configs]] (1 shared connections)

## Source Files

- `python/sglang/srt/function_call/poolside_v1_detector.py`

## Audit Trail

- EXTRACTED: 80 (75%)
- INFERRED: 27 (25%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*