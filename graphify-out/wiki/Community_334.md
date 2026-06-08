# Community 334

> 20 nodes

## Key Concepts

- **Req** (13 connections) — `python/sglang/srt/managers/scheduler_components/logprob_result_processor.py`
- **.add_input_logprob_return_values()** (9 connections) — `python/sglang/srt/managers/scheduler_components/logprob_result_processor.py`
- **._is_multi_item_scoring()** (8 connections) — `python/sglang/srt/managers/scheduler_components/logprob_result_processor.py`
- **LogitsProcessorOutput** (6 connections) — `python/sglang/srt/managers/scheduler_components/logprob_result_processor.py`
- **.add_logprob_return_values()** (6 connections) — `python/sglang/srt/managers/scheduler_components/logprob_result_processor.py`
- **._process_input_token_logprobs()** (5 connections) — `python/sglang/srt/managers/scheduler_components/logprob_result_processor.py`
- **._process_input_top_logprobs()** (5 connections) — `python/sglang/srt/managers/scheduler_components/logprob_result_processor.py`
- **._process_input_token_ids_logprobs()** (5 connections) — `python/sglang/srt/managers/scheduler_components/logprob_result_processor.py`
- **._calculate_relevant_tokens_len()** (5 connections) — `python/sglang/srt/managers/scheduler_components/logprob_result_processor.py`
- **.calculate_num_input_logprobs()** (4 connections) — `python/sglang/srt/managers/scheduler_components/logprob_result_processor.py`
- **._initialize_empty_logprob_containers()** (4 connections) — `python/sglang/srt/managers/scheduler_components/logprob_result_processor.py`
- **Process input token logprobs values and indices.** (1 connections) — `python/sglang/srt/managers/scheduler_components/logprob_result_processor.py`
- **Process input top logprobs.** (1 connections) — `python/sglang/srt/managers/scheduler_components/logprob_result_processor.py`
- **Process input token IDs logprobs.** (1 connections) — `python/sglang/srt/managers/scheduler_components/logprob_result_processor.py`
- **Calculate the expected length of logprob arrays based on whether multi-item scor** (1 connections) — `python/sglang/srt/managers/scheduler_components/logprob_result_processor.py`
- **Calculate the number of input logprobs based on whether multi-item scoring is en** (1 connections) — `python/sglang/srt/managers/scheduler_components/logprob_result_processor.py`
- **Check if request uses multi-item scoring.          Multi-item scoring applies to** (1 connections) — `python/sglang/srt/managers/scheduler_components/logprob_result_processor.py`
- **Incrementally add input logprobs to `req`.          Args:             i: The req** (1 connections) — `python/sglang/srt/managers/scheduler_components/logprob_result_processor.py`
- **Attach logprobs to the return values.** (1 connections) — `python/sglang/srt/managers/scheduler_components/logprob_result_processor.py`
- **Initialize logprob fields to empty lists if unset.          This is needed for p** (1 connections) — `python/sglang/srt/managers/scheduler_components/logprob_result_processor.py`

## Relationships

- [[Grammar Manager & HiCache Clear]] (9 shared connections)
- [[Model Config & Encode Server]] (2 shared connections)
- [[Model Configs & Pooler]] (2 shared connections)
- [[HiCache Controller & Radix Tree]] (2 shared connections)
- [[CLI Arg Parsing & Deprecation]] (2 shared connections)

## Source Files

- `python/sglang/srt/managers/scheduler_components/logprob_result_processor.py`

## Audit Trail

- EXTRACTED: 71 (90%)
- INFERRED: 8 (10%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*