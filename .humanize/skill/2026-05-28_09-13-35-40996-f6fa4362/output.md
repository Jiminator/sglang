CORE_RISKS:
- The mask is not merely “missing”; mask generation may be the real blocker. [calibrate.py](/sgl-workspace/sglang/python/sglang/srt/layers/attention/double_sparsity/calibrate.py:258) says `--tp > 1` is informational, then loads the whole HF model onto one CUDA device. The draft’s “OOM -> bump TP=2” fallback will not work without changing calibration.
- Benchmark duration is understated. [benchmark.sh](/sgl-workspace/sglang/development/benchmark.sh:45) defaults to `TRIALS=3`, conc `16 32 64`, `120s` warmup, `600s` measurement, with a hard duration guard at [line 93](/sgl-workspace/sglang/development/benchmark.sh:93). Full DS+DSA AC-11 is about 216 minutes of measured benchmark time alone.
- Smoke artifact counts are inconsistent: draft says “one DS JSON + one DSA JSON”, then “six JSONLs”, while scripts default to 9 JSONLs per side.
- Radix-off DS vs radix-on DSA is not publishable by the comparator. [benchmark_compare.py](/sgl-workspace/sglang/development/benchmark_compare.py:241) requires `disable_radix_cache` to match, and AC-11 explicitly refuses radix mismatch.
- The “2-node cluster” story is not represented in the scripts. [serve_double_sparsity.sh](/sgl-workspace/sglang/development/serve_double_sparsity.sh:55) and [serve_native_nsa.sh](/sgl-workspace/sglang/development/serve_native_nsa.sh:46) are single-node TP=8 launchers. If the actual target is 2 nodes / 16 GPUs, this plan is missing the core serving topology.
- Non-trivial DS selection is not proven by radix label capture. `bench_serving` does not currently aggregate `meta_info["double_sparsity"]` into `selected_tokens_mean` / `dense_fallback_total`; [bench_serving.py](/sgl-workspace/sglang/python/sglang/bench_serving.py:1719) writes generic benchmark fields only.

MISSING_REQUIREMENTS:
- Define separate commands for smoke vs AC-11. A smoke needs explicit `TRIALS=1`, `CONCURRENCIES=...`, and possibly shorter `MEASUREMENT_WINDOW_S`; otherwise operators will run the full AC-11 sweep.
- Require `MODEL_PATH=/cluster-storage/models/deepseek-ai/DeepSeek-V3.2`; both serve scripts default to `deepseek-ai/DeepSeek-V3.2`, which may trigger a download or wrong revision.
- Specify where the mask is generated, copied, and validated on both nodes. `/models/dsv32-fp8-channel-mask.safetensors` must be readable by every DS process, with a recorded SHA.
- Add calibration preflights: `transformers`, `datasets`, `safetensors`, HF dataset cache/network, free disk, GPU memory, and whether loading DeepSeek-V3.2 in bf16 is feasible.
- Clarify quality-smoke placement. Two TP=8 servers cannot both run on one 8-GPU node; either run DS/DSA on separate nodes with matching checkout, or generate/store DSA references sequentially.
- Define the AC-10 flip mechanism. `record_radix_fixture_passed(...)` exists in [validator.py](/sgl-workspace/sglang/python/sglang/srt/layers/attention/double_sparsity/validator.py:238), but the draft does not say how it runs before `validate_double_sparsity`.
- Treat mask `dtype=fp8_e4m3` as metadata, not tensor dtype. The artifact tensors are `channel_selection:int32` and `channel_weights:float32` per [channel_mask.py](/sgl-workspace/sglang/python/sglang/srt/layers/attention/double_sparsity/channel_mask.py:12).

TECHNICAL_GAPS:
- The `_write_token_labels` fix is necessary and correctly identified. Current code references undefined `forward_batch` at [dsa_backend.py:1582](/sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa_backend.py:1582), and call sites at [1664](/sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa_backend.py:1664), [1863](/sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa_backend.py:1863), [2387](/sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa_backend.py:2387) do not pass it.
- The AC-10 evidence requires both full fixtures: label capture and FP8 scale stability. The continuation fixture is explicitly “pre-flight only” in [test_dsv32_radix_cache_fixture.py](/sgl-workspace/sglang/test/manual/test_dsv32_radix_cache_fixture.py:1).
- CUDA graph evidence is underspecified. The scripts record `disable_cuda_graph`, but the real DS graph machinery is only unit-tested; there is no explicit H200/V3.2 capture-replay run in the draft.
- Chunked-prefill probing should happen before full benchmarks. If it fails after AC-8, all benchmark artifacts were collected under the wrong operating point.
- The current serve scripts have no easy `--chunked-prefill-size -1` knob, so the “disable on both sides if probe fails” branch needs launcher support.
- Smoke prompts must exceed `top_k=2048` if they are meant to prove sparsity. Short quality prompts can validly select all tokens, making DS effectively dense.

ALTERNATIVE_DIRECTIONS:
- Split Loop 5 into three gates: calibration feasibility, DS smoke, then comparable MVP. This prevents burning cluster time before proving the mask pipeline actually works.
- Add a dedicated `benchmark_smoke.sh` or documented env block instead of overloading AC-11 scripts. Tradeoff: one more script, but far less operator ambiguity.
- Run benchmarks sequentially on the same node for comparability; reserve the second node for simultaneous DS/DSA quality smoke if needed.
- If full-model calibration cannot fit, change direction early: build calibration from the SGLang TP=8 loaded model/runtime hooks, or use a sharded/offloaded calibration path. That is more code, but it matches the actual model size.
- Close AC-10 with an explicit launcher/CLI mechanism for the fixture artifact, not just an env override. Final radix-on evidence should boot without `SGLANG_DS_RADIX_OVERRIDE=1`.

QUESTIONS_FOR_USER:
- Is the MVP target single-node TP=8, or true two-node / 16-GPU serving?
- For Smoke MVP, do you want one concurrency or conc `16/32/64`?
- For smoke benchmarks, is a shortened measurement window acceptable, clearly labeled non-AC-11?
- Should radix-off DS be compared against radix-off DSA for smoke, or should comparator output wait until AC-10 is closed?
- Are code changes to the calibration path in scope if `calibrate.py --tp 1` cannot load DeepSeek-V3.2?
- How should `record_radix_fixture_passed` be wired before validation: CLI flag, launcher init module, or code-side policy?

CANDIDATE_CRITERIA:
- Calibration PASS: real, non-synthetic calibration writes the mask; `load_channel_mask` passes; metadata has `dtype=fp8_e4m3`, `page_size=64`, `label_dim=16`, `head_dim=128`; content SHA recorded.
- DS boot PASS: `/get_server_info` shows DS enabled, TP=8, FP8 KV, page=64, expected radix setting, and logs show `bind_runtime_data` for all layers/ranks.
- AC-10 producer bug PASS: capture-enabled `/generate` returns non-empty `meta_info["double_sparsity_radix_capture"]` with no capture error.
- Non-trivial DS PASS: a prompt longer than `top_k` returns `meta_info["double_sparsity"]` with `dense_fallback=0` and `selected_tokens < total_tokens`.
- Smoke bench PASS: explicitly configured smoke run produces the expected JSONL count, each duration meets its configured window, and sidecars exist.
- AC-11 PASS: 9 DSA + 9 DS JSONLs, each `duration >= 600`, sidecars valid, radix settings match, comparator exits 0.
- AC-10 radix PASS: label-capture fixture PASS, FP8 scale-stability fixture PASS, DS server boots radix-on without override.
- Quality PASS: smoke thresholds pass; final MVP requires AC-12 NIAH 4K/16K/64K and MMLU gates pass.
- Chunked-prefill PASS: probe result is recorded before full benchmark; if failed, both DS and DSA sidecars prove `chunked_prefill_size=-1`.
- CUDA graph PASS: final evidence includes actual capture/replay success or a clearly recorded failure, not only `disable_cuda_graph=False`.
