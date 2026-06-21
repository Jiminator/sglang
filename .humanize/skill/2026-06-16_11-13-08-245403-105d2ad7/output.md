AGREE:
- Broad structure is right: M-A op-point rebuild, M-B tax guard + AC-11 sweep, M-C UX cleanup, close-out evidence.
- Pass-1 points were mostly addressed: radix re-mint is default, full-file SHA/path pinning is acknowledged, aggregate throughput is not falsely made a comparator gate, same-seed trials are described as repeated measurements, and CLI ABI changes are out of scope.
- AC-2/AC-3 are now mostly verifiable against current `benchmark_compare.py --ac11`: per-request decode TPS p50, P99 TTFT, 30 TPS / 22 s absolute bars, mem asymmetry recorded-not-matched, radix-state matched.
- DEC-1 through DEC-6 are the right main human decisions.

DISAGREE:
- Not converged yet.
- The Lower Bound is inconsistent: it allows “radix-OFF if it cannot be authorized,” but AC-9 and the headline require DS and DSA both radix-ON. A radix-OFF sweep can be diagnostic only, not a HARD AC-2/AC-3 DS-vs-DSA verdict.
- The calibration path is under-specified and the example command is invalid: `calibrate.py` requires `--dtype`.
- AC-9 pairing is too vague. “ABBA where feasible” is not an enforceable plan with one TP=8 server at a time.
- AC-0.2 uses vague `/get_server_info confirms ...` language. The endpoint is legacy, and the exact fields are not named.
- AC-5 says “no-op / dense-fallback” but only hard-checks `dense_fallback_total`. If selection counters exist, `selected_tokens == total_tokens` must also refuse publication.
- Task IDs are confusing: candidate `task7/task8` no longer map to inherited loop-11 `task8/task9`.

REQUIRED_CHANGES:
- AC-0.1/task4: add calibration preflight and fix the command. Require `--dtype bfloat16 --kv-cache-dtype fp8_e4m3`, corpus availability or explicit default-streaming choice, token-block digest, `--dry-run-blocks 1` placement/dtype report, package/CUDA/driver versions, free-GPU/disk check, and OOM/allocator policy. Treat calibration as exclusive-node, long-running work.
- AC-0.1/AC-5: spell out recall comparability. The frozen baseline is the GLM-5.1 fp16-label-era quality baseline; the plan must either explicitly keep that owner-approved value-affecting gate with length/sample comparability checks, or define a new served-fp8 baseline. Do not just say “recall vs baseline.”
- AC-0.2/AC-0.3/AC-9: replace vague server-info checks with exact `/server_info` keys: `model_path`, `tp_size`, `page_size`, `kv_cache_dtype`, `enable_double_sparsity`, `double_sparsity_config`, `double_sparsity_radix_fixture_artifact`, `disable_radix_cache=false`, `disable_cuda_graph=false`, `disable_custom_all_reduce=false`, `mem_fraction_static`, `max_running_requests`, `cuda_graph_max_bs`, plus `internal_states[*].memory_usage.token_capacity` and effective running-request capacity.
- Path Boundaries: state that if radix-on cannot be re-minted, AC-0/AC-9 fail and the loop reports “no publishable radix-on verdict.” Any radix-OFF numbers are diagnostic.
- AC-9/task8: define the one-server run order. Example: per concurrency, alternate DSA/DS boots by trial with logged order, boot logs, server-info snapshots, and thermal/clock notes; if runs are block-scheduled by side, label them unpaired and require explicit acceptance before HARD verdict.
- AC-5/task8: require no-op observability fields to be present. Publication requires `dense_fallback_total == 0` and `selected_tokens_mean < total_tokens_mean` or an equivalent DS sparse-selection proof; missing fields are input refusal.
- Task Breakdown: preserve inherited names by labeling the measurement tasks “loop11 task8 tax guard” and “loop11 task9 AC-11 sweep,” even if local task numbers differ.
- AC-UX: explicitly include stale benchmark/comparator comments and references: `benchmark.sh`, `benchmark_baseline.sh`, `benchmark_compare.py`, `CLIENT_SLOS.md`, “independent trials,” mem-0.6, DeepSeek/dsv32, and TokenLabelTable-era text.

OPTIONAL_IMPROVEMENTS:
- Add a short explicit exclusion for `SLOS.md` deferred 128k ISL / 1024 OSL so “per SLOS.md” cannot be misread.
- Prefer `/server_info` in docs/runbook while allowing scripts to keep `/get_server_info` if compatibility is intentional.
- Add artifact filename conventions for provenance, run-order ledger, and per-trial reuse summaries.

UNRESOLVED:
- DEC-1: full-file/path fixture pinning vs tensor-content authorization.
- DEC-2: production-envelope headline only vs same-mem sensitivity.
- DEC-3: whether aggregate throughput is descriptive or a HARD gate.
- DEC-4: same-seed repeated trials vs independent seed family.
- DEC-5: confirm UX Category A/B only, no ABI/schema change.
- DEC-6: if native DSA misses 30 TPS / 22 s on the fresh node, invalidate the op-point or still judge DS.
