CORE_RISKS:
- The “regenerate mask and inherit fixture if SHA matches” path is probably unrealistic. `validator.py:452-475` hashes the entire mask file, while `calibrate.py:824-843` writes a fresh `created_at` into safetensors metadata. Same tensor content can still produce a different full-file SHA.
- Repointing `CHANNEL_MASK_PATH` is not equivalent to recreating `/models/...`: the fixture fingerprint pins both `channel_mask_path` and full mask SHA (`validator.py:468-475`, fixture lines 9-15). A new path forces re-mint even if bytes match.
- Calibration reproducibility is under-specified. `calibrate.py` explicitly says Pile-val streaming order is not seed-deterministic unless an explicit local corpus is used. Treat re-mint as the default unless the exact corpus artifact exists.
- AC-11 does not currently enforce the draft’s aggregate throughput criterion. `benchmark_compare.py` gates per-request decode TPS p50 and P99 TTFT, not aggregate TPS.
- AC-11 can publish without hard DS no-op/fallback proof. The comparator parses `dense_fallback_total`, but required trial metrics are only TPS and TTFT (`benchmark_compare.py:812-823`).
- The “3 independent trials” are not independent if scripts reuse the same seed per concurrency for every trial (`benchmark.sh:48-60`). That is repeated-run stability, not independent workload sampling.
- Fresh-node performance drift is a real risk: sequential DS/DSA sweeps can fold thermal state, driver/kernel cache state, noisy neighbors, or clock differences into the DS-vs-DSA gap.

MISSING_REQUIREMENTS:
- Add a calibration provenance artifact: command, corpus/token-block digest, model/tokenizer snapshot, package versions, CUDA/driver, env, dry-run placement, tensor content hash, and full file SHA.
- Record observed radix reuse for every SLO trial. DEC-12 only authorizes production-representative reuse; configured GSP shape does not prove actual cached-token distribution.
- Define the SLO metric exactly: per-request p50 decode TPS vs aggregate output throughput vs p10/min. `SLOS.md` wording is ambiguous; comparator currently uses per-request median.
- State what happens if DSA misses the absolute SLO on the new node. Is DS still a hard fail, or is the op-point itself invalid?
- Require DS/DSA run ordering discipline, ideally paired or ABBA by concurrency/trial, with server logs kept for each boot.
- Preserve raw JSONLs, sidecars, server logs, fixture, mask hash output, `/server_info`, and benchmark commands as close-out evidence.
- Scope UX cleanup beyond serve scripts: `server_args.py` help text, benchmark comments, comparator comments, and package docstrings still carry DeepSeek/dsv32 or TokenLabelTable-era assumptions.

TECHNICAL_GAPS:
- `calibrate.py --tp` is informational; it does not run SGLang TP calibration. The plan should not imply TP=8 calibration semantics without placement/hook evidence.
- Serve defaults are dangerous today: `serve_double_sparsity.sh:29-40` and `serve_native_nsa.sh:36` still default to DeepSeek/dsv32 paths.
- `server_args.py:6090-6104` still advertises DeepSeek/dsv32 in user-facing CLI help, so a “production UX” pass limited to docs/scripts is incomplete.
- `benchmark_compare.py` ignores `mem_fraction_static` cross-side and its comments still describe DS mem 0.6 plus TokenLabelTable. If DS=0.8 and DSA=0.85 is intentional, make that a declared production-envelope comparison.
- Capacity exactness may be over-specified for fresh hardware. `504640` and `410560` are useful references, but hard gates should center on bs cap ≥64, graph capture, and no material DSA regression.

ALTERNATIVE_DIRECTIONS:
- Make “fresh mask + DEC-12 re-mint” the primary path; use inherited authorization only if the exact old file is recovered.
- Publish two comparisons: production envelope DS0.8-vs-DSA0.85, plus a same-mem sensitivity run if the headline is contested.
- Use deterministic per-trial seed families if you want independent trials; otherwise call them same-workload reruns and report min/median/max.
- Longer term, consider pinning fixture auth to tensor `content_sha256` plus config rather than full safetensors bytes, but that is an owner decision because current code intentionally pins full file/path.
- If AC-4 fails, profile the graph-mode served path first; only pull q4/q5 if the trace proves top-k/gather/score-reduce is the bottleneck.

QUESTIONS_FOR_USER:
- Is the exact old GLM mask file or exact calibration corpus recoverable?
- Should fixture authorization continue to pin full file bytes and path, or is tensor-content identity acceptable?
- Is the headline comparison production-envelope or same-resource?
- Is the 30 TPS bar per-request median decode TPS, aggregate output throughput, or both?
- Are the three trials meant to be independent workloads or repeated measurements of the same workload?
- Should CLI ABI cleanup be explicitly out of scope for loop 11b?

CANDIDATE_CRITERIA:
- AC-0a: Mask provenance records corpus/token digest, command, env, model snapshot, package versions, tensor content hash, full file SHA, and inherited-vs-reearned authorization.
- AC-0b: Inherited fixture only if model path, mask path, full mask SHA, TP, page size, KV dtype, and selector mode match exactly.
- AC-0c: DS radix-on boots without `SGLANG_DS_RADIX_OVERRIDE`; logs artifact authorization; `/server_info` confirms GLM, radix on, fp8 KV, graph on, custom all-reduce on.
- AC-11: Comparator/report includes per-conc DS/DSA decode p50, aggregate throughput, P99 TTFT, achieved concurrency, observed prefix hit, and DS/DSA ratios.
- AC-11: Missing DS fallback/no-op observability or cached-token distribution is an input refusal, not “unknown but publish.”
- AC-4: bs64 same-batch graph-mode tax guard declares radix state, seq shape, warmup, mem fractions, and reports ratio plus bs30 reference.
- AC-UX: No stale DeepSeek/dsv32 defaults or help text remain in the GLM production path; diagnostic knobs are labeled; no CLI ABI change without owner approval.
