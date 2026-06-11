VERDICT: GO-WITH-CHANGES

REQUIRED_CHANGES:
- Scope AC-1.1 to the actual gated capture sizes, or document unreachable shapes. bf16 is custom-AR eligible at bs=29, but not at bs=64 with width 202752.
- Make the DS reduce helper fail loudly unless the actual backend is `CustomAllReduceV2` custom-AR, not pynccl/torch-dist/torch-symm fallback.
- Cover both DS reduce call sites through one abstraction.
- Start with preallocated fp32→bf16→custom-AR→fp32 copy-back, captured with zero replay allocations.
- Prove the final path inside production `cuda_graph_runner` before landing Phase B.
- Update AC-1.1/ledger language: savings are from bf16 byte reduction, not custom-AR being faster at equal bytes; residual cost is dead static width.

ANSWERS:
1. GO for bf16-through-coordinator at the bs29 gated point. It satisfies AC-1.1 only if the DS profile shows no `ncclDevKernel_AllReduce_Sum_f32_RING` in the DS score-reduce bucket and shows a named custom-AR kernel there. The honest attribution caveat is acceptable because AC-2 explicitly allows lower-precision reduction levers. It is not a GO for all batch sizes unless the plan scopes or handles larger ineligible shapes.

2. Default to (a), cast-and-reduce. It isolates the lossy change to reduction transport/output quantization while leaving scorer and top-k storage fp32. Native bf16 scores should be a second, separately gated change if cast overhead is too expensive. Note that (a)’s cast overhead likely moves the expected bucket above 82k us/window unless the benchmark already includes it.

3. Custom-AR v2 two-shot looks low risk for cross-rank divergence: it reduces in a fixed rank order and writes each rank-owned shard back to every rank’s output buffer. Still verify early with an 8-rank test at the real DS shape, bf16 dtype, eager and graph replay, comparing reduced-score bytes and selected-index tensors across all ranks. Include `-inf` masks and near-tie score cases.

4. Standalone graph capture is enough for feasibility, not for landing. Phase B must prove production `cuda_graph_runner` capture/replay using the real DS scratch tensors, actual bind path, copy-back, both reduce sites, named bf16 custom-AR kernel, and zero replay allocations.

5. Yes. AC-1.1 should be reframed around the corrected 23.5MB fp32 reality: fp32 custom-AR is ineligible under the 16MB pull cap at bs29, and bf16 only works while `bs * max_seq_len * 2 <= max_size`. Per DEC-1, unreachable fp32/custom-AR expectations should be downgraded to documented trend reduction with the measured backend, dtype, shape, and dead-width remainder in the ledger.

RISKS:
- bf16 quantization changes selected sets or recall. Mitigation: land only under AC-2 recall and exact cross-rank selected-index gates.
- Backend silently changes with shape, flags, or symmetric-memory settings. Mitigation: require-CA guard plus nsys ledger.
- Graph replay allocates through accidental `.to()` or temporary creation. Mitigation: preallocate bf16 scratch and use captured `copy_`.
- Larger batch capture sizes fall back to NCCL. Mitigation: scope the AC or add a documented DEC-1 downgrade per shape.
