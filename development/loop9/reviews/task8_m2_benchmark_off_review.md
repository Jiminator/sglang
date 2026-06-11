VERDICT_A: accept-disqualification

VERDICT_B: triton-now-aot-followon

DESIGN_NOTES:
- The B scheme is correct in principle: exact threshold key `T`, `n_above`, `n_tie`, and tie quota `r = K - n_above` implement score-desc / pos-asc selection if tie rank is computed by ascending logical position.
- Do not rely only on `seq_len` for validity. Exclude `-inf` inside the live prefix from histograms, counts, tie rank, and emission. `valid_lengths = min(num_finite, K)`.
- For `num_finite <= K`, bypass threshold admission and emit all finite positions ascending, padded with `-1`.
- Per-block prefix + in-block cumsum is deterministic if blocks cover contiguous logical-position ranges and in-block cumsum is over ascending offsets. Prefix selected counts, not just raw above/tie counts.
- Canonicalize numeric zero before sortable-key generation, or `-0.0` and `+0.0` will break the “equal score ties by pos” contract. NaNs should either be impossible/asserted or mapped invalid.
- Avoid `[bs, 65536]` histograms as the default simplification; clear/scan/scratch cost likely dominates at live windows. If bf16-in-fp32 is a hard served invariant, consider a guarded 16-bit key path using two 8-bit rounds, not a large 16-bit histogram.

ANSWERS:
1. Accept A as disqualified. The raw 17.7 us/call number is useful evidence, but the nondeterministic boundary-bin admission violates the hard cross-rank gate. A wrapper repair is already measured or estimated out of budget; building it would spend loop time on a known nonstarter.
2. Use Triton multi-pass now, with fused AOT as follow-on. The gate is reachable, Triton is already production-acceptable in this DS path, and forcing an sgl-kernel source build mid-loop is not justified for M2.
3. No fundamental hole in B, but finite-score filtering, zero canonicalization, deterministic position-ordered block layout, and selected-count prefixing must be explicit.
4. Yes. If Triton lands repeatably at ~60-80k/window and passes equivalence, cross-rank, graph replay, and captured-shape benchmarks, M2 should land. Record the ~20-30 us fused AOT kernel as headroom toward stretch, not an M2 blocker.

RISKS:
- Tight margin near 80k: mitigate by benchmarking all captured shapes, not one median.
- Triton launch/scratch overhead: preallocate scratch, JIT before capture, verify zero replay allocations.
- Tie-heavy bf16 boundary cases: add plateau, all-equal, `num_finite < K`, dense `<=K`, and masked-in-prefix tests.
- Future fp32 score path: either keep 32-bit-key correctness or guard any bf16-key shortcut behind an explicit invariant.
