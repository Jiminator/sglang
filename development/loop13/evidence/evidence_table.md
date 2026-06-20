# Loop 13 — Per-arm GSM8K evidence table (AC-1 / AC-4)

Run metadata: see `meta/run_meta.json` (git_sha 180f6dd6d, mask sha256 5c89c516…).
Config fixed: GLM-5.1-FP8, TP=8, page 64, fp8_e4m3 KV, seed 42, mem-frac 0.8, cuda-graph ON (piecewise off), temp 0, max_tokens 512, completion API.
Dense = 5-shot/200 (~763 tok, seq<top_k 2048). Sparse = 24-shot/150 (~4.2k tok, seq>2048). Batched = 64 threads; Serial = 1 thread.

| Arm | Mode | Radix cache | Dense | Sparse | head_agg | reduce dtype | sel-width | Notes |
|---|---|---|---|---|---|---|---|---|
| DSA (native) | batched | on | 0.975 | 0.973 | — | — | — | accuracy target; reproduces ≈0.970/0.953 |
| DSA (native) | serial | on | 0.965 | 0.947 | — | — | — | serial≈batched (no batching gap) |
| DSA-radix-off | batched | off | 0.960 | 0.940 | — | — | — | control: ≈DSA → --disable-radix-cache is output-neutral |
| production DS | batched | off | 0.620 | 0.000 | max | bf16 | 5120 | **regression reproduced** (DSA 0.975/0.973); sparse collapses to garbage @ length cap |
| production DS | serial(dense) | off | 0.655 | (collapse) | max | bf16 | 5120 | serial≈batched in dense (gap ~3.5pt); sparse collapse is mode-independent (batched 0.000) |
