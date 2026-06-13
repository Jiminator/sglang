# DS-Offload Rejection Memo

## Verdict

Reject DS-Offload for TP=8 SLO serving at this op-point. Document it; do not build it.

## What The Paper Repo Actually Does

The paper repo's DS-Offload path keeps the selector label cache resident while offloading the full K/V cache. In `development/past_implementations/DoubleSparse/offloading/model.py`, `KVCache` allocates `k_cache_cpu` and `v_cache_cpu` as CPU pinned tensors (`pin_memory=True`) and allocates GPU staging buffers sized by `heavy_const` (`k_cache_gpu` / `v_cache_gpu`) plus `k_label` (`model.py:82-93`). The generation path creates caches inside the CUDA device context (`generate.py:183-184`), and the model itself is moved to CUDA before use (`generate.py:275`), so `k_label` is the resident dense-read selector structure.

Decode computes label scores against the full `k_label` table with a dense matmul over cached positions, then top-k selects `heavy_const` token rows (`model.py:368-375`). The selected K/V rows are gathered from pinned CPU memory using DGL `gather_pinned_tensor_rows`, imported at `model.py:10` and called for both K and V at `model.py:141-142`. Writes into the CPU cache happen in `update()` via `k_val.cpu()` / `v_val.cpu()` assignments (`model.py:106-119`), so this path is research-grade rather than a production async offload pipeline. The survey says the same thing: DS-Offload keeps labels GPU-resident and small while full `K,V` lives in pinned CPU memory and selected rows are gathered per step (`development/past_implementations/study/00-survey.md:26-29`, `:86-97`); the SGLang fork did not port this offload path (`:119`).

## Why The Premise Inverts For MLA+DS

DS-Offload works in the paper setting because the dense-read tensor is small and the sparse-read tensor is large. For Llama-style MHA, `K_label` is `H*r` with `r << D`, while full K/V is `2*H*D`; keeping labels on GPU and fetching selected full K/V rows can reduce GPU cache bytes substantially.

Our GLM-5.1-FP8 MLA+DS op-point reverses that structure. The dense-read tensor is the TokenLabelTable, 5.29 GB/rank, read every decode step to score every cached token. The sparse-read tensor is the fp8 latent payload, already compact, already pool-resident, and only 576 B/token/layer for the gathered latent payload used in the arithmetic. Offloading the sparse side no longer moves the expensive structure out of GPU memory; it moves the compact structure onto PCIe and adds a per-step gather on the critical path.

## PCIe Arithmetic

Measured H200 pinned-memory bandwidth for the node is 55.5 GB/s H2D and 55.1 GB/s D2H with a 1 GiB buffer over 20 iterations (`development/loop11/runs/20260613_m0/pcie_bandwidth.txt:1-4`). Using the requested 55.1 GB/s bound:

```text
per request per step = 2048 selected tokens * 78 layers * 576 B
                     = 92,012,544 B ~= 92 MB

bs30 = 2.76 GB/step / 55.1 GB/s ~= 50 ms/step
bs64 = 5.89 GB/step / 55.1 GB/s ~= 107 ms/step
```

The decode budget is about 33 ms/step for a 30 tok/s/request SLO. The bs30 transfer alone exceeds the whole step budget, and bs64 is more than 3x the budget.

This is an optimistic lower bound: it assumes sequential pinned-memory bandwidth, zero gather overhead, zero contention with weights/activations traffic, and ignores row-granularity inefficiency from the DGL-style gathered access pattern. Real offload latency would be worse.

## Why The Table Cannot Be Offloaded

The table is not a sparse payload. The selector must score every cached token every decode step, the same dense `Q_label * K_label^T` pattern shown in the paper repo (`model.py:370`, with next-layer prefetch also reading `next_k_label` at `model.py:385-389`). Offloading our 5.29 GB/rank TokenLabelTable would require a dense full-table read at decode rate. At 30 steps/s, that implies about 159 GB/s/rank of sustained PCIe traffic before compute; one full-table transfer alone is about 96 ms at 55.1 GB/s. That is structurally incompatible with the SLO.

## Revisit Condition

Only revisit offload if it is a genuinely new design, such as a context-horizon hybrid that keeps a recent window resident and accepts a new lossiness discussion; that is outside this memo's scope.
