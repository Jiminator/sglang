RANKING:
1. **A. `TOKEN_BLOCK=256`**: first attempt. It is surgical, measured at 43.5 us/call, meets the 40 ms/window gate, and does not intentionally change per-token math.
2. **D. Persistent / bounded live-grid redesign**: best follow-on if this kernel remains a bottleneck, because it attacks the real 60 us dead-grid floor, but it is too large for the first gate-closing step.
3. **E. Fuse masking/top-k prep**: useful elsewhere, but separate bucket and not needed for this gate.
4. **C. Dead-store high-water mark**: low ROI after A and adds stale-score correctness risk on row reuse; keep as a rejected or low-priority note.
5. **B. Seq-adaptive block size**: no-go for the captured graph path as stated.

ANSWERS:
1. Implement **A first**. The plan gate is already met, and the project maxims favor the smallest reliable change that preserves behavior. Pair it with the harness checks and a short context/long context sweep, but do not bundle C/D into the same implementation loop.

2. The bit-identity claim is **sound as a design claim**, with one caveat: phrase it as “no numerics-design change; bit identity must be confirmed by harness.” In the current kernel, `TOKEN_BLOCK` changes the token vector width and grid partitioning. The dot reduction axis is still `LABEL_DIM_POW2`, with the same `d_offs`, same loads, same fp32 casts, same `tl.sum(..., axis=1)`, and same per-head max loop order. I do not see a semantic Triton reason for a given position’s fp32 score to change.

   The caveat is that `TOKEN_BLOCK=256` creates a separately specialized Triton kernel, and Triton does not give a useful public promise that all meta-parameter changes are bit-stable. If the harness finds drift, check generated launch meta/codegen first, possibly pin `num_warps`/`num_stages`, then decide whether recall gating is needed. If score tensors and selected indices are bit-identical, skipping recall re-gate is reasonable.

3. Pick **`TOKEN_BLOCK=256`** for this gate. `tb=128` is a nicer all-live compromise but misses the explicit Case-1 target: 60.9 us/call is about 47.5 ms/window. `tb=256` hits about 34 ms/window and is still slightly better than `tb=64` at all-live in the measurements provided. Since the gate is measured at the Case-1 op point, `128` would need either a target change or another bundled optimization, which violates the surgical-first doctrine.

4. Record C/D as follow-on notes, but do not queue them into this implementation loop. Keep **D** as the serious next-loop candidate if the DS logical-score bucket remains material after A or if long-context behavior becomes the primary target. Treat **C** as mostly not worth implementing unless later evidence shows store traffic, not launch/dead-grid overhead, is the remaining limiter.

RISKS:
- **Triton codegen bit drift**: run bitwise score comparison `tb64` vs `tb256`, selected-index identity, and cross-rank identity. If drift appears, inspect launch meta/codegen before recall gating.
- **Mid-context regression**: sweep representative seq lengths, not just 4608 and all-live. Gate against current `tb64` and total DS bucket, not only this microbench.
- **Accidental wider scope**: change only the logical-score served path; avoid altering unrelated score kernels/defaults unless separately measured.
- **Production graph capture mismatch**: ensure the static grid is captured/replayed with the new fixed block size and no adaptive branch sneaks in.
- **Future C/D stale-score hazards**: any dead-store or persistent rewrite needs adversarial row-reuse tests with shorter requests, because stale finite scores past `seq_len` would corrupt selection.
