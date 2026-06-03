**Review Result**

The direction is right, with three important constraints:

1. `physical -> page_table_1_flattened -> compact` is correct only if compact IDs are ordinal positions in the compact dequant buffer, not a global lookup by physical slot value. Prefix sharing makes physical slots non-unique across requests, so map request-locally: selected logical pos -> physical slot -> compact row `request_base + selected_rank`.

2. Prefix-sharing duplicates across requests are expected and safe if each request gets its own compact index space. Duplicates within one query row are not safe: `flash_mla_sparse_fwd` will attend duplicate valid indices twice. Keep selector output unique per row, and assert/dedup after physical remap.

3. Padding must be masked before dequant. `dequantize_k_cache_paged` blindly loads `page_table_1_flattened`; feeding `-1` there is invalid. Use safe physical placeholders for padded rows and keep corresponding compact indices as `-1`, or pack only valid rows.

Also: fixed padded shape is fine, but the effective selected count should exceed 2048 only for the proven 4K budget-limited regime; fill the lifted tail with `-1` elsewhere.

Deferring the alloc-free `out=` dequant/CUDA-graph path to task16 is consistent with the stated DEC-4/DEC-6 rule only if task14/15 is explicitly eager/research-gated, has recall evidence, and cannot enter production graph capture. One more required proof: run a direct `flash_mla_sparse_fwd` 4K-topk smoke/accuracy test; local tests only cover sparse prefill top-k up to 512.
