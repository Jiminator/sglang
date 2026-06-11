# Width-Bucketing Design Dossier

## 1. KEY CONTRACT

For DS-on normal decode, the graph-variant key is a Python tuple:

```python
graph_key = (bs_bucket, selector_width)
# example at the frozen op point: (32, 5120)
# full-width fallback example: (32, 202752)
```

`bs_bucket` is the existing padded CUDA-graph batch bucket selected from `CudaGraphRunner.capture_bs`. `selector_width` is the static DS selector score width used by `DSGraphState.max_seq_len`. It is not the runtime max sequence length.

The width-key path is enabled only under this exact runner gate:

```python
self._use_ds_selector_width_keys = (
    self.capture_forward_mode == ForwardMode.DECODE
    and not self.enable_pdmux
    and not self.is_encoder_decoder
    and bool(getattr(self.attn_backend, "enable_double_sparsity", False))
)
```

This gate keeps the feature DS-on decode only. `ForwardMode.TARGET_VERIFY`, `ForwardMode.DLLM_EXTEND`, encoder-decoder graph capture, and PDMux are structurally excluded. Width-key helper functions must not be called unless this flag is true.

Surfaces that must adopt `graph_key` for DS-on decode:

- `CudaGraphRunner.graphs`: key by `(bs, width)` for DS-on decode variants.
- `CudaGraphRunner.output_buffers`: same key as `graphs`; never reuse by `bs` alone because same `bs` has multiple captured graph outputs.
- `CudaGraphRunner.capture`: keep `capture_bs` derivation unchanged, but for DS-on decode iterate `(bs, width)` variants and store by tuple key.
- `CudaGraphRunner.capture_one_batch_size`: receive `graph_key` and `selector_width` for DS-on decode; pass both to DSA metadata capture.
- `CudaGraphRunner.can_run`: use the same pure variant-selection helper as replay. It returns false only when no valid `(bs, width)` graph exists; `seq_len > 5120` must not cause eager fallback because full width is always captured.
- `CudaGraphRunner.replay_prepare`: after the existing `bisect` over `capture_bs`, select width and store `self.graph_key`, `self.bs`, `self.raw_bs`, and `self.selector_width`.
- `CudaGraphRunner.replay`: use `self.graph_key` for `self.graphs[self.graph_key].replay()` and `self.output_buffers[self.graph_key]`. Do not reconstruct from `self.bs` alone.
- `DeepseekSparseAttnBackend.decode_cuda_graph_metadata`: metadata entries for DS-on decode are keyed by the full `graph_key`, not by `bs`. Existing backing entries such as `"page_table"`, `"cache_seqlens"`, `"cu_seqlens_q"`, `"cu_seqlens_k"`, and `"flashmla_metadata"` remain shared backing tensors.
- `DeepseekSparseAttnBackend.init_forward_metadata_capture_cuda_graph`: accept optional `graph_key` and `selector_width`. Store `DSAMetadata` at `decode_cuda_graph_metadata[graph_key]` and allocate `DSGraphState.max_seq_len=selector_width`.
- `DeepseekSparseAttnBackend.init_forward_metadata_replay_cuda_graph` and `_from_precomputed`: accept optional `graph_key`, load `decode_cuda_graph_metadata[graph_key]`, and stamp `metadata.ds_graph_state.last_replay_graph_key = graph_key`.
- `DSGraphState` ownership: one stable `DSAMetadata` object and one stable `DSGraphState` object per captured DS graph variant. “Owns” means stable keyed lifetime for those objects. It does not mean duplicating the full-context DSA backing tensors or `req_to_token`.

DS-off keys must remain exactly today’s shapes:

- Non-PDMux DS-off: plain `int` batch key, e.g. `32`.
- PDMux DS-off: existing string key, `f"{stream_idx}_{bs}"`, e.g. `"0_32"`.
- PDMux DS-on is out of this feature by gate and must keep the same PDMux string path if it exists.
- Speculative decode and encoder paths keep their current keying because `capture_forward_mode != ForwardMode.DECODE` or `is_encoder_decoder` disables width keys.

The runner discovers the width ladder through the DS config, not hard-coded inside replay:

- Extend `DoubleSparsityConfig` in `double_sparsity/config.py` with `selector_width_buckets`.
- Patch 1 exposes the plumbing but uses an empty compact list, so the runner sees only `[full_width]`.
- Patch 2 sets the DEC-2 compact list to `[5120]`.
- `DeepseekSparseAttnBackend.__init__` stores this as `self.ds_selector_width_buckets`.
- `CudaGraphRunner` reads `self.attn_backend.ds_selector_width_buckets` only when `_use_ds_selector_width_keys` is true, appends `full_width = int(self.attn_backend.req_to_token.shape[1])`, sorts/deduplicates, and guarantees full width is present.

## 2. WIDTH LADDER AND DISPATCH

The Patch 2 ladder is:

```python
compact_widths = [5120]
full_width = int(attn_backend.req_to_token.shape[1])  # 202752 at the frozen op point
selector_widths = sorted(set([5120, full_width]))
```

If `full_width <= 5120`, the ladder collapses to `[full_width]`.

Coverage is across the whole verified `capture_bs` ladder from `get_batch_sizes_to_capture`:

```text
[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96,
 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192,
 200, 208, 216, 224, 232, 240, 248, 256, 272, 288, 304, 320,
 336, 352, 368, 384, 400, 416, 432, 448, 464, 480, 496, 512]
```

Dispatch happens at the existing pre-replay bisect site in `CudaGraphRunner.replay_prepare`:

```python
raw_bs = forward_batch.batch_size
bs = self.capture_bs[bisect.bisect_left(self.capture_bs, raw_bs_or_global_batch)]

max_real_seq_len = int(forward_batch.seq_lens_cpu[:raw_bs].max())
width = first_captured_width_ge(max_real_seq_len)
graph_key = (bs, width)
```

Use real rows only: `forward_batch.seq_lens_cpu[:raw_bs]`.

Do not use padded metadata slices such as `buffers.seq_lens_cpu[:bs]`. Padded rows are not requests. `DecodeInputBuffers.populate_from_forward_batch` fills padded rows with `seq_len_fill_value`, which can force a false full-width route or carry non-request values. Width dispatch must be based only on live request rows.

No GPU sync is introduced. `forward_batch.seq_lens_cpu` is already a CPU tensor at this point; `DecodeInputBuffers.create` also keeps `seq_lens_cpu` on CPU, and current `CudaGraphRunner.replay_prepare` already uses host-visible batch metadata before graph replay.

Dispatch order is bs first, width second:

1. Existing bs bisect chooses the padded graph batch.
2. Width dispatch chooses the smallest captured width that covers real `seq_lens_cpu`.
3. `graph_key = (bs, width)` performs one dict lookup.

Do not flatten `(bs, width)` into one sorted product ladder for bisect. Batch padding and width overflow have different semantics: bs padding is runner capacity; width overflow is selector correctness and must route to full width.

At the frozen op point, raw bs 29 pads to bs 32. The served 4096 to 4608 decode window selects `(32, 5120)`. If any real row reaches 5121, dispatch must select `(32, 202752)`.

## 3. COMPACTION SCOPE

Only DS selector scratch is compacted. DSA attention metadata and page-table mapping remain full-context.

Compacted per `DSGraphState.max_seq_len = W`:

- `scratch_scores`: fp32 `[max_bs, W]`.
- `scratch_scores_bf16`: bf16 `[max_bs, W]` when `score_reduce_dtype == "bf16"`.
- `scratch_pv_mask`: bool `[max_bs, W]`.
- Radix block scratch:
  - `nblocks = ceil(W / topk_block)`, with current `topk_block=1024`.
  - `scratch_topk_block_above`: int32 `[max_bs, nblocks]`.
  - `scratch_topk_block_tie`: int32 `[max_bs, nblocks]`.
  - `scratch_topk_above_pref`: int32 `[max_bs, nblocks]`.
  - `scratch_topk_tie_pref`: int32 `[max_bs, nblocks]`.
- Width-derived scalar/vector scratch:
  - `scratch_boundary` must be initialized to `W`, not full context.
  - `DSGraphState.max_seq_len` must be `W`, because `deepseek_v2._select_topk_indices` passes it to `retrieve_topk_graph_safe`.

Per-bs but not width-proportional, still one copy per graph key:

- `selected_indices`, `valid_lengths`.
- `scratch_topk_values`, `scratch_topk_indices`, `scratch_invalid_mask`, `scratch_sorted_vals`, `scratch_throwaway_idx`.
- `scratch_topk_hist`, `scratch_topk_key_prefix`, `scratch_topk_quota`.
- `scratch_req_pool_indices`, `scratch_seq_lens`, `lp_error_scratch`.
- Selection-capture mirrors `capture_indices` and `capture_lengths`.
- `DSAMetadata.ds_topk_indices_out`.

Full-context, never compact:

- `DSAMetadata.max_seq_len_k`.
- `DSAMetadata.page_table_1`.
- `DSAMetadata.real_page_table`.
- `decode_cuda_graph_metadata["page_table"]`.
- `req_to_token` and every consumer in `page_table_adapter.logical_to_physical`.
- Page-table transform inputs for DSA attention.
- `TokenLabelTable.signatures`, `TokenLabelTable.written`, and `TokenLabelTable.scales`.

Prefix-window semantics preserve positions exactly. `_logical_score_kernel` treats `tok_offs` as logical row positions. For compact W it scores positions `[0, W)`, loads physical slots with:

```python
phys = req_to_token[req_pool_indices[b], logical_position]
```

Because dispatch guarantees `max(seq_lens_cpu[:raw_bs]) <= W`, every live logical position `0..seq_len-1` is present. No compact-to-logical inverse exists or is needed. The top-k still returns logical positions sorted ascending, and `page_table_adapter.logical_to_physical` gathers from the full `req_to_token`.

Dead-position handling:

- Positions `seq_len..W-1` remain invalid exactly as before.
- When `_store_dead` is true, `_logical_score_kernel` stores `-inf`.
- When radix top-k is active and `_store_dead` is false, `select_topk_sequence_order_triton` is sequence-bounded by `seq_lens` and never reads dead positions.
- If `per_request_valid` is present, `deepseek_v2._select_topk_indices` must pass only `per_request_valid[:, :W]` into `retrieve_topk_graph_safe`; `scratch_pv_mask[:bs, :W]` then masks invalid positions to `-inf`.

No page table, DSA metadata, or physical KV mapping may be compacted.

## 4. TRANSPORT PINNING

The two-shot pin must live on the DS score-reduce object bound to `DoubleSparsitySelector.reduce_ca`, not on the global model collective communicator.

Mechanism:

- Add a small DS-only wrapper, e.g. `PinnedDSScoreReduceCA`, around the attention TP group’s `ca_comm`.
- Bind this wrapper in `DeepseekV2AttentionMLA._bind_double_sparsity_runtime_data` instead of binding the raw `_attn_tp_group.ca_comm`.
- Existing default model collectives continue to use `_attn_tp_group.ca_comm` directly through `GroupCoordinator.all_reduce`.

The wrapper contract:

```python
class PinnedDSScoreReduceCA:
    algorithm = AllReduceAlgo.TWO_SHOT_PULL

    def __init__(self, base_ca):
        self.base_ca = base_ca

    def should_custom_ar(self, inp):
        return self.base_ca.should_custom_ar(inp)

    def custom_all_reduce(self, inp):
        return self.base_ca.custom_all_reduce(inp, override_algo=self.algorithm)
```

Patch 2 must add a per-call override path to `CustomAllReduceV2.custom_all_reduce` or an equivalent non-persistent method. Do not set `base_ca.override_algo` globally for the whole communicator. Do not use `override_shot(2)`: current `CustomAllReduceV2.override_shot(2)` only adjusts `one_shot_pull_threshold` and leaves `one_shot_push_threshold` intact, so bs buckets `<=16` at W=5120 would still select `ONE_SHOT_PUSH`.

Runtime contiguity assertion:

- The exact tensor handed to custom-AR is `bf16_view` inside `reduce_token_scores`.
- It must be a real compact allocation view: `scratch_scores_bf16[:bs, :W]` where the allocation width is exactly W.
- Assert `is_weak_contiguous(bf16_view)` before calling `custom_all_reduce`.
- A strided slice like `full_scratch[:, :5120]` must fail the assertion and must not silently fall back to NCCL.

Transport evidence log, once per `(bs, width, dtype)` bucket during capture/warmup:

```text
double_sparsity score reduce bucket:
  graph_key=(32, 5120)
  shape=(32, 5120)
  dtype=torch.bfloat16
  bytes=327680
  weak_contiguous=True
  custom_ar=True
  algorithm=TWO_SHOT_PULL
```

For any fallback, log `custom_ar=False` and `algorithm=NCCL_BF16` with the same shape/bytes/contiguity fields. In the exact M1 compact patch, fallback or one-shot on a compact bucket is a gate failure unless explicitly reclassified as value-affecting.

## 5. CAPTURE ORDER AND MEMORY BUDGET

Patch 1 keeps 52 captures: one full-width variant per bs, keyed as `(bs, full_width)` for DS-on decode.

Patch 2 captures 104 whole-model graphs: 52 bs buckets times two widths.

Capture order inside `CudaGraphRunner.capture`:

```python
for bs in reversed(self.capture_bs):              # existing largest-bs-first rule
    for width in reversed(self.selector_widths):  # full first, then 5120
        graph_key = (bs, width)
        capture_one_batch_size(bs, ..., graph_key=graph_key, selector_width=width)
```

This preserves the existing largest-bs-first memory-pool behavior and captures the largest DS scratch for a given bs before the compact variant. PDMux keeps its current stream loop and string keys; no width loop is added there.

Memory math for the verified ladder:

```text
sum(capture_bs) = 10515
full_width = 202752
compact_width = 5120
topk_block = 1024
nblocks(5120) = 5
nblocks(202752) = 198
```

Width-proportional DS selector scratch per bs, including `scratch_scores`, `scratch_scores_bf16`, `scratch_pv_mask`, and radix block/hist/quota scratch:

```text
W=5120:   about 36,956 bytes per bs bucket
W=202752: about 1,423,468 bytes per bs bucket
```

Ladder-wide totals for that scratch set:

```text
W=5120 set:   about 370.6 MiB
W=202752 set: about 14,274 MiB, about 13.94 GiB
```

Including fixed per-key top-k/output scratch (`scratch_topk_*`, `selected_indices`, `ds_topk_indices_out`) gives:

```text
W=5120 set:   about 1,130.5 MiB, about 1.10 GiB
W=202752 set: about 15,034 MiB, about 14.68 GiB
```

This matches the loop-9 M4 audit scale: full-ladder DS graph state is about 14.7 GiB, and `scratch_scores + scratch_scores_bf16` alone was about 11.9 GiB. The compact W=5120 addition is roughly 1.10 GiB including fixed per-variant scratch, far below the ~14.2 GiB recoverable graph-state headroom identified in `development/loop9/reviews/task13_m4_memory_audit.md`. Still, BL-20260530-int8-memfraction-ceiling-is-cudagraph-capture applies: capture memory is a boot-time constraint and must be measured, not assumed.

Boot-time impact:

- Capture count doubles from 52 to 104.
- Expected capture wall time can approach 2x if per-capture overhead dominates.
- Compact variants run less DS scratch work but still capture the whole model, so the expected range is roughly 1.6x to 2.0x.
- Patch 2 must publish measured before/after capture wall time, peak CUDA reserved/allocated, available memory before/after capture, and the transport bucket logs.

Shared-pool risks:

- `global_graph_memory_pool` remains shared through `get_global_graph_memory_pool` / `set_global_graph_memory_pool`.
- Do not deduplicate `output_buffers` by bs. Same bs/full and bs/compact variants can have distinct output tensor objects even if the pool reuses addresses.
- Replay must always read `output_buffers[self.graph_key]`.

## 6. TWO-PATCH SPLIT

Patch 1: keying and metadata lifetime, full-width only.

Contents:

- Add DS-on decode graph-key helper in `CudaGraphRunner`.
- Use tuple `(bs, full_width)` only when `_use_ds_selector_width_keys` is true.
- Keep `capture_bs` unchanged and capture only one width: `full_width = req_to_token.shape[1]`.
- Store `graphs[(bs, full_width)]` and `output_buffers[(bs, full_width)]`.
- Store DSA metadata at `decode_cuda_graph_metadata[(bs, full_width)]`.
- Replay stamps `DSGraphState.last_replay_graph_key = (bs, full_width)`.
- `DSGraphState.max_seq_len` remains full width, so `retrieve_topk_graph_safe` sees the same shape as today.
- DS-off, PDMux, speculative, and encoder paths stay on current keying and current metadata lookup.

“Zero behavior change” means mechanically:

- Same bs bucket selected by the existing `bisect`.
- Same selector width captured: 202752 at the frozen op point.
- Same DS scratch shapes.
- Same `reduce_token_scores` dtype and algorithm.
- Same logical selected indices and valid lengths.
- Only DS-on graph key representation and metadata key representation change.

Patch 1 gates:

- bs-1 selection-capture digest versus frozen baseline.
- op-point bs-29 selection-capture digest versus frozen baseline.
- Cross-rank bit identity through `selection_capture_tool.py verify`.
- `selection_capture_tool.py diff-digest --allow-identity-change graph_key`.
- DS-off smoke and DS-off keying invariant tests.
- Same-round Case-2 regression per AC-4.2 because `cuda_graph_runner.py` and `dsa_backend.py` are touched.

Expected identity changes:

- `graph_key`: int `32` becomes tuple/list `(32, 202752)`.
- `selector_width`: remains 202752.
- `raw_bs`, `padded_bs`, `replay_path`: unchanged.
- `indices_sha256` and `lengths_sha256`: must not change.

Patch 2: compact W=5120 allocation, dispatch, pinning, tests.

Contents:

- Add `selector_width_buckets=[5120]` config plumbing.
- Capture `(bs, 5120)` and `(bs, full_width)` for every bs bucket.
- Dispatch width from `forward_batch.seq_lens_cpu[:raw_bs].max()` after bs bisect.
- Allocate `DSGraphState.max_seq_len=5120` for compact variants.
- Slice `per_request_valid` to `:selector_width` before graph-safe retrieval.
- Keep DSA metadata/page tables full-context.
- Pin DS score reduce to `AllReduceAlgo.TWO_SHOT_PULL` through the DS-only wrapper.
- Add weak-contiguity assertion on the actual bf16 reduce tensor.
- Add transport evidence logging by bucket.
- Add boundary tests for `seq_len == 5120`, `seq_len == 5121`, op-point 4096→4608 growth, and padded rows.
- Add capture-memory and boot-time measurement artifact.

Patch 2 gates:

- bs-1 selection-capture digest versus Patch 1/frozen baseline with zero index diffs.
- op-point bs-29 selection-capture digest versus Patch 1/frozen baseline with zero index diffs.
- `selection_capture_tool.py diff-digest --allow-identity-change graph_key --allow-identity-change selector_width`.
- DS-off smoke and invariant tests.
- Same-round Case-2 regression per AC-4.2.
- Boundary tests and contiguity assertion tests.
- Transport log review: compact buckets show `TWO_SHOT_PULL`, bf16, expected bytes, `weak_contiguous=True`.

Expected identity changes:

- Op point: `selector_width` changes 202752 to 5120.
- `graph_key` changes accordingly, e.g. `(32, 202752)` to `(32, 5120)`.
- bs-1 prompts longer than 5120 tokens may correctly remain full width; declare only observed identity changes.
- `indices_sha256` and `lengths_sha256`: must not change in either patch.

## 7. DS-OFF INVARIANTS (test design)

Add unit tests around pure key helpers so DS-off behavior is proven without booting a full model.

Required tests:

- `test_cuda_graph_runner_ds_off_graph_keys_remain_plain_int`
  - Build a minimal runner object with `_use_ds_selector_width_keys=False`, `enable_pdmux=False`.
  - Assert capture/replay helper returns `32`, not `(32, width)`.
  - Assert `graphs` and `output_buffers` accept plain int keys.

- `test_cuda_graph_runner_pdmux_keys_remain_stream_bs_strings`
  - Set `enable_pdmux=True`.
  - Assert key is `f"{stream_idx}_{bs}"`.
  - Assert no tuple key is produced even if `enable_double_sparsity` is present.

- `test_cuda_graph_runner_width_gate_requires_ds_decode`
  - Parameterize DS off, `ForwardMode.TARGET_VERIFY`, `ForwardMode.DLLM_EXTEND`, `is_encoder_decoder=True`, and `enable_pdmux=True`.
  - Assert `_use_ds_selector_width_keys` is false in each case.
  - Monkeypatch the width-selection helper to raise; prove `can_run` and DS-off replay key selection do not call it.

- `test_dsa_decode_metadata_key_is_plain_bs_when_ds_off`
  - With `DeepseekSparseAttnBackend.enable_double_sparsity=False`, call `init_forward_metadata_capture_cuda_graph`.
  - Assert `decode_cuda_graph_metadata[bs]` exists.
  - Assert `decode_cuda_graph_metadata[(bs, full_width)]` does not exist.

- `test_dsa_backend_does_not_allocate_ds_graph_state_when_ds_off`
  - Monkeypatch `dsa_backend.allocate_graph_state` to raise.
  - Run DS-off `init_forward_metadata` and `init_forward_metadata_capture_cuda_graph`.
  - Assert no raise and `DSAMetadata.ds_graph_state is None`.
  - Assert `DSAMetadata.ds_topk_indices_out is None`.

- `test_spec_decode_keying_untouched_by_ds_width_helpers`
  - Exercise the helper under `ForwardMode.TARGET_VERIFY`.
  - Assert metadata lookup remains `decode_cuda_graph_metadata[bs]`.

- `test_encoder_cuda_graph_keying_untouched_by_ds_width_helpers`
  - Set `is_encoder_decoder=True`.
  - Assert keying remains current int/string behavior and width dispatch is unreachable.

PDMux/spec-decode surfaces to leave untouched:

- `CudaGraphRunner.maybe_init_pdmux`.
- PDMux capture key `f"{stream_idx}_{bs}"`.
- `DeepseekSparseAttnMultiStepBackend.init_forward_metadata_capture_cuda_graph`.
- `DeepseekSparseAttnMultiStepBackend.init_forward_metadata_replay_cuda_graph`.
- Existing speculative CUDA graph runners under `python/sglang/srt/speculative/`.

Proof is structural: the feature gate prevents tuple keys and width dispatch from executing on those paths, and tests assert the exact old key forms.

## 8. RISK REGISTER

1. Same-bs metadata overwrite across widths.
   - Cause: leaving `decode_cuda_graph_metadata[bs]` for DS-on multi-width capture.
   - Observable: op-point digest shows wrong `selector_width` or stale `graph_key`; unit test finds compact and full metadata are the same object; selection SHA mismatch.

2. Compact route taken for overflow.
   - Cause: dispatch uses `<=5120` incorrectly or reads padded rows incorrectly.
   - Observable: `seq_len == 5121` boundary test selects `(bs, 5120)` and fails; selected-index SHA differs from full-width baseline.

3. Padded rows drive width dispatch.
   - Cause: using `buffers.seq_lens_cpu[:bs]` after padding instead of `forward_batch.seq_lens_cpu[:raw_bs]`.
   - Observable: op-point digest unexpectedly stays `selector_width=202752` despite max real seq_len <=4608; transport bucket does not shrink.

4. Accidental eager fallback at op point.
   - Cause: `can_run` returns false or graph key missing after dispatch.
   - Observable: selection-capture digest has `replay_path=False`; gate fails.

5. Two-shot pin incomplete.
   - Cause: using `override_shot(2)` and leaving `one_shot_push_threshold` active.
   - Observable: transport log shows `ONE_SHOT_PUSH` for bs<=16 compact buckets; exact-regime gate rejects.

6. Strided custom-AR input.
   - Cause: allocating full-width scratch and slicing `[:, :5120]`.
   - Observable: `is_weak_contiguous` assertion fails, or transport log shows `weak_contiguous=False` / NCCL fallback.

7. DS-off key regression.
   - Cause: tuple key helper used outside `_use_ds_selector_width_keys`.
   - Observable: DS-off unit tests see tuple keys; DS-off smoke or same-round Case-2 regression fails.

8. DSA metadata/page-table compaction by accident.
   - Cause: applying `selector_width` to `page_table_1` or `DSAMetadata.max_seq_len_k`.
   - Observable: page-table shape assertions fail, DSA attention OOB/errors, Case-2 regression or boundary tests fail.

9. Output buffer collision between same-bs variants.
   - Cause: `output_buffers` keyed by bs while `graphs` keyed by tuple.
   - Observable: replay returns stale or wrong-shaped output; unit test sees missing `(bs,width)` output key; op-point digest instability.

10. Capture OOM or boot-time blowup.
   - Cause: 104 whole-model captures and added compact per-key buffers exceed boot budget.
   - Observable: boot fails during CUDA graph capture or Patch 2 memory artifact exceeds budget; BL-20260530 requires treating this as capture-memory failure, not generation OOM.

11. Hidden selection-index change.
   - Cause: compact masking, radix nblocks, or transport algorithm changes alter top-k boundary behavior.
   - Observable: any nonzero `indices_sha256` diff in bs-1 or op-point digest. The milestone is exact-by-design; nonzero selected-index diffs are a hard failure, not an accepted tolerance.
