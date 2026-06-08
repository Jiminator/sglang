# Community 211

> 32 nodes

## Key Concepts

- **parse_dims()** (11 connections) — `python/sglang/srt/debug_utils/comparator/dims_spec/dims_parser.py`
- **axis_aligner.py** (8 connections) — `python/sglang/srt/debug_utils/comparator/aligner/axis_aligner.py`
- **compute_axis_aligner_plan()** (8 connections) — `python/sglang/srt/debug_utils/comparator/aligner/axis_aligner.py`
- **_SingletonDimUtil** (7 connections) — `python/sglang/srt/debug_utils/comparator/dims_spec/dims_parser.py`
- **_build_canonical_order()** (6 connections) — `python/sglang/srt/debug_utils/comparator/aligner/axis_aligner.py`
- **resolve_dim_names()** (6 connections) — `python/sglang/srt/debug_utils/comparator/dims_spec/dims_parser.py`
- **_semantic_names_match()** (5 connections) — `python/sglang/srt/debug_utils/comparator/aligner/axis_aligner.py`
- **DimSpec** (5 connections) — `python/sglang/srt/debug_utils/comparator/aligner/axis_aligner.py`
- **_build_side_pattern()** (5 connections) — `python/sglang/srt/debug_utils/comparator/aligner/axis_aligner.py`
- **Pair** (4 connections) — `python/sglang/srt/debug_utils/comparator/aligner/axis_aligner.py`
- **_CommentSuffix** (4 connections) — `python/sglang/srt/debug_utils/comparator/dims_spec/comment_parser.py`
- **_parse_comment_suffix()** (4 connections) — `python/sglang/srt/debug_utils/comparator/dims_spec/comment_parser.py`
- **.is_squeeze()** (4 connections) — `python/sglang/srt/debug_utils/comparator/dims_spec/dims_parser.py`
- **.sanitize_names()** (4 connections) — `python/sglang/srt/debug_utils/comparator/dims_spec/dims_parser.py`
- **_normalize_dim_name()** (3 connections) — `python/sglang/srt/debug_utils/comparator/aligner/axis_aligner.py`
- **_expand_and_skip_squeeze()** (3 connections) — `python/sglang/srt/debug_utils/comparator/aligner/axis_aligner.py`
- **dims_parser.py** (3 connections) — `python/sglang/srt/debug_utils/comparator/dims_spec/dims_parser.py`
- **.filter_out()** (3 connections) — `python/sglang/srt/debug_utils/comparator/dims_spec/dims_parser.py`
- **comment_parser.py** (2 connections) — `python/sglang/srt/debug_utils/comparator/dims_spec/comment_parser.py`
- **DimSpec** (2 connections) — `python/sglang/srt/debug_utils/comparator/dims_spec/dims_parser.py`
- **.make_name()** (2 connections) — `python/sglang/srt/debug_utils/comparator/dims_spec/dims_parser.py`
- **Check that both sides share the same semantic name set (ignoring squeeze dims).** (1 connections) — `python/sglang/srt/debug_utils/comparator/aligner/axis_aligner.py`
- **Expand DimSpecs to flat semantic names, skipping squeeze dims.** (1 connections) — `python/sglang/srt/debug_utils/comparator/aligner/axis_aligner.py`
- **Build canonical dim order following y, preferring fused representation.      Eac** (1 connections) — `python/sglang/srt/debug_utils/comparator/aligner/axis_aligner.py`
- **Build an einops pattern for one side to reach ``canonical_order``.      Fused sp** (1 connections) — `python/sglang/srt/debug_utils/comparator/aligner/axis_aligner.py`
- *... and 7 more nodes in this community*

## Relationships

- [[Community 109]] (5 shared connections)
- [[Community 55]] (2 shared connections)
- [[Community 175]] (2 shared connections)
- [[MoE Dispatch/Combine (Cutlass)]] (1 shared connections)
- [[Community 486]] (1 shared connections)
- [[Community 128]] (1 shared connections)

## Source Files

- `python/sglang/srt/debug_utils/comparator/aligner/axis_aligner.py`
- `python/sglang/srt/debug_utils/comparator/dims_spec/comment_parser.py`
- `python/sglang/srt/debug_utils/comparator/dims_spec/dims_parser.py`

## Audit Trail

- EXTRACTED: 96 (87%)
- INFERRED: 14 (13%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*