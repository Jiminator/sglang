#!/usr/bin/env python3
"""CPU unit tests for the loop13 diagnostic DS reference selectors.

Run: python3 development/loop13/test_reference_selectors.py

Covers the diagnostic helpers added for the accuracy-ceiling study and locks in
the two load-bearing proofs:
  - the gather-then-dequant scorer is bit-equal to the full-pool dequant;
  - the cosine path's pre-division numerator (materialized-raw) is selection-equal
    to the absorbed raw-dot, so cosine-vs-rawdot isolates ONLY the normalization.
"""
import torch

from sglang.srt.layers.attention.double_sparsity.absorbed_latent import (
    absorbed_latent_score_logical,
    absorbed_latent_score_logical_fp8,
    absorbed_latent_cosine_logical_fp8,
    apply_forced_all_dense,
    dequantize_resident_latent,
    reference_rawdot_select,
    reference_cosine_select,
)
from sglang.srt.layers.attention.double_sparsity.selection_kernel import (
    select_topk_sequence_order,
)


def _fixture(seed=0, H=8, qk=32, lab=4, lora=256, T=60, bs=3, S=20):
    torch.manual_seed(seed)
    q = torch.randn(bs, H, qk)
    csel = torch.randint(0, qk, (H, lab))
    cw = torch.rand(H, lab)
    wsel = torch.randn(H, lab, lora) * 0.1
    fp8 = torch.randn(T, lora).to(torch.float8_e4m3fn)
    scl = torch.rand(T, lora // 128 if lora >= 128 else 1) + 0.5
    rtt = torch.randint(0, T, (bs, S))
    rpi = torch.arange(bs)
    seq = torch.tensor([S, S - 2, S - 5])[:bs]
    written = torch.ones(T, dtype=torch.bool)
    return dict(q=q, csel=csel, cw=cw, wsel=wsel, fp8=fp8, scl=scl, rtt=rtt,
               rpi=rpi, seq=seq, written=written, S=S, bs=bs)


def _sel_set(sel, b):
    return set(int(i) for i in sel[b].tolist() if int(i) >= 0)


def test_dequant_roundtrip():
    fp8 = torch.randn(3, 256).to(torch.float8_e4m3fn)
    scl = torch.rand(3, 2) + 0.5
    c = dequantize_resident_latent(fp8, scl)
    expect = (fp8.to(torch.float32).view(3, 2, 128) * scl.view(3, 2, 1)).view(3, 256)
    assert torch.equal(c, expect)


def test_forced_all_dense():
    sel = torch.full((2, 5), -1, dtype=torch.int32)
    sel[0, :3] = torch.tensor([4, 2, 0], dtype=torch.int32)
    sel[1, :5] = torch.tensor([9, 8, 7, 6, 5], dtype=torch.int32)
    vl = torch.tensor([3, 5], dtype=torch.int32)
    oi, ovl = apply_forced_all_dense(sel, vl, torch.tensor([4, 100]), max_top_k=5)
    assert oi[0].tolist() == [0, 1, 2, 3, -1] and int(ovl[0]) == 4
    assert oi[1].tolist() == sel[1].tolist() and int(ovl[1]) == 5  # sparse untouched


def test_gather_equals_fullpool():
    f = _fixture()
    ckv = dequantize_resident_latent(f["fp8"], f["scl"])
    s_full = absorbed_latent_score_logical(f["q"], ckv, f["wsel"], f["csel"], f["cw"],
                                           f["rpi"], f["rtt"], f["seq"], f["S"],
                                           written=f["written"], head_agg="max")
    s_gat = absorbed_latent_score_logical_fp8(f["q"], f["fp8"], f["scl"], f["wsel"], f["csel"],
                                              f["cw"], f["rpi"], f["rtt"], f["seq"], f["S"],
                                              written=f["written"], head_agg="max")
    fin = torch.isfinite(s_full)
    assert torch.equal(torch.isfinite(s_full), torch.isfinite(s_gat))
    assert torch.allclose(s_full[fin], s_gat[fin], rtol=0, atol=0)


def test_cosine_in_range_and_include_current():
    f = _fixture()
    cs = absorbed_latent_cosine_logical_fp8(f["q"], f["fp8"], f["scl"], f["wsel"], f["csel"],
                                            f["cw"], f["rpi"], f["rtt"], f["seq"], f["S"],
                                            written=f["written"], head_agg="max")
    fin = torch.isfinite(cs)
    assert cs[fin].abs().max() <= 1.0 + 1e-4
    for fn in (reference_rawdot_select, reference_cosine_select):
        sel, _ = fn(f["q"], f["fp8"], f["scl"], f["wsel"], f["csel"], f["cw"], f["rpi"],
                    f["rtt"], f["seq"], f["S"], max_top_k=4, written=f["written"],
                    head_agg="max", include_current=True)
        for b in range(f["bs"]):
            assert (int(f["seq"][b]) - 1) in _sel_set(sel, b)


def test_materialized_raw_equals_absorbed_raw():
    """The decisive single-variable proof: cosine path with normalize=False (the
    cosine numerator) is selection-equal to the absorbed raw-dot, so cosine-vs-rawdot
    isolates ONLY the normalization."""
    f = _fixture(seed=1)
    raw = absorbed_latent_score_logical_fp8(f["q"], f["fp8"], f["scl"], f["wsel"], f["csel"],
                                            f["cw"], f["rpi"], f["rtt"], f["seq"], f["S"],
                                            written=f["written"], head_agg="max")
    mat = absorbed_latent_cosine_logical_fp8(f["q"], f["fp8"], f["scl"], f["wsel"], f["csel"],
                                             f["cw"], f["rpi"], f["rtt"], f["seq"], f["S"],
                                             written=f["written"], head_agg="max", normalize=False)
    fin = torch.isfinite(raw)
    assert torch.equal(torch.isfinite(raw), torch.isfinite(mat))
    rel = (raw[fin] - mat[fin]).abs().max().item() / (raw[fin].abs().max().item() + 1e-9)
    assert rel < 1e-4, rel
    sr, _ = select_topk_sequence_order(raw, 6)
    sm, _ = select_topk_sequence_order(mat, 6)
    for b in range(f["bs"]):
        assert _sel_set(sr, b) == _sel_set(sm, b)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print(f"\nALL {len(fns)} reference-selector tests pass.")
