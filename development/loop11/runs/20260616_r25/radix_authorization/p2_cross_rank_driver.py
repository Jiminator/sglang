"""R25 PROBE 2 — cross-rank selection identity, WITHIN a single path.

For ONE booted DS path (radix-ON or radix-OFF, set by the caller's boot), issue a
single deterministic decode request with selection_capture=true (EAGER). Every TP
rank dumps per-(layer, decode-step) selected indices + valid_lengths to
<capdir>/rank{r}_step{s}.pt. This driver asserts that all `--expect-ranks` ranks
produced BYTE-IDENTICAL selection (the full [num_layers, bs, max_top_k] indices and
[num_layers, bs] lengths tensors) at every captured decode step, taking rank 0 as
the reference.

PASS = every rank present AND every rank's per-step selection SHA == rank0's per-step
selection SHA, for every step. Reports per-rank SHA agreement. FAIL-CLOSED: any
missing rank, any step where SHAs differ, or zero captured steps => FAIL.

This is the cross-rank-identity field only; it does NOT compare cold vs warm (that is
the value-neutral-flips characterization, probe 5, not a gate). Run once per path.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys

import requests
import torch

PORT = int(os.environ.get("PORT", "30000"))
BASE = f"http://127.0.0.1:{PORT}"
CAPDIR = os.environ.get(
    "SGLANG_DS_SELECTION_CAPTURE_DIR", "/sgl-workspace/sglang/.sglang_ds_selcap"
)


def _gen(prompt: str, max_new_tokens: int = 4):
    r = requests.post(
        f"{BASE}/generate",
        json={
            "text": prompt,
            "sampling_params": {
                "max_new_tokens": max_new_tokens,
                "temperature": 0,
                "ignore_eos": True,
            },
        },
        timeout=600,
    )
    r.raise_for_status()
    return r.json()


def _ranks_present(capdir):
    ranks = set()
    for f in glob.glob(os.path.join(capdir, "rank*_step*.pt")):
        base = os.path.basename(f)
        r = int(base.split("rank", 1)[1].split("_", 1)[0])
        ranks.add(r)
    return sorted(ranks)


def _steps_for_rank(capdir, rank):
    files = glob.glob(os.path.join(capdir, f"rank{rank}_step*.pt"))
    steps = sorted(
        int(os.path.basename(f).split("_step", 1)[1].split(".pt", 1)[0]) for f in files
    )
    return steps


def _sha_selection(capdir, rank, step):
    """SHA256 of the (indices, lengths) tensors for one (rank, step) capture file.

    Byte-exact: identical selected indices + valid_lengths across ranks collide.
    Returns (sha, n_layers, bs, valid_lengths_minmax).
    """
    f = os.path.join(capdir, f"rank{rank}_step{step:05d}.pt")
    if not os.path.isfile(f):
        return None, 0, 0, None
    rec = torch.load(f, weights_only=False)
    idx = rec["indices"]  # [L, bs, max_top_k]
    lens = rec["lengths"]  # [L, bs]
    h = hashlib.sha256()
    h.update(idx.to(torch.int64).contiguous().cpu().numpy().tobytes())
    h.update(lens.to(torch.int64).contiguous().cpu().numpy().tobytes())
    vmin = int(lens.min().item())
    vmax = int(lens.max().item())
    return h.hexdigest(), int(idx.shape[0]), int(idx.shape[1]), [vmin, vmax]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path-label", required=True, help="radix_on | radix_off")
    ap.add_argument("--prefix-tokens", type=int, default=2400)
    ap.add_argument("--expect-ranks", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base_sentence = (
        "The quick brown fox jumps over the lazy dog near the riverbank at dawn. "
    )
    n_sent = max(1, args.prefix_tokens // 12)
    long_prefix = "".join(f"[{i}] {base_sentence}" for i in range(n_sent))
    prompt = long_prefix + " Summarize the passage above in one sentence."

    result = {
        "probe": "P2_cross_rank_selection_identity",
        "path_label": args.path_label,
        "capdir": CAPDIR,
        "prompt_chars": len(prompt),
        "expect_ranks": args.expect_ranks,
    }

    if os.path.isdir(CAPDIR):
        import shutil

        shutil.rmtree(CAPDIR)
    os.makedirs(CAPDIR, exist_ok=True)

    resp = _gen(prompt)
    meta = resp.get("meta_info", {})
    result["prompt_tokens"] = meta.get("prompt_tokens")
    result["cached_tokens"] = meta.get("cached_tokens", 0)

    ranks = _ranks_present(CAPDIR)
    result["ranks_present"] = ranks
    expect = list(range(args.expect_ranks))
    if ranks != expect:
        result["status"] = "FAIL"
        result["reason"] = f"not all {args.expect_ranks} ranks present: {ranks}"
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=2)
        print(json.dumps(result, indent=2))
        return 1

    # Common steps across all ranks (every rank must capture the same step set).
    step_sets = {r: set(_steps_for_rank(CAPDIR, r)) for r in expect}
    common_steps = sorted(set.intersection(*step_sets.values())) if step_sets else []
    result["per_rank_step_counts"] = {str(r): len(step_sets[r]) for r in expect}
    result["common_steps"] = common_steps
    if not common_steps:
        result["status"] = "FAIL"
        result["reason"] = "no common decode step captured across all ranks"
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=2)
        print(json.dumps(result, indent=2))
        return 1

    all_identical = True
    per_step = {}
    layers = bs = None
    for s in common_steps:
        ref_sha, L, B, ref_vl = _sha_selection(CAPDIR, 0, s)
        if layers is None:
            layers, bs = L, B
        rank_shas = {}
        step_ok = True
        for r in expect:
            sha, _, _, vl = _sha_selection(CAPDIR, r, s)
            rank_shas[str(r)] = sha[:16] if sha else None
            if sha != ref_sha:
                step_ok = False
                all_identical = False
        per_step[str(s)] = {
            "rank0_sha16": ref_sha[:16] if ref_sha else None,
            "all_ranks_match_rank0": step_ok,
            "n_layers": L,
            "bs": B,
            "valid_len_range_rank0": ref_vl,
            "rank_sha16": rank_shas,
        }
    result["layers"] = layers
    result["bs"] = bs
    result["per_step"] = per_step
    result["all_ranks_identical_all_steps"] = all_identical
    result["status"] = "PASS" if all_identical else "FAIL"
    if not all_identical:
        bad = [s for s, v in per_step.items() if not v["all_ranks_match_rank0"]]
        result["reason"] = f"cross-rank selection diverged at steps {bad}"

    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))
    return 0 if all_identical else 1


if __name__ == "__main__":
    sys.exit(main())
