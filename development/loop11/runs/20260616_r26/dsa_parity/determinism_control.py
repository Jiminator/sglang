"""Determinism control: on the SAME radix-OFF (fresh) server, issue each measured
prompt (P+PARTIAL_TAIL) TWICE and check whether the two greedy outputs are identical.

This establishes the NOISE FLOOR: if fresh-vs-fresh already diverges, then the
fresh-vs-reuse 74.3% divergence is just nondeterminism, not a reuse effect. If
fresh-vs-fresh is ~0% divergent (deterministic) but fresh-vs-reuse is high, the
reuse divergence is a REAL selection deviation under radix reuse.

Radix-OFF so neither call hits a cache; both recompute from scratch. Same server,
so weights/kernels/seed identical.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import requests

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = _HERE
while _REPO != "/" and not (
    os.path.isdir(os.path.join(_REPO, "test", "manual"))
    and os.path.isdir(os.path.join(_REPO, "python", "sglang"))
):
    _REPO = os.path.dirname(_REPO)
sys.path.insert(0, os.path.join(_REPO, "test", "manual"))
sys.path.insert(0, os.path.join(_REPO, "python"))

import test_double_sparsity_v32 as h  # noqa: E402

PARTIAL_TAIL = " Additionally, note the following minor clarifying remark here."


def _gen(base, prompt, n):
    r = requests.post(
        f"{base}/generate",
        json={"text": prompt, "sampling_params": {
            "max_new_tokens": n, "temperature": 0, "ignore_eos": True}},
        timeout=600,
    )
    r.raise_for_status()
    d = r.json()
    return (d.get("text", "") or "").strip()[:200], d["meta_info"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=4096)
    ap.add_argument("--num", type=int, default=48)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    base = os.environ.get("DS_BASE_URL", "http://127.0.0.1:30000")
    L = args.length
    diverge = 0
    n = 0
    with open(args.out, "w") as fh:
        for idx in range(args.num):
            needle = h._niah_needle(L, idx)
            P = h._make_niah_prompt(L, seed=args.seed_base + idx, needle=needle) + PARTIAL_TAIL
            t1, m1 = _gen(base, P, args.max_new_tokens)
            t2, m2 = _gen(base, P, args.max_new_tokens)
            same = (t1 == t2)
            if not same:
                diverge += 1
            n += 1
            fh.write(json.dumps({
                "idx": idx, "run1_cached": m1.get("cached_tokens"),
                "run2_cached": m2.get("cached_tokens"), "same": same,
                "t1": t1, "t2": t2,
            }) + "\n")
            fh.flush()
    print(f"[determinism] fresh-vs-fresh divergence: {diverge}/{n} = "
          f"{100.0*diverge/n:.2f}% (noise floor)")


if __name__ == "__main__":
    main()
