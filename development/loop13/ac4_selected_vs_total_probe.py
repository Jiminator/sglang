#!/usr/bin/env python3
"""AC-4 selected-vs-total probe — artifact-backed DS selected/total per arm per regime (fail-closed).

Replaces the static ds={...} literals in build_ledger.py with evidence read from the live server's
``meta_info["double_sparsity"]`` (the per-request DS summary the server already emits): for the running DS
arm, send a DENSE (< top_k) and a SPARSE (> top_k) /generate, read selected_tokens / total_tokens /
dense_fallback, and record evidence/ac4_selected_vs_total.json[arm][regime]. The DS-active invariants are
asserted per arm: dense ``selected == total`` (the selector keeps every token when seq <= top_k), sparse
``selected < total`` (genuine pruning), and ``dense_fallback == 0`` (DS not silently falling back) in BOTH
regimes. Run during each DS arm's uptime (production_ds, ref_faithful, ref_cosine).

Fail-closed: nonzero exit + NO write to that arm if meta_info["double_sparsity"] is absent or any invariant
fails. The JSON is updated arm-by-arm (load → set this arm → atomic .tmp→os.replace), so reruns accumulate.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
EVID = os.path.join(HERE, "evidence")
_FILLER = ("The quarterly logistics review records routine warehouse activity across the regional depots, "
           "with nothing unusual to report for this period. ")


def _gen(base, prompt):
    r = requests.post(f"{base}/generate",
                      json={"text": prompt, "sampling_params": {"max_new_tokens": 8, "temperature": 0}},
                      timeout=600)
    r.raise_for_status()
    return r.json().get("meta_info", {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--base-url", default=os.environ.get("DS_BASE_URL", "http://127.0.0.1:30000"))
    ap.add_argument("--dense-units", type=int, default=14)     # ~334 tokens  < top_k 2048
    ap.add_argument("--sparse-units", type=int, default=160)   # ~3700 tokens > top_k 2048 (80 units was ~1852, too small)
    ap.add_argument("--out", default=os.path.join(EVID, "ac4_selected_vs_total.json"))
    args = ap.parse_args()

    prompts = {"dense": _FILLER * args.dense_units + " Summarize.",
               "sparse": _FILLER * args.sparse_units + " Summarize."}
    arm_rec = {}
    problems = []
    for regime, prompt in prompts.items():
        meta = _gen(args.base_url, prompt)
        ds = meta.get("double_sparsity")
        if not ds:
            problems.append(f"{regime}: no meta_info['double_sparsity'] (DS off/inactive)")
            continue
        sel = int(ds.get("selected_tokens", -1))
        tot = int(ds.get("total_tokens", -1))
        fb = int(ds.get("dense_fallback", 1))
        arm_rec[regime] = {"selected_tokens": sel, "total_tokens": tot, "dense_fallback": fb,
                           "server_prompt_tokens": meta.get("prompt_tokens")}
        if fb != 0:
            problems.append(f"{regime}: dense_fallback={fb} != 0 (DS fell back)")
        if regime == "dense" and not (sel == tot and sel > 0):
            problems.append(f"{regime}: selected={sel} total={tot}, expected selected==total>0")
        if regime == "sparse" and not (0 < sel < tot):
            problems.append(f"{regime}: selected={sel} total={tot}, expected 0<selected<total")

    print(json.dumps({args.arm: arm_rec}, indent=2))
    if problems or set(arm_rec) != {"dense", "sparse"}:
        print(f"FAIL (fail-closed) [{args.arm}]: " + " | ".join(problems or ["missing a regime"]) +
              " — NOT recording this arm.", file=sys.stderr)
        raise SystemExit(2)

    data = {}
    if os.path.exists(args.out):
        try:
            data = json.load(open(args.out))
        except ValueError:
            data = {}
    data[args.arm] = arm_rec
    tmp = args.out + ".tmp"
    json.dump(data, open(tmp, "w"), indent=2)
    os.replace(tmp, args.out)
    print("wrote", args.out, "arm", args.arm)


if __name__ == "__main__":
    main()
