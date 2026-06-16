"""DSA-native radix-reuse parity probe — FALLBACK metric (end-to-end answer recall +
exact greedy-output divergence), client-only.

WHY FALLBACK (not the PRIMARY indexer-topk selection-recall):
  --enable-return-indexer-topk surfaces PHYSICAL kv-cache slot indices (page-table
  values from fast_topk_transform_fused), NOT logical token positions. Under radix
  reuse the logical->physical mapping is arbitrary (cached slots), so cross-arm
  (fresh vs reuse) LOGICAL needle selection-recall is NOT computable from the HTTP
  client without a new server-side debug field. The prompt forbids new production
  code. The row-2047 all-select reconstruction trick only covers needles whose
  tokens are <=2047, but 77/144 NIAH needles at L4096 exceed that. So the apples-to-
  apples PRIMARY metric is INFEASIBLE under the constraints. We use the documented
  FALLBACK and report saturation explicitly.

This driver runs ONE arm per invocation (selected by --arm), mirroring the DS R26
edge probe's HEAVY (partial-page, ~4288 cached) construction that produced DS's
+1.57pp deviation — so the parity comparison is apples-to-apples:
  fresh : radix-OFF server (--disable-radix-cache). cached_tokens MUST be 0. No warmup.
          MEASURED PROMPT = P+PARTIAL_TAIL (identical to reuse's measured prompt).
  reuse : radix-ON server (default). HEAVY reuse: warm the L4096 prompt P itself, then
          re-request P+PARTIAL_TAIL so the shared prefix diverges MID-PAGE — the radix
          cache reuses floor(shared/64) aligned pages (~4288 tokens) and recomputes the
          partial page. This is the EXACT DS R26 'partial' arm construction
          (edge_highn_driver.py PARTIAL_TAIL), the one that gave DS +1.57pp.

The MEASURED PROMPT is byte-identical across arms (P+PARTIAL_TAIL); the ONLY difference
is radix reuse (fresh recomputes everything; reuse hits ~4288 cached tokens). The needle
lives inside the shared prefix P (depth 0.35-0.65) so recall is measured on the reused
region — identical to how DS measured it. Because the prompt is identical, exact greedy
output (temp=0) is also a clean fresh-vs-reuse selection-divergence signal.

Greedy (temperature=0, ignore_eos) decoding: if DSA's selection were bit-identical
fresh vs reuse, the generated token sequence would be identical. Per-needle:
  * answer_hit  : needle string is a substring of the generated answer (recall).
  * output_text : the exact generated text (for token-level divergence vs fresh).
  * cached_tokens / prompt_tokens : engagement proof from raw server meta_info.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

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

# SAME partial-page tail as the DS R26 edge probe (edge_highn_driver.py PARTIAL_TAIL):
# appended so the shared prefix diverges mid-page, forcing reuse of floor(shared/64)
# aligned pages (~4288 cached) + recompute of the partial page.
PARTIAL_TAIL = " Additionally, note the following minor clarifying remark here."


def _generate(base, prompt, max_new_tokens):
    r = requests.post(
        f"{base}/generate",
        json={
            "text": prompt,
            "sampling_params": {
                "max_new_tokens": int(max_new_tokens),
                "temperature": 0,
                "ignore_eos": True,
            },
        },
        timeout=600,
    )
    r.raise_for_status()
    d = r.json()
    return d.get("text", ""), d["meta_info"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True, choices=["fresh", "reuse"])
    ap.add_argument("--length", type=int, default=4096)
    ap.add_argument("--num", type=int, default=144)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--warmup-length", type=int, default=1024)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = os.environ.get("DS_BASE_URL", "http://127.0.0.1:30000")
    L = args.length
    W = args.warmup_length
    arm = args.arm

    issued = 0
    t0 = time.time()
    with open(args.out, "w") as fh:
        for idx in range(args.num):
            needle = h._niah_needle(L, idx)
            P = h._make_niah_prompt(L, seed=args.seed_base + idx, needle=needle)

            if arm == "reuse":
                # HEAVY reuse (DS R26 'partial' arm): warm P itself, then re-request
                # P+PARTIAL_TAIL so the shared prefix diverges mid-page -> radix reuses
                # ~4288 aligned tokens (cached at the heavy level) + recomputes the
                # partial page. Needle is inside the shared prefix P.
                try:
                    _generate(base, P, 2)
                except Exception as e:  # warmup failure is non-fatal; logged
                    print(f"[dsa:{arm}] WARN warmup i{idx} failed: {e}")
            # MEASURED PROMPT identical across arms: only radix reuse differs.
            measured_prompt = P + PARTIAL_TAIL

            try:
                text, meta = _generate(base, measured_prompt, args.max_new_tokens)
            except Exception as e:
                print(f"[dsa:{arm}] WARN measured i{idx} failed: {e}")
                continue

            rec = {
                "arm": arm,
                "idx": idx,
                "request_id": f"L{L}-{arm}-i{idx}",
                "needle": needle,
                "answer_hit": (needle in (text or "")),
                "output_text": (text or "").strip()[:200],
                "prompt_tokens": meta.get("prompt_tokens"),
                "cached_tokens": meta.get("cached_tokens"),
                "completion_tokens": meta.get("completion_tokens"),
            }
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            issued += 1

    print(f"[dsa:{arm}] issued {issued}/{args.num} at L{L} "
          f"(warmup_length={W if arm=='reuse' else 0}) ({time.time()-t0:.1f}s)")
    print(f"[dsa:{arm}] per-request index -> {args.out}")


if __name__ == "__main__":
    main()
