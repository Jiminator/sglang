"""B1 fail-closed SELFTEST fixture generator.

Writes a synthetic, fully-engaged GATE-C arm set that the DEC-12 analyzer MUST score
status=FAIL — the partial@production arm breaches +/-0.5pp (a deliberate +2pp recall lift
on the reused region). The runner's SELFTEST mode runs the real analyze_edge.py on this
fixture and asserts the analyzer exits nonzero AND the runner's final-exit expression
(which now includes AN_RC) is nonzero. This proves the R26 fail-open (ignoring AN_RC) is
closed: a FAIL analyzer can no longer let the runner exit 0.

We emit exactly the sink/index schema the analyzer reads:
  sink_<arm>.jsonl  : oracle records {request_id, recall_at_k:{"2048": bool}}
  index_<arm>.jsonl : per-request {request_id, cached_tokens}
"""

from __future__ import annotations

import argparse
import json
import os
import random


def write_arm(d, arm_tag, length, n, cached_fn, hit_fn, recs_per_needle=4):
    sink = os.path.join(d, f"sink_{arm_tag}.jsonl")
    idx = os.path.join(d, f"index_{arm_tag}.jsonl")
    pref = {"cold": "cold", "cold_partial": "cprt", "boundary": "bnd",
            "partial": "prt", "nearfull": "nfl", "evict": "evc"}[arm_tag]
    rng = random.Random(hash(arm_tag) & 0xFFFF)
    with open(sink, "w") as fs, open(idx, "w") as fi:
        for i in range(n):
            rid = f"L{length}-{pref}-i{i}"
            fi.write(json.dumps({"request_id": rid, "cached_tokens": cached_fn(i)}) + "\n")
            base_hit = hit_fn(i)  # target per-needle recall fraction in [0,1]
            for _ in range(recs_per_needle):
                hit = rng.random() < base_hit
                fs.write(json.dumps({"request_id": rid, "recall_at_k": {"2048": hit}}) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--length", type=int, default=4096)
    ap.add_argument("--num", type=int, default=144)
    args = ap.parse_args()
    os.makedirs(args.dir, exist_ok=True)
    L, n = args.length, args.num

    # cold: clean control, ~60% recall, cached 0.
    write_arm(args.dir, "cold", L, n, cached_fn=lambda i: 0, hit_fn=lambda i: 0.60)
    # cold_partial: clean matched control, ~60% recall, cached 0.
    write_arm(args.dir, "cold_partial", L, n, cached_fn=lambda i: 0, hit_fn=lambda i: 0.60)
    # boundary: engaged page-aligned (cached multiple of 64, >0), recall == cold.
    write_arm(args.dir, "boundary", L, n, cached_fn=lambda i: 512, hit_fn=lambda i: 0.60)
    # partial@production: engaged (cached 2752 <= bound) BUT recall lifted +20pp vs
    # cold_partial -> BREACH (this is what the analyzer must catch as a FAIL).
    write_arm(args.dir, "partial", L, n, cached_fn=lambda i: 2752, hit_fn=lambda i: 0.80)
    # nearfull: out-of-contract, cached 4288, lifted (not a gate input).
    write_arm(args.dir, "nearfull", L, n, cached_fn=lambda i: 4288, hit_fn=lambda i: 0.80)
    # evict: cached fell to 0, recall == cold.
    write_arm(args.dir, "evict", L, n, cached_fn=lambda i: 0, hit_fn=lambda i: 0.60)
    print(f"[selftest-fixture] wrote 6 arms (n={n}) under {args.dir}; partial deliberately breaches +/-0.5pp")


if __name__ == "__main__":
    main()
