"""R25 PROBE 5 — cold_warm_flips_value_neutral_documented.

This is the value-neutral CHARACTERIZATION, NOT a bit-identity gate. A radix cache
hit changes the decode query (v_h) UPSTREAM of DS selection — the cached fp8 latent
bytes are bit-identical, so the divergence lives in the model forward, shared with the
shipped dense/DSA paths. The cold/warm top-k therefore reshuffles a small fraction of
borderline near-cutoff tokens. We document this with two citations:

  1. The cold/warm selected-index symdiff magnitude (R24 P1 measured ~2% near-cutoff
     reshuffling: symdiff_pct_of_selected ~ 1.99%, cross-rank IDENTICAL on both passes),
     re-read from the committed R24 P1 verdict.
  2. The recall-neutrality of those flips: R25 Probe 1 (recall@2048 radix-OFF vs
     radix-ON) is within +/-0.5pp overall AND per length. The flips do not move recall.

`documented` is True iff BOTH citations are present and consistent (the R24 symdiff is
a small near-cutoff fraction, and the R25 recall-equivalence probe PASSED). This is a
documentation step gated on real evidence — it is NOT claiming bit-identity.
"""

from __future__ import annotations

import argparse
import json
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r24-p1-verdict", required=True)
    ap.add_argument("--r25-recall-verdict", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.r24_p1_verdict) as fh:
        p1 = json.load(fh)
    with open(args.r25_recall_verdict) as fh:
        recall = json.load(fh)

    # Citation 1: R24 P1 cold/warm selected-index symdiff (near-cutoff reshuffling).
    per_rank = p1.get("per_rank", {})
    sym_pcts = sorted(
        {round(float(v.get("symdiff_pct_of_selected")), 4)
         for v in per_rank.values() if v.get("symdiff_pct_of_selected") is not None}
    )
    cross_rank_identical = bool(p1.get("cross_rank_identical"))
    # The flips are a SMALL near-cutoff fraction (well under, say, 5% of selected).
    symdiff_small = bool(sym_pcts) and max(sym_pcts) < 5.0

    # Citation 2: R25 recall-equivalence (the flips are recall-neutral).
    recall_pass = recall.get("verdict") == "PASS"
    overall = recall.get("overall", {})

    documented = bool(symdiff_small and recall_pass)

    out = {
        "probe": "P5_cold_warm_flips_value_neutral_documented",
        "claim": (
            "Cold/warm selected-index flips are value-neutral near-cutoff reshuffling, "
            "NOT a bit-identity claim. A radix cache hit changes the decode query "
            "upstream of DS selection (cached fp8 latent bytes are bit-identical), so "
            "the top-k reshuffles a small fraction of borderline tokens; recall is "
            "unchanged."
        ),
        "citation_1_symdiff": {
            "source": args.r24_p1_verdict,
            "symdiff_pct_of_selected_per_rank": sym_pcts,
            "cross_rank_identical_both_passes": cross_rank_identical,
            "is_small_near_cutoff_fraction(<5pct)": symdiff_small,
            "r24_note": p1.get("reason"),
        },
        "citation_2_recall_neutral": {
            "source": args.r25_recall_verdict,
            "r25_recall_equivalence_verdict": recall.get("verdict"),
            "overall_delta_pp": overall.get("delta_pp"),
            "recall_equivalence_passed": recall_pass,
        },
        "documented": documented,
        "status": "PASS" if documented else "FAIL",
    }
    if not documented:
        out["reason"] = (
            f"symdiff_small={symdiff_small} (pcts={sym_pcts}) recall_pass={recall_pass}"
        )

    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))
    return 0 if documented else 1


if __name__ == "__main__":
    sys.exit(main())
