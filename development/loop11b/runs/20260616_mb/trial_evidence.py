#!/usr/bin/env python3
"""AC-5 fail-closed per-trial evidence gate for one DS SLO trial.

Reads a bench_serving JSONL (the ``result_for_dump`` record emitted with
``--output-details``) for a single Double Sparsity SLO trial, prints + writes a
JSON evidence summary, and REFUSES (exits nonzero) on missing or contradictory
DS NO-OP evidence.

The comparator (``development/benchmark_compare.py``) only OBSERVES these fields.
Publication must refuse externally — that refusal lives here.

Refuse (exit 2) when ANY of:
  * ``dense_fallback_total`` / ``selected_tokens_mean`` / ``total_tokens_mean``
    is absent (key missing or null) from the trial record, OR
  * ``dense_fallback_total != 0`` (any dense fallback = DS did not stay a NO-OP), OR
  * NOT (``selected_tokens_mean`` < ``total_tokens_mean``) (sparsity must select
    strictly fewer tokens than the total; equality/inversion = no sparsity), OR
  * the reported aggregate disagrees with the per-request ``selected_tokens`` /
    ``total_tokens`` arrays, or any per-request row violates the DS metric contract
    ``sparsity_rate == 1 - selected/total`` (catches a mislabeled total aggregate).

Prefix-reuse distribution is computed from the per-request ``cached_tokens`` and
``input_lens`` arrays (cached_fraction = cached_tokens / prompt_tokens, per
request). A missing/empty cached_tokens array yields a null distribution but is
NOT by itself a refusal — the three DS aggregates above are the hard gate.

Usage:
    python3 trial_evidence.py <trial.jsonl> [--out <summary.json>]

Reads the LAST JSON record in the file (bench_serving appends; one trial = one
``--output-file`` so the last line is the trial of interest).
"""
from __future__ import annotations

import argparse
import json
import sys


def _percentiles(xs):
    """min/p25/p50/p75/p99/max/mean over a non-empty sorted-able list."""
    xs = sorted(float(x) for x in xs)
    n = len(xs)

    def pct(p):
        if n == 1:
            return xs[0]
        # linear interpolation on the inclusive [0, n-1] index range
        idx = p / 100.0 * (n - 1)
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        frac = idx - lo
        return xs[lo] + (xs[hi] - xs[lo]) * frac

    return {
        "min": xs[0],
        "p25": pct(25),
        "p50": pct(50),
        "p75": pct(75),
        "p99": pct(99),
        "max": xs[-1],
        "mean": sum(xs) / n,
        "n": n,
    }


def _cached_fraction_distribution(record):
    """cached_fraction = cached_tokens / prompt_tokens, per request.

    Requires the ``--output-details`` arrays ``cached_tokens`` and ``input_lens``.
    Returns None when either array is absent/empty or no request had a non-None
    cached_tokens with a positive prompt length.
    """
    cached = record.get("cached_tokens")
    prompts = record.get("input_lens")
    if not cached or not prompts or len(cached) != len(prompts):
        return None
    fracs = [
        c / p
        for c, p in zip(cached, prompts)
        if c is not None and p is not None and p > 0
    ]
    if not fracs:
        return None
    return _percentiles(fracs)


def _consistency_refusals(record, selected_mean, total_mean, rel_tol=0.01):
    """Refuse when the reported aggregates disagree with the per-request arrays,
    the per-request DS contract ``sparsity_rate == 1 - selected/total`` is violated,
    or DS instrumentation is PARTIAL. Fails CLOSED: the per-request arrays
    ``selected_tokens`` / ``total_tokens`` / ``sparsity_rate`` must each be present,
    equal-length (one entry per request), and have NO ``None`` rows — otherwise a
    subset of instrumented requests could silently satisfy the aggregate means. The
    contract check then runs over the row-aligned full arrays (no independent
    filtering that would misalign rows after a missing entry).
    """
    out = []
    sel = record.get("selected_tokens")
    tot = record.get("total_tokens")
    sr = record.get("sparsity_rate")
    for name, arr in (
        ("selected_tokens", sel),
        ("total_tokens", tot),
        ("sparsity_rate", sr),
    ):
        if not arr:
            out.append(f"per-request {name} array absent (cannot verify aggregate)")
    if out:
        return out
    n = len(sel)
    if not (len(tot) == n and len(sr) == n):
        out.append(
            f"per-request DS arrays length mismatch "
            f"(selected={len(sel)} total={len(tot)} sparsity={len(sr)})"
        )
        return out
    n_missing = sum(
        1 for i in range(n) if sel[i] is None or tot[i] is None or sr[i] is None
    )
    if n_missing:
        out.append(
            f"{n_missing} of {n} requests missing DS metadata "
            "(partial instrumentation; fail closed)"
        )
        return out
    if abs(sum(sel) / n - selected_mean) > rel_tol * max(1.0, selected_mean):
        out.append(
            f"selected_tokens_mean {selected_mean} disagrees with per-request "
            f"mean {sum(sel)/n:.3f}"
        )
    if abs(sum(tot) / n - total_mean) > rel_tol * max(1.0, total_mean):
        out.append(
            f"total_tokens_mean {total_mean} disagrees with per-request "
            f"mean {sum(tot)/n:.3f}"
        )
    for i in range(n):
        s, t, r = sel[i], tot[i], sr[i]
        if t <= 0:
            continue
        if abs((1.0 - s / t) - r) > 1e-3:
            out.append(
                f"req{i}: sparsity_rate {r} != 1-selected/total "
                f"{1.0 - s/t:.4f} (DS metric contract violated)"
            )
            break
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("jsonl", help="bench_serving result JSONL for one DS trial")
    ap.add_argument(
        "--out",
        default=None,
        help="path to write the JSON summary (default: <jsonl>.evidence.json)",
    )
    args = ap.parse_args(argv)

    # Read the last JSON record in the file (one trial appends one line).
    record = None
    with open(args.jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
    if record is None:
        print(f"REFUSE: {args.jsonl} contains no JSON record", file=sys.stderr)
        return 2

    dense_fallback_total = record.get("dense_fallback_total")
    selected_tokens_mean = record.get("selected_tokens_mean")
    total_tokens_mean = record.get("total_tokens_mean")

    refusals = []
    if dense_fallback_total is None:
        refusals.append("dense_fallback_total absent (key missing or null)")
    if selected_tokens_mean is None:
        refusals.append("selected_tokens_mean absent (key missing or null)")
    if total_tokens_mean is None:
        refusals.append("total_tokens_mean absent (key missing or null)")

    if not refusals:
        if dense_fallback_total != 0:
            refusals.append(
                f"dense_fallback_total={dense_fallback_total} != 0 "
                "(DS fell back to dense; not a NO-OP)"
            )
        if not (selected_tokens_mean < total_tokens_mean):
            refusals.append(
                f"NOT (selected_tokens_mean={selected_tokens_mean} < "
                f"total_tokens_mean={total_tokens_mean}); sparsity selected "
                "no fewer tokens than the total"
            )
        refusals.extend(
            _consistency_refusals(
                record, selected_tokens_mean, total_tokens_mean
            )
        )

    summary = {
        "source": args.jsonl,
        "dense_fallback_total": dense_fallback_total,
        "selected_tokens_mean": selected_tokens_mean,
        "total_tokens_mean": total_tokens_mean,
        "prefix_reuse_cached_fraction": _cached_fraction_distribution(record),
        "verdict": "REFUSE" if refusals else "PASS",
        "refusals": refusals,
    }

    out_path = args.out or (args.jsonl + ".evidence.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"wrote {out_path}", file=sys.stderr)

    if refusals:
        print("REFUSE: " + "; ".join(refusals), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
