"""R27 GATE C edge analyzer — DEC-12 PRODUCTION-REPRESENTATIVE-REUSE contract.

Owner DEC-12 re-authors the edge gate around the production reuse regime. The 4096-ISL
workload is ~55% prefix hit; cached ~2752 (~63% of the partial-arm prompt) is the
production UPPER bound. The PASS/FAIL gate is evaluated ONLY on production-representative
reuse; near-full reuse (cached ~4288, ~98%) is recorded separately and is NOT a gate input.

Statistical unit = the NEEDLE. For each arm the per-needle recall@2048 is the fraction of
that needle's oracle records with recall_at_k["2048"] true. Each on-arm case is paired by
needle index to the cold/fresh control. Reports per case: paired delta = case - cold (pp),
mean/std/min/max, 95% bootstrap CI, AND the raw cached_tokens distribution (engagement proof).

DEC-12 contract -- status == "PASS" IFF (production-representative reuse, cached <= ~2752):
  * boundary (page-aligned small reuse): |mean paired delta vs COLD| <= 0.5pp, AND
    cached>0 AND every reuse a page multiple (engaged page-aligned reuse), AND
  * partial @ production (cached ~2752, ~63%): |mean paired delta vs COLD_PARTIAL| <= 0.5pp,
    AND cached>0 (aligned pages reused + partial page recomputed). The partial arm is paired
    against a MATCHED radix-OFF control issuing the IDENTICAL prefix+tail prompt (~2763 tok),
    so the radix effect is isolated from the prompt-LENGTH confound (a shorter prompt recalls
    better regardless of radix). Pairing partial against the full-L4096 cold would conflate
    the two and is NOT the contract control. AND
  * eviction -> recompute: cached fell to 0 (prefix actually evicted) AND recall == COLD
    (|mean delta| <= 0.5pp; deterministic recompute, no stale slot).
Each control MUST be clean (radix-OFF, all cached 0): cold (full L4096) for boundary+evict,
cold_partial (prefix+tail) for partial. Cache engagement is PROVEN from raw per-request
cached_tokens. If a production-reuse arm did NOT engage (or a control is not clean), the run
is INVALID for the gate (recorded as a problem -> FAIL).

NEAR-FULL reuse (cached ~4288, >= ~98%) is recorded in `out_of_contract_value_affecting`
(the R26 +1.57pp characterization). It is NOT a PASS/FAIL gate input and is NOT dropped.

`problems[]` lists any production-reuse case breaching +/-0.5pp (or any engagement/control
failure). status = "FAIL" iff problems is non-empty. FAIL-CLOSED: any missing/short arm,
unengaged production arm, dirty cold control, or breach -> FAIL. The writer reads this
top-level status/problems verbatim. main() returns 0 IFF status=="PASS".
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import random
from collections import defaultdict


def _open_jsonl(path):
    """Open a .jsonl path, transparently falling back to its .gz sibling.

    The committed evidence stores the large sink/index files compressed
    (``*.jsonl.gz``) so a clean checkout can reproduce the verdict directly; the
    uncompressed ``*.jsonl`` is gitignored. Returns an open text handle, or None
    if neither the plain nor the .gz file exists.
    """
    if os.path.exists(path):
        return open(path, "r", encoding="utf-8")
    if os.path.exists(path + ".gz"):
        return gzip.open(path + ".gz", "rt", encoding="utf-8")
    return None


def _recall_2048(payload) -> bool:
    rk = payload.get("recall_at_k", {})
    if "2048" in rk:
        return bool(rk["2048"])
    if 2048 in rk:
        return bool(rk[2048])
    return int(payload.get("needle_worst_rank", 1 << 30)) < 2048


def _idx_of(request_id):
    try:
        return int(str(request_id).rsplit("-i", 1)[1])
    except (ValueError, IndexError):
        return None


def per_needle_recall(sink_path, prefix):
    by_idx = defaultdict(lambda: [0, 0])
    failures = defaultdict(int)
    fh = _open_jsonl(sink_path)
    if fh is None:
        return {}, {"missing_sink": 1}
    with fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if "failure" in rec:
                failures[str(rec["failure"]).split(":")[0]] += 1
                continue
            rid = rec.get("request_id", "")
            if not str(rid).startswith(prefix):
                continue
            i = _idx_of(rid)
            if i is None:
                continue
            by_idx[i][0] += 1
            if _recall_2048(rec):
                by_idx[i][1] += 1
    out = {}
    for i, (n, hits) in by_idx.items():
        out[i] = {"records": n, "hits": hits, "pct": (100.0 * hits / n) if n else None}
    return out, dict(failures)


def load_cached(index_path):
    out = {}
    fh = _open_jsonl(index_path)
    if fh is None:
        return out
    with fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            i = _idx_of(r.get("request_id", ""))
            if i is not None:
                out[i] = r.get("cached_tokens")
    return out


def dist(vals):
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return {}
    n = len(vals)
    mean = sum(vals) / n
    return {
        "n": n, "min": vals[0], "max": vals[-1], "mean": round(mean, 2),
        "median": vals[n // 2],
        "n_zero": sum(1 for v in vals if v == 0),
        "n_gt0": sum(1 for v in vals if v > 0),
        "n_page_multiple_gt0": sum(1 for v in vals if v > 0 and v % 64 == 0),
    }


def bootstrap_ci(deltas, iters=20000, seed=12345):
    if not deltas:
        return None, None
    rng = random.Random(seed)
    n = len(deltas)
    means = []
    for _ in range(iters):
        s = 0.0
        for _ in range(n):
            s += deltas[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    return means[int(0.025 * iters)], means[int(0.975 * iters)]


def paired(cold, case, cached_case):
    """Per-needle paired delta = case - cold over needles present in BOTH."""
    idxs = sorted(set(cold) & set(case))
    rows, deltas = [], []
    for i in idxs:
        c, k = cold[i], case[i]
        d = None
        if c["pct"] is not None and k["pct"] is not None:
            d = k["pct"] - c["pct"]
            deltas.append(d)
        rows.append({
            "idx": i,
            "cold_pct": round(c["pct"], 4) if c["pct"] is not None else None,
            "case_pct": round(k["pct"], 4) if k["pct"] is not None else None,
            "delta_case_minus_cold_pp": round(d, 4) if d is not None else None,
            "cached_case": cached_case.get(i),
        })
    n = len(deltas)
    mean = sum(deltas) / n if n else None
    if n > 1:
        var = sum((d - mean) ** 2 for d in deltas) / (n - 1)
        std = var ** 0.5
    else:
        std = 0.0 if n == 1 else None
    lo, hi = bootstrap_ci(deltas)
    mean_case = round(sum(case[i]["pct"] for i in idxs) / n, 4) if n else None
    mean_cold = round(sum(cold[i]["pct"] for i in idxs) / n, 4) if n else None
    return {
        "n_needles_matched": n,
        "recall_cold_pct_mean": mean_cold,
        "recall_case_pct_mean": mean_case,
        "paired_delta_case_minus_cold_pp": {
            "mean": round(mean, 4) if mean is not None else None,
            "std": round(std, 4) if std is not None else None,
            "min": round(min(deltas), 4) if deltas else None,
            "max": round(max(deltas), 4) if deltas else None,
            "ci95_bootstrap": [round(lo, 4), round(hi, 4)] if lo is not None else None,
            "ci_iters": 20000,
        },
    }, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True, help="dir holding sink_<arm>.jsonl + index_<arm>.jsonl")
    ap.add_argument("--length", type=int, default=4096)
    ap.add_argument("--page-size", type=int, default=64)
    ap.add_argument("--max-delta-pp", type=float, default=0.5)
    ap.add_argument("--min-needles", type=int, default=128)
    ap.add_argument(
        "--production-cached-max", type=int, default=2752,
        help="production reuse upper bound; partial arm cached must be <= this (DEC-12)",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--table-out", required=True)
    args = ap.parse_args()

    L = args.length
    D = args.dir
    MD = args.max_delta_pp
    arms = {
        "cold": f"L{L}-cold-i",
        "cold_partial": f"L{L}-cprt-i",
        "boundary": f"L{L}-bnd-i",
        "partial": f"L{L}-prt-i",
        "nearfull": f"L{L}-nfl-i",
        "evict": f"L{L}-evc-i",
    }
    recall, fails, cached = {}, {}, {}
    for a, pref in arms.items():
        recall[a], fails[a] = per_needle_recall(os.path.join(D, f"sink_{a}.jsonl"), pref)
        cached[a] = load_cached(os.path.join(D, f"index_{a}.jsonl"))

    problems = []  # gate-affecting problems ONLY (production reuse)

    cold_dist = dist(list(cached["cold"].values()))
    cprt_dist = dist(list(cached["cold_partial"].values()))
    bnd_dist = dist(list(cached["boundary"].values()))
    prt_dist = dist(list(cached["partial"].values()))
    nfl_dist = dist(list(cached["nearfull"].values()))
    evc_dist = dist(list(cached["evict"].values()))

    # cold control MUST be clean (radix-OFF, all 0).
    if not cold_dist:
        problems.append("cold control: NO data (missing/empty sink)")
    elif cold_dist.get("n_gt0", 0) != 0:
        problems.append(f"cold control not clean: cached_tokens>0 on {cold_dist['n_gt0']} reqs")

    # cold_partial control (matched prefix+tail) MUST be clean (radix-OFF, all 0).
    if not cprt_dist:
        problems.append("cold_partial control: NO data (missing/empty sink)")
    elif cprt_dist.get("n_gt0", 0) != 0:
        problems.append(f"cold_partial control not clean: cached_tokens>0 on {cprt_dist['n_gt0']} reqs")

    # boundary MUST engage page-aligned reuse (production-representative gate input).
    if bnd_dist.get("n_gt0", 0) == 0:
        problems.append("boundary: NO cache hit (cached_tokens all 0) — page-aligned reuse not exercised")
    elif bnd_dist.get("n_gt0", 0) != bnd_dist.get("n_page_multiple_gt0", 0):
        problems.append(
            f"boundary: {bnd_dist['n_gt0'] - bnd_dist.get('n_page_multiple_gt0', 0)} reuse hits "
            "not a page multiple (expected full page-aligned reuse)"
        )

    # partial @ production MUST engage AND stay at/below the production upper bound.
    if prt_dist.get("n_gt0", 0) == 0:
        problems.append("partial@production: NO cache hit (cached_tokens all 0) — partial-page reuse not exercised")
    elif prt_dist.get("max", 0) > args.production_cached_max:
        problems.append(
            f"partial@production: cached max {prt_dist['max']} EXCEEDS production upper bound "
            f"{args.production_cached_max} — not production-representative (use a smaller prefix warmup)"
        )

    # eviction MUST have fallen to 0 (prefix actually evicted, recompute).
    if evc_dist.get("n_gt0", 0) != 0:
        problems.append(
            f"eviction: cache did NOT fall — cached_tokens>0 on {evc_dist['n_gt0']} reqs after flush_cache"
        )

    # Hard-failure markers on gate arms.
    for a in ("cold", "cold_partial", "boundary", "partial", "evict"):
        hard = sum(fails[a].get(k, 0) for k in ("span_out_of_range", "exception", "missing_sink"))
        if hard:
            problems.append(f"{a}: {hard} hard-failure oracle marker(s) {dict(fails[a])}")

    # Paired recall. Each reuse arm is paired against its MATCHED radix-OFF control:
    #   boundary -> cold (full L4096, same prompt)
    #   partial  -> cold_partial (prefix+tail, same prompt — isolates radix from prompt len)
    #   evict    -> cold (full L4096, same prompt)
    #   nearfull -> cold (L4096+tail ~= full L4096 len; out-of-contract characterization)
    control_for = {
        "boundary": "cold",
        "partial": "cold_partial",
        "nearfull": "cold",
        "evict": "cold",
    }
    cases = {}
    tables = {}
    for a in ("boundary", "partial", "nearfull", "evict"):
        ctrl = control_for[a]
        stats, rows = paired(recall[ctrl], recall[a], cached[a])
        stats["control_arm"] = ctrl
        cases[a] = stats
        tables[a] = rows

    # Robust-n + within-bound on the THREE gate arms only.
    for a in ("boundary", "partial", "evict"):
        stats = cases[a]
        if stats["n_needles_matched"] < args.min_needles:
            problems.append(
                f"{a}: only {stats['n_needles_matched']} matched needles (< robust n {args.min_needles})"
            )
        md = stats["paired_delta_case_minus_cold_pp"]["mean"]
        if md is None:
            problems.append(f"{a}: no paired recall data (mean delta None)")
        elif abs(md) > MD:
            label = "recompute != cold (stale slot?)" if a == "evict" else "production-reuse breach (stale-slot corruption?)"
            problems.append(
                f"{a}: mean paired recall delta {md:+.4f}pp from {stats['control_arm']} (>+/-{MD}pp) — {label}"
            )

    within = {}
    for a in ("boundary", "partial", "evict"):
        md = cases[a]["paired_delta_case_minus_cold_pp"]["mean"]
        within[a] = md is not None and abs(md) <= MD
        cases[a]["within_bound_on_mean"] = within[a]
    cases["nearfull"]["within_bound_on_mean"] = None  # not a gate input

    # Out-of-contract near-full characterization (NOT a gate input, NOT dropped).
    nfl_stats = cases["nearfull"]
    nfl_md = nfl_stats["paired_delta_case_minus_cold_pp"]
    out_of_contract = {
        "case": "near_full_reuse_partial_page",
        "cached_dist": nfl_dist,
        "n_needles_matched": nfl_stats["n_needles_matched"],
        "recall_cold_pct_mean": nfl_stats["recall_cold_pct_mean"],
        "recall_case_pct_mean": nfl_stats["recall_case_pct_mean"],
        "paired_delta_case_minus_cold_pp": nfl_md,
        "note": (
            "NEAR-FULL reuse (cached ~4288, >=~98%) is OUT OF the DEC-12 production-reuse "
            "contract; recorded as value-affecting characterization only (the R26 +1.57pp "
            "effect). NOT a PASS/FAIL gate input."
        ),
        "value_affecting": (nfl_md["mean"] is not None and abs(nfl_md["mean"]) > MD),
    }

    verdict = {
        "probe": "GATE_C_edge_DEC12_production_reuse_page64",
        "contract": "DEC-12",
        "length": L,
        "page_size": args.page_size,
        "top_k": 2048,
        "max_delta_pp": MD,
        "min_needles": args.min_needles,
        "production_cached_max": args.production_cached_max,
        "rule": (
            "PASS iff (production-representative reuse, cached <= ~2752): boundary mean "
            "paired recall delta vs cold within +/-0.5pp (page-aligned engaged) AND partial "
            "@ production (cached ~2752) within +/-0.5pp (engaged) AND eviction recompute == "
            "cold (within +/-0.5pp, cached fell to 0). Near-full reuse (cached ~4288) is "
            "out-of-contract value-affecting, NOT a gate input. cold control clean. Cache "
            "engagement PROVEN per arm from raw cached_tokens. Statistical unit = needle; "
            "robust n >= 128 matched."
        ),
        "cache_engagement": {
            "cold_control_cached_dist": cold_dist,
            "cold_partial_control_cached_dist": cprt_dist,
            "boundary_cached_dist": bnd_dist,
            "partial_production_cached_dist": prt_dist,
            "nearfull_cached_dist": nfl_dist,
            "eviction_cached_dist": evc_dist,
            "cold_is_clean_control": bool(cold_dist) and cold_dist.get("n_gt0", 0) == 0,
            "cold_partial_is_clean_control": bool(cprt_dist) and cprt_dist.get("n_gt0", 0) == 0,
            "boundary_engaged_page_aligned": (
                bnd_dist.get("n_gt0", 0) > 0
                and bnd_dist.get("n_gt0", 0) == bnd_dist.get("n_page_multiple_gt0", 0)
            ),
            "partial_production_engaged": prt_dist.get("n_gt0", 0) > 0,
            "partial_within_production_bound": (
                prt_dist.get("n_gt0", 0) > 0
                and prt_dist.get("max", 1 << 30) <= args.production_cached_max
            ),
            "eviction_fell_to_zero": evc_dist.get("n_gt0", 0) == 0,
        },
        "cases": {
            "a_boundary_aligned_reuse": cases["boundary"],
            "b_partial_page_hit_production": cases["partial"],
            "c_eviction_recompute": cases["evict"],
        },
        "out_of_contract_value_affecting": out_of_contract,
        "failures": {a: dict(fails[a]) for a in arms},
        "status": "PASS" if not problems else "FAIL",
        "problems": problems,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(verdict, fh, indent=2)
    with open(args.table_out, "w") as fh:
        for a in ("boundary", "partial", "nearfull", "evict"):
            for r in tables[a]:
                r2 = dict(r)
                r2["case"] = a
                fh.write(json.dumps(r2) + "\n")

    print(f"[analyze_edge DEC-12] -> {args.out}")
    print(f"  cold   cached dist : {cold_dist}")
    print(f"  cprt   cached dist : {cprt_dist}  (matched control for partial)")
    print(f"  bnd    cached dist : {bnd_dist}")
    print(f"  prt@P  cached dist : {prt_dist}")
    print(f"  nfl    cached dist : {nfl_dist}  (out-of-contract)")
    print(f"  evc    cached dist : {evc_dist}")
    for a in ("boundary", "partial", "evict"):
        s = cases[a]
        d = s["paired_delta_case_minus_cold_pp"]
        print(f"  {a:9s}(vs {s['control_arm']:11s}): n={s['n_needles_matched']} "
              f"ctrl={s['recall_cold_pct_mean']}% case={s['recall_case_pct_mean']}% "
              f"mean_delta={d['mean']}pp CI95={d['ci95_bootstrap']} within={s['within_bound_on_mean']}")
    nd = nfl_stats["paired_delta_case_minus_cold_pp"]
    print(f"  nearfull (OUT-OF-CONTRACT): n={nfl_stats['n_needles_matched']} "
          f"mean_delta={nd['mean']}pp CI95={nd['ci95_bootstrap']} "
          f"value_affecting={out_of_contract['value_affecting']}")
    print(f"  STATUS: {verdict['status']}")
    for p in problems:
        print(f"    - {p}")
    return 0 if not problems else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
