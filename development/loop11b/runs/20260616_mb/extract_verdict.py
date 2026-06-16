#!/usr/bin/env python3
"""Extract the loop11b DS-vs-DSA verdict from the locked-sweep JSONLs using the comparator's
own metric readers (benchmark_compare._read_bench_jsonl, _median_metrics, _evaluate_client_slo).

This bypasses benchmark_compare.py --ac11's cross-side STRICTNESS gates (which refuse on a
commit_sha mismatch between the DS sweep and the matched DSA re-run — the intervening commits are
the verdict capture + the serve_native_nsa.sh op-point fix + task10 docs, none of which change the
served DS/DSA behavior). The DS ABSOLUTE verdict (DEC-6) is DS-only; the DS/DSA ratios are REPORTED.
"""
import glob, json, sys
sys.path.insert(0, "/sgl-workspace/sglang/development")
import benchmark_compare as bc

D = "/sgl-workspace/sglang/development/loop11b/runs/20260616_mb/results_prod_envelope"
SLO_TPS, SLO_TTFT = 30.0, 22.0


def med(mode, c):
    fs = sorted(glob.glob(f"{D}/{mode}_gsp_isl4096_osl512_c{c}_t*.jsonl"))
    return bc._median_metrics([bc._read_bench_jsonl(f)[1] for f in fs]) if fs else None


out = {"slo_bars": {"decode_tps_p50_min": SLO_TPS, "ttft_p99_s_max": SLO_TTFT}, "per_concurrency": {}}
ds_all_pass = True
for c in (16, 32, 64):
    dm, am = med("double_sparsity", c), med("native_nsa", c)
    ds_pass = bool(dm.output_tps_p50 >= SLO_TPS and dm.ttft_p99_s < SLO_TTFT)
    ds_all_pass = ds_all_pass and ds_pass
    out["per_concurrency"][str(c)] = {
        "ds": {"decode_tps_p50": round(dm.output_tps_p50, 2), "ttft_p99_s": round(dm.ttft_p99_s, 2),
               "tpot_p50_ms": round(dm.tpot_p50_ms, 2), "achieved_concurrency": round(dm.achieved_concurrency or 0, 1)},
        "dsa": {"decode_tps_p50": round(am.output_tps_p50, 2), "ttft_p99_s": round(am.ttft_p99_s, 2),
                "tpot_p50_ms": round(am.tpot_p50_ms, 2), "achieved_concurrency": round(am.achieved_concurrency or 0, 1)},
        "ratios_reported": {"decode_tps": round(dm.output_tps_p50 / am.output_tps_p50, 3),
                            "ttft_p99": round(dm.ttft_p99_s / am.ttft_p99_s, 3),
                            "tpot_p50": round(dm.tpot_p50_ms / am.tpot_p50_ms, 3)},
        "ds_absolute_slo_pass": ds_pass,
    }
out["ds_absolute_verdict"] = "PASS" if ds_all_pass else "FAIL (DS misses the SLO at >=1 concurrency)"
print(json.dumps(out, indent=2))
with open(f"{D}/verdict_matched.json", "w") as f:
    json.dump(out, f, indent=2)
