#!/usr/bin/env python3
"""Parse one bench_serving JSONL result + the matching server startup log into
a sweep-table row and a human summary.

Official per-user-speed metric (owner re-baseline): client TPS =
total_output_tokens / (total_latency - TTFT) = Sigma tokens / Sigma decode_time
(>= 30). median_itl_ms / 1000-over-ITL is retained only as a speculation-inflated
cross-check (it can look optimistic under speculative token bursts).

Loop 2 selection semantics (owner decision): the SELECTION metric is client_TPS
(>= 30 target). p99_ttft_ms < 22000 is REPORT-ONLY (recorded per candidate, shown
as an informational sub-22s flag) and does NOT disqualify a candidate.
"""
import argparse
import json
import os
import re

CONCURRENCY = 64
# OFFICIAL metric (owner re-baseline, round 2): client TPS = total_output_tokens /
# (total_latency - TTFT) = Sigma tokens / Sigma decode_time per run; target >= 30 TPS.
# median_itl_ms is retained only as a (speculation-inflated) cross-check.
TPS_TARGET = 30.0
TTFT_TARGET_MS = 22000.0
ITL_TARGET_MS = 33.3  # cross-check only (client's retracted 1000/ITL form)


def client_tps_from_record(rec):
    """Client ground-truth TPS for one run: Sigma output_tokens / Sigma decode_time,
    where decode_time per request = latency - ttft = sum(itls). Needs --output-details."""
    itls = rec.get("itls")
    if not itls:
        return float("nan")
    olens = rec.get("output_lens")
    tot_tok = 0.0
    tot_dt = 0.0
    for i, row in enumerate(itls):
        if not row:
            continue
        tot_dt += sum(row)
        tot_tok += (olens[i] if olens else len(row) + 1)
    return tot_tok / tot_dt if tot_dt > 0 else float("nan")

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
TABLE = os.path.join(os.path.dirname(__file__), "sweep_table.md")
FACTS_DIR = os.path.join(os.path.dirname(__file__), "logs")

HEADER = (
    "| tag | changed knob(s) | **client_TPS** (select) | tps>=30 | "
    "median_itl_ms (xcheck) | mean_tpot_ms | thr/req | p99_ttft_ms | ttft<22s (info) | "
    "p99_itl_ms | accept_len | conc | max_conc | "
    "completed | errors | max_total_num_tokens | rationale |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
)


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def scrape_server_facts(serve_log):
    """Pull resolved capacity/backend facts out of the server startup log."""
    facts = {
        "max_total_num_tokens": None,
        "kv_cache_tokens": None,
        "attn_backend": None,
        "page_size": None,
        "kv_dtype": None,
        "lines": [],
    }
    if not serve_log or not os.path.exists(serve_log):
        return facts
    with open(serve_log, errors="replace") as f:
        text = f.read()

    m = re.search(r"max_total_num_tokens\D{0,3}(\d[\d,]*)", text)
    if m:
        facts["max_total_num_tokens"] = int(m.group(1).replace(",", ""))
    # log line: "KV Cache is allocated. dtype: torch.bfloat16, #tokens: 300352, KV size: ..."
    m = re.search(r"KV Cache is allocated\..*?#tokens:\s*(\d[\d,]*)", text)
    if m:
        facts["kv_cache_tokens"] = int(m.group(1).replace(",", ""))
    for pat, key in [
        (r"[Aa]ttention backend[^\n]*", "attn_backend"),
        (r"(?:--)?page[_ ]size[^\n]*", "page_size"),
        (r"kv[_ ]cache[_ ]dtype[^\n]*", "kv_dtype"),
    ]:
        mm = re.search(pat, text)
        if mm:
            facts[key] = mm.group(0).strip()[:160]
    # keep a compact set of capacity/backend lines for the record
    for line in text.splitlines():
        if re.search(
            r"max_total_num_tokens|KV Cache is allocated|attention backend|"
            r"Attention backend|page_size|kv_cache_dtype|Capture cuda graph|"
            r"fired up|Memory pool|chunked_prefill",
            line,
        ):
            facts["lines"].append(line.strip())
    return facts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--serve-log", default="")
    ap.add_argument("--extra-args", default="")
    ap.add_argument("--rationale", default="")
    args = ap.parse_args()

    jsonl = os.path.join(
        RESULTS_DIR, f"{args.tag}_isl4096_osl512_c{CONCURRENCY}.jsonl"
    )
    if not os.path.exists(jsonl):
        print(f"NO_RESULT_FILE {jsonl}")
        return 1
    with open(jsonl) as f:
        lines = [ln for ln in f if ln.strip()]
    rec = json.loads(lines[-1])

    median_itl = fnum(rec.get("median_itl_ms"))
    p99_ttft = fnum(rec.get("p99_ttft_ms"))
    mean_tpot = fnum(rec.get("mean_tpot_ms"))
    out_thr = fnum(rec.get("output_throughput"))
    completed = rec.get("completed")
    max_conc = rec.get("max_concurrent_requests")
    conc = fnum(rec.get("concurrency"))
    accept = rec.get("accept_length")
    p99_itl = fnum(rec.get("p99_itl_ms"))
    errs = rec.get("errors", [])
    err_count = sum(1 for e in errs if e) if isinstance(errs, list) else "n/a"

    client_tps = client_tps_from_record(rec)          # OFFICIAL metric (owner re-baseline)
    thr_per_req = out_thr / CONCURRENCY if out_thr == out_thr else float("nan")
    facts = scrape_server_facts(args.serve_log)

    # Loop 2 SELECTION metric is client TPS (>= 30). p99 TTFT is REPORT-ONLY
    # (informational sub-22s flag), not a disqualifier (owner decision DEC-2).
    tps_meets = client_tps >= TPS_TARGET
    ttft_sub22 = p99_ttft < TTFT_TARGET_MS  # informational only
    target_met = tps_meets  # selection on client TPS alone
    valid = (completed == 320) and (err_count == 0)

    accept_s = f"{accept:.3f}" if isinstance(accept, (int, float)) else str(accept)
    row = (
        f"| {args.tag} | {args.extra_args or '(base)'} | {client_tps:.2f} | {tps_meets} | "
        f"{median_itl:.2f} | {mean_tpot:.2f} | {thr_per_req:.2f} | "
        f"{p99_ttft:.1f} | {ttft_sub22} | {p99_itl:.2f} | {accept_s} | {conc:.1f} | "
        f"{max_conc} | {completed} | {err_count} | "
        f"{facts['max_total_num_tokens']} | {args.rationale} |\n"
    )
    if not os.path.exists(TABLE):
        with open(TABLE, "w") as f:
            f.write("# GLM-5.1-FP8 flags-only hill-climb — sweep table\n\n")
            f.write(HEADER)
    with open(TABLE, "a") as f:
        f.write(row)

    # per-candidate facts file for reproducibility
    with open(os.path.join(FACTS_DIR, f"facts_{args.tag}.txt"), "w") as f:
        f.write(f"tag={args.tag}\nextra_args={args.extra_args}\n")
        for k in ("max_total_num_tokens", "kv_cache_tokens", "attn_backend",
                  "page_size", "kv_dtype"):
            f.write(f"{k}={facts[k]}\n")
        f.write("\n--- server-log facts ---\n")
        f.write("\n".join(facts["lines"]))

    print("=" * 64)
    print(f"CANDIDATE {args.tag}   valid={valid}  tps_meets_30={tps_meets}  (ttft_sub22={ttft_sub22}, info)")
    print(f"  client_TPS    = {client_tps:.2f}  (OFFICIAL; target >= {TPS_TARGET}; =Sigma tok/Sigma decode)")
    print(f"  median_itl_ms = {median_itl:.2f}  (cross-check only; 1000/itl={1000.0/median_itl:.2f} — spec-inflated)")
    print(f"  mean_tpot_ms  = {mean_tpot:.2f}  (1000/mean_tpot={1000.0/mean_tpot:.2f}; thr/req={thr_per_req:.2f})")
    print(f"  p99_ttft_ms   = {p99_ttft:.1f}  (REPORT-ONLY; sub-22s={ttft_sub22}; not a disqualifier)")
    print(f"  p99_itl_ms    = {p99_itl:.2f}   accept_length={accept_s}")
    print(f"  completed={completed}  errors={err_count}  conc={conc:.1f}  max_conc={max_conc}")
    print(f"  max_total_num_tokens={facts['max_total_num_tokens']}  kv_cache_tokens={facts['kv_cache_tokens']}")
    print(f"  needed >= {CONCURRENCY}*4608 = {CONCURRENCY*4608} tokens")
    print("=" * 64)
    if not valid:
        print(f"INVALID_RESULT completed={completed} errors={err_count} (expected 320 / 0)")
        return 6
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
