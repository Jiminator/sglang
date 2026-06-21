#!/usr/bin/env python3
"""AC-2.4 NIAH recall-oracle@2048 — GLM-5.1-FP8 driver + reducer (corroboration only).

Self-contained GLM Needle-In-A-Haystack driver for the DS selection-recall oracle. The server-side oracle
(selection_recall_oracle.py, gated by the config-borne ``recall_oracle`` flag) records, per
(request,trial,layer,decode_step), the needle's score rank among the live all-reduced DS selection scores
and ``recall_at_k`` (the all-needle-tokens-in-top-K rule). This driver registers each trial's needle span
via the cross-process oracle sink, issues a single-request ``/generate`` that forces a few DECODE forwards
(DS selection runs only in decode), then reduces the fail-closed sink into per-regime recall@2048.

This is CORROBORATION, not exoneration (plan AC-2.4 / DEC): recall@2048 is a property of the score ranking
on a NIAH task, not a generic selected-index equivalence proof. The DENSE regime (prompt < top_k) selects
every token, so recall@2048 is trivially 1.0 (it confirms dense selects all). The SPARSE regime
(prompt > top_k) is the informative one: it measures whether the production scorer (scorer_norm=off raw-dot)
ranks the needle inside the 2048-token budget.

Cross-process paths (CRITICAL): env vars set at server launch do NOT reach SGLang TP worker subprocesses
(BL-20260602), so the worker resolves the sink/trial dir from ITS cwd default (``cwd/.sglang_ds_oracle``).
The server is therefore launched with cwd=evidence/ and this driver points at evidence/.sglang_ds_oracle —
the same dir — via the driver-side env overrides. A wrong needle span makes the server emit a
``span_out_of_range`` hard-failure marker, so an incorrect offline tokenization FAILS LOUD (exit 2) rather
than producing a silently-wrong artifact.

Usage:  niah_recall_oracle.py [--base-url URL] [--oracle-dir DIR] [--dense-tokens N] [--sparse-tokens N]
                              [--num N] [--decode-steps N]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import defaultdict

import requests
from transformers import AutoTokenizer

from sglang.srt.layers.attention.double_sparsity import oracle_artifact_sink as sink

HERE = os.path.dirname(os.path.abspath(__file__))
EVID = os.path.join(HERE, "evidence")
MODEL = os.environ.get(
    "DS_MODEL_PATH",
    "/cluster-storage/models/models--zai-org--GLM-5.1-FP8/snapshots/f396cf805182f4ca10fa675e1a99815b3ca384db",
)
INDEX_TOPK = 2048
_HARD_FAILURES = ("span_out_of_range", "exception")
# Benign filler; one unit is a few tokens. The needle is a unique magic number per trial.
_FILLER = ("The quarterly logistics review records routine warehouse activity across the regional "
           "depots, with nothing unusual to report for this period. ")


def _niah_needle(seed: int) -> str:
    return f"Important: the hidden vault passcode for record {seed} is {7_000_000 + seed * 31 + 13}."


def _make_niah_prompt(target_tokens: int, tok, seed: int, needle: str) -> str:
    """Filler + needle (near the middle) + recall question, ~target_tokens tokens."""
    unit_tok = max(1, len(tok(_FILLER, add_special_tokens=False)["input_ids"]))
    question = " Question: what is the hidden vault passcode mentioned above? Answer:"
    q_tok = len(tok(question, add_special_tokens=False)["input_ids"])
    needle_tok = len(tok(needle + " ", add_special_tokens=False)["input_ids"])
    n_units = max(4, (target_tokens - q_tok - needle_tok) // unit_tok)
    half = n_units // 2
    return (_FILLER * half) + needle + " " + (_FILLER * (n_units - half)) + question


def needle_logical_span(tok, prompt: str, needle: str):
    """Logical token span of the needle, via raw-prompt offset mapping (no special tokens — the same
    logical domain the server's DS selector indexes)."""
    enc = tok(prompt, return_offsets_mapping=True, add_special_tokens=False)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    cstart = prompt.find(needle)
    if cstart < 0:
        return [], len(ids)
    cend = cstart + len(needle)
    span = [i for i, (a, b) in enumerate(offs) if a < cend and b > cstart]
    return span, len(ids)


def _generate_decode(base: str, prompt: str, decode_steps: int):
    """Raw /generate that FORCES decode forwards (DS selection/oracle runs only in decode). ignore_eos
    forces continuation past the immediate EOS these instruction prompts emit. Returns prompt_tokens."""
    r = requests.post(
        f"{base}/generate",
        json={"text": prompt,
              "sampling_params": {"max_new_tokens": int(decode_steps) + 1, "temperature": 0,
                                  "ignore_eos": True}},
        timeout=600,
    )
    r.raise_for_status()
    return r.json()["meta_info"].get("prompt_tokens")


def _read_sink(path: str):
    recs = []
    if not path or not os.path.exists(path):
        return recs
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    recs.append(json.loads(line))
                except ValueError:
                    pass  # trailing partial write
    return recs


def _recall_at(rec, k: int):
    rk = rec.get("recall_at_k") or {}
    # JSON serialized int keys -> strings; tolerate either.
    if str(k) in rk:
        return bool(rk[str(k)])
    if k in rk:
        return bool(rk[k])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=os.environ.get("DS_BASE_URL", "http://127.0.0.1:30000"))
    ap.add_argument("--oracle-dir", default=os.path.join(EVID, ".sglang_ds_oracle"))
    ap.add_argument("--dense-tokens", type=int, default=1200)   # < top_k 2048 -> selects all
    ap.add_argument("--sparse-tokens", type=int, default=4500)  # > top_k 2048 -> prunes
    ap.add_argument("--num", type=int, default=8)
    ap.add_argument("--decode-steps", type=int, default=4)
    args = ap.parse_args()

    # Point THIS process's trial/sink at the same dir the server's worker uses (its cwd/.sglang_ds_oracle).
    os.makedirs(args.oracle_dir, exist_ok=True)
    trial_file = os.path.join(args.oracle_dir, "trial.json")
    sink_path = os.path.join(args.oracle_dir, "sink.jsonl")
    os.environ["SGLANG_DS_RECALL_ORACLE_TRIAL_FILE"] = trial_file
    os.environ["SGLANG_DS_RECALL_ORACLE_PATH"] = sink_path
    print(f"[niah] base={args.base_url} oracle_dir={args.oracle_dir}")

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    regimes = {"dense": args.dense_tokens, "sparse": args.sparse_tokens}

    # --- Alignment probe: the server may prepend a BOS, shifting its logical KV positions by a constant
    # relative to our offline (add_special_tokens=False) span. A wrong span would silently measure the wrong
    # tokens' rank (span_out_of_range only catches a GROSS overflow), so measure the server-vs-offline token
    # delta on a representative prompt of EACH regime and assert it is a single consistent small offset. ---
    deltas = {}
    for regime, target in regimes.items():
        p = _make_niah_prompt(target, tok, seed=999, needle=_niah_needle(999))
        _, ntok = needle_logical_span(tok, p, _niah_needle(999))
        ptoks = _generate_decode(args.base_url, p, 1)
        deltas[regime] = (ptoks - ntok) if ptoks is not None else None
    if len(set(deltas.values())) != 1 or None in deltas.values() or deltas["dense"] not in (0, 1):
        print(f"FAIL (fail-closed): inconsistent/implausible server-vs-offline token delta {deltas} — "
              f"offline tokenization does not match the server's logical domain.", file=sys.stderr)
        raise SystemExit(2)
    delta = deltas["dense"]
    print(f"[niah] server-vs-offline token delta = {delta} (applied to every needle span)", flush=True)

    open(sink_path, "w").close()  # truncate AFTER the probe; start the measured run clean
    issued = defaultdict(list)        # regime -> [request_id]
    server_tokens = {}                # request_id -> server prompt_tokens
    for regime, target in regimes.items():
        t0 = time.time()
        for idx in range(args.num):
            needle = _niah_needle(idx)
            prompt = _make_niah_prompt(target, tok, seed=1000 + idx, needle=needle)
            span, ntok = needle_logical_span(tok, prompt, needle)
            if not span:
                print(f"[niah] WARN {regime} trial {idx}: needle not found offline; skipping", flush=True)
                continue
            span = [p + delta for p in span]   # shift to the server's KV domain
            req_id = f"{regime}-i{idx}"
            sink.set_active_trial(req_id, idx, span)
            try:
                ptoks = _generate_decode(args.base_url, prompt, args.decode_steps)
            finally:
                sink.clear_active_trial()
            issued[regime].append(req_id)
            server_tokens[req_id] = ptoks
        print(f"[niah] {regime:>6} target~{target}t: issued {len(issued[regime])}/{args.num} "
              f"trials ({time.time()-t0:.1f}s)", flush=True)

    # ---- Fail-closed verification + reduction ----
    recs = _read_sink(sink_path)
    by_req = defaultdict(list)
    failures = defaultdict(int)
    for r in recs:
        if "failure" in r:
            failures[str(r["failure"]).split(":")[0]] += 1
        elif r.get("request_id") is not None:
            by_req[r["request_id"]].append(r)

    problems = []
    regime_out = {}
    for regime in ("dense", "sparse"):
        ids = issued[regime]
        with_recs = [rid for rid in ids if by_req.get(rid)]
        missing = [rid for rid in ids if not by_req.get(rid)]
        if not ids:
            problems.append(f"{regime}: zero trials issued")
        if missing:
            problems.append(f"{regime}: {len(missing)} trial(s) produced no oracle record: {missing[:4]}")
        rows = [r for rid in with_recs for r in by_req[rid]]
        rk = [_recall_at(r, INDEX_TOPK) for r in rows]
        rk = [v for v in rk if v is not None]
        worst = [int(r["needle_worst_rank"]) for r in rows if "needle_worst_rank" in r]
        contains = [bool(r["selected_contains_needle"]) for r in rows if "selected_contains_needle" in r]
        if not rk:
            problems.append(f"{regime}: zero usable recall records")
        regime_out[regime] = {
            "trials_issued": len(ids),
            "trials_with_records": len(with_recs),
            "oracle_records": len(rows),
            "recall_at_2048": (round(sum(rk) / len(rk), 4) if rk else None),
            "recall_at_2048_records": len(rk),
            "selected_contains_needle_rate": (round(sum(contains) / len(contains), 4) if contains else None),
            "needle_worst_rank": ({"min": min(worst), "median": int(statistics.median(worst)),
                                   "max": max(worst)} if worst else None),
            "server_prompt_tokens_sample": {rid: server_tokens.get(rid) for rid in ids[:3]},
        }

    hard = sum(failures[k] for k in _HARD_FAILURES)
    if hard:
        problems.append(f"hard oracle failure markers: "
                        f"{ {k: failures[k] for k in _HARD_FAILURES if failures[k]} }")

    report = {
        "ac": "AC-2.4 NIAH recall-oracle@2048 — production DS scorer (scorer_norm=off raw-dot)",
        "arm": "production_ds",
        "corroboration_only": True,
        "note": ("recall@2048 is a NIAH score-ranking property, NOT a generic selected-index equivalence "
                 "proof and NOT scorer exoneration (plan AC-2.4/DEC). Dense (prompt<top_k) selects all "
                 "tokens so recall is trivially 1.0; sparse (prompt>top_k) measures whether the production "
                 "raw-dot scorer ranks the needle inside the 2048 budget."),
        "index_topk": INDEX_TOPK,
        "decode_steps_per_trial": args.decode_steps,
        "server_vs_offline_token_delta": delta,
        "failure_markers": dict(failures),
        "regimes": regime_out,
        "source_oracle_dir_basename": os.path.basename(os.path.normpath(args.oracle_dir)),
    }
    out = os.path.join(EVID, "ac2_4_recall_oracle.json")
    json.dump(report, open(out, "w"), indent=2)
    print(json.dumps(regime_out, indent=2))
    print("wrote", out)

    if problems:
        print("FAIL (fail-closed): " + " | ".join(problems), file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
