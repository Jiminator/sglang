"""R25 PROBE 3 — edge_probe (page 64), radix-ON, recall_oracle + selection_capture.

Exercises three radix-cache edge cases that would expose stale-slot corruption if the
table-free DS selection read a wrong physical slot after a page reuse/recompute. The
authorizing claim is VALUE-equivalence: recall@2048 stays equivalent to a cold
baseline (a stale slot would tank recall) AND DS selection stays valid (sparse top-k,
never dense degrade) across all three.

Method (NIAH oracle, page_size=64, top_k=2048):
  Build ONE long NIAH prompt P with a known needle span. Issue requests and read each
  request's recall@2048 from the recall-oracle sink (keyed by request_id). The needle
  span is the SAME logical span for every variant (identical prefix), so the recall
  number is directly comparable. We compare each edge variant's recall@2048 to the
  COLD baseline (first, cache-cold request of P) within +/-0.5pp; any drop is the
  stale-slot signal.

Edge cases:
  (a) BOUNDARY  — page-aligned shared prefix: warm request = the IDENTICAL prompt P
      (the entire prompt is reused; shared prefix length == prompt_tokens). Plus a
      shared prefix that is an exact multiple of 64 (aligned) realized by the radix
      cache page granularity. cached_tokens MUST be > 0 (a real hit) and equal to a
      multiple of page_size for the reused pages.
  (b) PARTIAL   — partial-page hit: warm request shares prefix P then appends EXTRA
      text so the divergence point falls mid-page; the radix cache reuses the aligned
      pages and recomputes the partial page. We still measure the needle (which lives
      inside the shared prefix) and require recall equivalence.
  (c) EVICTION  — flood the cache with several distinct long prompts to evict P, then
      re-request P -> cache MISS recompute (cached_tokens drops back toward 0). Recall
      must match the cold baseline (a recompute must not corrupt selection).

Each request forces >=1 decode step (ignore_eos) so the decode-time DS selector and
recall oracle fire. Selection-capture validity (no dense fallback) is checked
separately over the preserved capture dir by no_dense_fallback_check.py.

PASS = every edge variant's recall@2048 within +/-0.5pp of the cold baseline AND every
variant produced oracle records (no silent drop) AND the boundary/partial variants
showed a real cache hit (cached_tokens>0) AND the eviction variant showed the prefix
was actually evicted (cached_tokens fell). FAIL-CLOSED otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import requests

sys.path.insert(0, "/sgl-workspace/sglang")
sys.path.insert(0, "/sgl-workspace/sglang/test/manual")
import test_double_sparsity_v32 as h  # noqa: E402
from transformers import PreTrainedTokenizerFast  # noqa: E402

from sglang.srt.layers.attention.double_sparsity import (  # noqa: E402
    oracle_artifact_sink as sink,
)

PORT = int(os.environ.get("PORT", "30000"))
BASE = f"http://127.0.0.1:{PORT}"
TOKENIZER_FILE = os.environ.get("DS_TOKENIZER_FILE")


def _needle_span(tok, prompt, needle):
    enc = tok(prompt, return_offsets_mapping=True, add_special_tokens=False)
    offs = enc["offset_mapping"]
    cstart = prompt.find(needle)
    if cstart < 0:
        return [], len(enc["input_ids"])
    cend = cstart + len(needle)
    span = [i for i, (a, b) in enumerate(offs) if a < cend and b > cstart]
    return span, len(enc["input_ids"])


def _generate(prompt, decode_steps=4):
    r = requests.post(
        f"{BASE}/generate",
        json={
            "text": prompt,
            "sampling_params": {
                "max_new_tokens": int(decode_steps) + 1,
                "temperature": 0,
                "ignore_eos": True,
            },
        },
        timeout=900,
    )
    r.raise_for_status()
    return r.json()


def _read_sink(path):
    recs = []
    if not path or not os.path.exists(path):
        return recs
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except ValueError:
                pass
    return recs


def _recall2048_for_request(recs, request_id):
    """Aggregate recall@2048 over all (layer, step) oracle records for one request."""
    hits = n = 0
    fails = 0
    for r in recs:
        if r.get("request_id") != request_id:
            continue
        if "failure" in r:
            fails += 1
            continue
        rk = r.get("recall_at_k", {})
        val = rk.get("2048", rk.get(2048))
        if val is None:
            val = int(r.get("needle_worst_rank", 1 << 30)) < 2048
        n += 1
        hits += 1 if val else 0
    pct = (100.0 * hits / n) if n else None
    return {"samples": n, "hits": hits, "pct": (round(pct, 3) if pct is not None else None), "failures": fails}


def _issue_trial(tok, sink_path, prompt, needle, request_id, idx, decode_steps=4):
    span, _ = _needle_span(tok, prompt, needle)
    if not span:
        return None, None
    sink.set_active_trial(request_id, idx, span)
    try:
        resp = _generate(prompt, decode_steps)
    finally:
        sink.clear_active_trial()
    meta = resp.get("meta_info", {})
    return meta, span


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=4096)
    ap.add_argument("--idx", type=int, default=3)
    ap.add_argument("--page-size", type=int, default=64)
    ap.add_argument("--top-k", type=int, default=2048)
    ap.add_argument("--max-delta-pp", type=float, default=0.5)
    ap.add_argument("--evict-prompts", type=int, default=6)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    trial_file = os.environ.get("SGLANG_DS_RECALL_ORACLE_TRIAL_FILE") or sink.default_trial_file()
    sink_path = os.environ.get("SGLANG_DS_RECALL_ORACLE_PATH") or sink.default_sink_path()
    os.environ["SGLANG_DS_RECALL_ORACLE_TRIAL_FILE"] = trial_file
    try:
        open(sink_path, "w").close()
    except OSError:
        pass

    tok = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_FILE)
    L, idx, ps = args.length, args.idx, args.page_size

    needle = h._niah_needle(L, idx)
    P = h._make_niah_prompt(L, seed=1000 + idx, needle=needle)

    result = {
        "probe": "P3_edge_probe_page64",
        "length": L,
        "page_size": ps,
        "top_k": args.top_k,
        "max_delta_pp": args.max_delta_pp,
        "cases": {},
    }

    # --- COLD baseline: first request of P (cache cold) ---
    meta_cold, span = _issue_trial(tok, sink_path, P, needle, "EDGE-cold", idx)
    if meta_cold is None:
        result["status"] = "FAIL"
        result["reason"] = "needle span not found in prompt P"
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=2)
        print(json.dumps(result, indent=2))
        return 1
    cold_cached = meta_cold.get("cached_tokens", 0)
    prompt_tokens = meta_cold.get("prompt_tokens")

    # --- (a) BOUNDARY: IDENTICAL prompt P -> full page-aligned reuse (warm hit) ---
    meta_b, _ = _issue_trial(tok, sink_path, P, needle, "EDGE-boundary", idx)
    boundary_cached = meta_b.get("cached_tokens", 0)

    # --- (b) PARTIAL-page: P + appended text so the divergence is mid-page ---
    # Append a short tail (not a multiple of page_size of new tokens) so the radix
    # cache reuses floor(shared/ps) aligned pages and recomputes the partial page.
    P_partial = P + " Additionally, note the following minor clarifying remark here."
    meta_p, _ = _issue_trial(tok, sink_path, P_partial, needle, "EDGE-partial", idx)
    partial_cached = meta_p.get("cached_tokens", 0)

    # --- (c) EVICTION: force the prefix OUT, then re-request P -> cache miss recompute.
    # /flush_cache clears the radix tree (the canonical "force the prefix out"
    # mechanism: the ~500K-token KV pool is far larger than a handful of long
    # prompts, so an LRU flood would NOT reliably evict P; an explicit flush IS a
    # real eviction of the cached prefix). After the flush the re-request MUST be a
    # cache miss (cached_tokens back to ~0) and recompute from scratch.
    flush_resp = requests.post(f"{BASE}/flush_cache", timeout=120)
    flushed_ok = flush_resp.status_code == 200
    # belt-and-suspenders: also push some distinct long prompts so any partially
    # rebuilt tree does not silently re-cache P before the re-request.
    for e in range(args.evict_prompts):
        en = h._niah_needle(L, 500 + e)
        ep = h._make_niah_prompt(L, seed=9000 + e, needle=en)
        _generate(ep, decode_steps=2)
    meta_e, _ = _issue_trial(tok, sink_path, P, needle, "EDGE-evicted", idx)
    evicted_cached = meta_e.get("cached_tokens", 0)
    result["flush_cache_ok"] = flushed_ok

    recs = _read_sink(sink_path)
    rc_cold = _recall2048_for_request(recs, "EDGE-cold")
    rc_boundary = _recall2048_for_request(recs, "EDGE-boundary")
    rc_partial = _recall2048_for_request(recs, "EDGE-partial")
    rc_evicted = _recall2048_for_request(recs, "EDGE-evicted")

    problems = []

    def _delta(a, b):
        if a is None or b is None:
            return None
        return round(b - a, 3)

    base_pct = rc_cold["pct"]
    result["cases"]["cold_baseline"] = {
        "cached_tokens": cold_cached,
        "prompt_tokens": prompt_tokens,
        "recall2048": rc_cold,
    }
    result["cases"]["a_boundary_aligned_full_reuse"] = {
        "cached_tokens": boundary_cached,
        "recall2048": rc_boundary,
        "delta_pp_vs_cold": _delta(base_pct, rc_boundary["pct"]),
        "cache_hit": boundary_cached > 0,
        "cached_is_page_multiple": (boundary_cached % ps == 0),
    }
    result["cases"]["b_partial_page_hit"] = {
        "cached_tokens": partial_cached,
        "recall2048": rc_partial,
        "delta_pp_vs_cold": _delta(base_pct, rc_partial["pct"]),
        "cache_hit": partial_cached > 0,
        "cached_is_page_multiple": (partial_cached % ps == 0),
    }
    result["cases"]["c_eviction_recompute"] = {
        "cached_tokens": evicted_cached,
        "recall2048": rc_evicted,
        "delta_pp_vs_cold": _delta(base_pct, rc_evicted["pct"]),
        "prefix_evicted(cached_fell_vs_boundary)": evicted_cached < boundary_cached,
    }

    # Fail-closed checks.
    if rc_cold["samples"] == 0:
        problems.append("cold baseline produced no oracle records")
    for name, rc in (
        ("a_boundary", rc_boundary),
        ("b_partial", rc_partial),
        ("c_evicted", rc_evicted),
    ):
        if rc["samples"] == 0:
            problems.append(f"{name}: no oracle records (silent drop)")
        if rc["failures"]:
            problems.append(f"{name}: {rc['failures']} oracle failure marker(s)")
    for name, rc in (
        ("a_boundary", rc_boundary),
        ("b_partial", rc_partial),
        ("c_evicted", rc_evicted),
    ):
        d = _delta(base_pct, rc["pct"])
        if d is None:
            problems.append(f"{name}: recall pct is None (cold={base_pct})")
        elif abs(d) > args.max_delta_pp:
            problems.append(
                f"{name}: recall@2048 {rc['pct']}% deviates {d:+.3f}pp from cold {base_pct}% "
                f"(>±{args.max_delta_pp}pp) — possible stale-slot corruption"
            )
    # The boundary case MUST be a real hit (else the edge is not exercised).
    if boundary_cached <= 0:
        problems.append(
            f"boundary case had NO cache hit (cached_tokens={boundary_cached}); "
            "page-aligned reuse edge not exercised"
        )
    # The partial case MUST also be a real hit (the aligned pages were reused).
    if partial_cached <= 0:
        problems.append(
            f"partial case had NO cache hit (cached_tokens={partial_cached}); "
            "partial-page reuse edge not exercised"
        )
    # The eviction case MUST be a real recompute: after flushing the prefix, the
    # re-request must NOT reuse the cached prefix (cached_tokens fell well below the
    # boundary hit). If it did not fall, the prefix was not actually evicted and the
    # recompute edge was not exercised.
    if not result.get("flush_cache_ok"):
        problems.append("flush_cache did not return 200; eviction not forced")
    if evicted_cached >= boundary_cached:
        problems.append(
            f"eviction case still hit the cache (evicted_cached={evicted_cached} >= "
            f"boundary_cached={boundary_cached}); prefix was not forced out, recompute "
            "edge not exercised"
        )

    result["needle_span_first_last"] = [span[0], span[-1]] if span else None
    result["status"] = "PASS" if not problems else "FAIL"
    result["problems"] = problems

    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
