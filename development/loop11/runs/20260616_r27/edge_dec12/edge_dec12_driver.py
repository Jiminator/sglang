"""R27 GATE C (DEC-12) — production-representative-reuse edge probe (length 4096, page 64).

DEC-12 re-authors the edge gate around the PRODUCTION reuse regime. On the 4096-ISL
workload the radix prefix hit is ~55% (cached ~2752 is the production upper bound, ~63%
of the partial-arm prompt). The R26 partial arm warmed the WHOLE L4096 prompt and
re-requested it with a tail, which reuses ~4288 tokens (~98%, NEAR-FULL) — that is NOT
the production operating point. R27 splits the partial measurement into two arms:

  partial  @ production : warm a ~2752-TOKEN PREFIX of P, then request prefix+tail so the
                          radix reuses floor(2752/64) aligned pages and recomputes the
                          partial page. The needle lives at depth<=0.65*L4096 (~<=2662
                          logical tokens) so it is INSIDE the reused prefix. cached~2752.
  nearfull @ ~4288      : the R26 mechanic (warm full P, request P+tail). cached~4288.
                          Recorded as out_of_contract_value_affecting (NOT a gate input).

Plus the unchanged cold / boundary / eviction arms (same proven cache-engaging niah path
as R26: loop7 _niah_needle/_make_niah_prompt helpers, L1024 same-seed warmup, recall
oracle keyed by request_id, per-request meta_info["cached_tokens"] recorded raw).

Statistical unit = the NEEDLE, paired by needle index across arms (same seed/needle/prompt).
One arm per invocation (cold needs a radix-OFF server, the reuse arms a radix-ON server).

Per-arm request_id scheme:
  cold     -> "L4096-cold-iN"   (OFF server, cached MUST be 0)
  boundary -> "L4096-bnd-iN"    (ON server, full page-aligned small reuse, cached>0 %64==0)
  partial  -> "L4096-prt-iN"    (ON server, production ~2752 reuse, aligned pages + partial)
  nearfull -> "L4096-nfl-iN"    (ON server, ~4288 near-full reuse; out-of-contract char.)
  evict    -> "L4096-evc-iN"    (ON server, flush_cache -> recompute, cached MUST fall to 0)
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
from transformers import PreTrainedTokenizerFast  # noqa: E402

from sglang.srt.layers.attention.double_sparsity import oracle_artifact_sink as sink  # noqa: E402

TOKENIZER_FILE = os.environ.get("DS_TOKENIZER_FILE")

# The partial-page tail: a short fixed clause whose token count is NOT a page multiple,
# so the divergence point falls mid-page. SAME string as the R26 / R25 p3 partial driver.
PARTIAL_TAIL = " Additionally, note the following minor clarifying remark here."


def needle_logical_span(tok, prompt, needle):
    enc = tok(prompt, return_offsets_mapping=True, add_special_tokens=False)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    cstart = prompt.find(needle)
    if cstart < 0:
        return [], len(ids)
    cend = cstart + len(needle)
    span = [i for i, (a, b) in enumerate(offs) if a < cend and b > cstart]
    return span, len(ids)


def prefix_to_tokens(tok, prompt, n_tokens):
    """Return (prefix_text, n_prefix_tokens) for the first ~n_tokens whitespace-aligned
    tokens of `prompt`. We cut on a WORD boundary at-or-below the token target so the
    prefix stays valid text and the needle (which lives at depth<=0.65) is preserved."""
    words = prompt.split(" ")
    lo, hi = 1, len(words)
    best = words
    # Binary search the largest word-count whose tokenization is <= n_tokens.
    while lo <= hi:
        mid = (lo + hi) // 2
        cand = " ".join(words[:mid])
        ntok = len(tok(cand, add_special_tokens=False)["input_ids"])
        if ntok <= n_tokens:
            best = words[:mid]
            lo = mid + 1
        else:
            hi = mid - 1
    text = " ".join(best)
    return text, len(tok(text, add_special_tokens=False)["input_ids"])


def _generate_decode(base, prompt, decode_steps):
    r = requests.post(
        f"{base}/generate",
        json={
            "text": prompt,
            "sampling_params": {
                "max_new_tokens": int(decode_steps) + 1,
                "temperature": 0,
                "ignore_eos": True,
            },
        },
        timeout=600,
    )
    r.raise_for_status()
    return r.json()["meta_info"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--arm",
        required=True,
        choices=["cold", "cold_partial", "boundary", "partial", "nearfull", "evict"],
    )
    ap.add_argument("--length", type=int, default=4096)
    ap.add_argument("--num", type=int, default=144)
    ap.add_argument("--decode-steps", type=int, default=4)
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--warmup-length", type=int, default=1024)
    ap.add_argument(
        "--partial-prefix-tokens",
        type=int,
        default=2752,
        help="production reuse upper bound: warm this many tokens of the prefix",
    )
    ap.add_argument("--page-size", type=int, default=64)
    ap.add_argument("--out", required=True, help="per-request index jsonl (with cached_tokens)")
    args = ap.parse_args()

    base = os.environ.get("DS_BASE_URL", "http://127.0.0.1:30000")

    trial_file = os.environ.get("SGLANG_DS_RECALL_ORACLE_TRIAL_FILE") or sink.default_trial_file()
    sink_path = os.environ.get("SGLANG_DS_RECALL_ORACLE_PATH") or sink.default_sink_path()
    os.environ["SGLANG_DS_RECALL_ORACLE_TRIAL_FILE"] = trial_file
    print(f"[edge:{args.arm}] trial_file={trial_file}")
    print(f"[edge:{args.arm}] sink={sink_path}")

    try:
        open(sink_path, "w").close()
    except OSError as e:
        print(f"[edge:{args.arm}] WARNING: could not truncate sink {sink_path}: {e}")

    tok = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_FILE)

    L = args.length
    W = args.warmup_length
    ps = args.page_size
    arm = args.arm
    tag = {
        "cold": "cold",
        "cold_partial": "cprt",
        "boundary": "bnd",
        "partial": "prt",
        "nearfull": "nfl",
        "evict": "evc",
    }[arm]

    def issue(prompt, needle, req_id, idx):
        span, ntok = needle_logical_span(tok, prompt, needle)
        if not span:
            print(f"[edge:{arm}] WARNING: needle span empty for {req_id}; skipping")
            return None
        sink.set_active_trial(req_id, idx, span)
        try:
            meta = _generate_decode(base, prompt, args.decode_steps)
        finally:
            sink.clear_active_trial()
        return {
            "request_id": req_id,
            "needle_span_start": span[0],
            "needle_span_end": span[-1],
            "offline_tokens": ntok,
            "server_tokens": meta.get("prompt_tokens"),
            "cached_tokens": meta.get("cached_tokens"),
            "token_match": (meta.get("prompt_tokens") == ntok),
        }

    issued = 0
    t0 = time.time()
    with open(args.out, "w") as fh:
        for idx in range(args.num):
            needle = h._niah_needle(L, idx)
            P = h._make_niah_prompt(L, seed=args.seed_base + idx, needle=needle)

            if arm == "cold":
                # COLD/FRESH control: radix-OFF server -> cached MUST be 0. No warmup.
                # Pairs with boundary + eviction (both request the full L4096 P).
                rec = issue(P, needle, f"L{L}-cold-i{idx}", idx)

            elif arm == "cold_partial":
                # MATCHED COLD control for the partial@production arm: radix-OFF server,
                # issues the EXACT prefix_2752+tail prompt the partial arm measures (same
                # token length ~2763) so the partial radix effect is isolated from the
                # prompt-length confound. cached MUST be 0. No warmup.
                prefix_text, n_pref = prefix_to_tokens(tok, P, args.partial_prefix_tokens)
                if needle not in prefix_text:
                    print(
                        f"[edge:{arm}] WARNING: needle not in {n_pref}-tok prefix for i{idx}; skipping"
                    )
                    continue
                rec = issue(prefix_text + PARTIAL_TAIL, needle, f"L{L}-cprt-i{idx}", idx)

            elif arm == "boundary":
                # Full page-aligned SMALL reuse: warm the L1024-iN same-seed prefix, then
                # the measured L4096-iN reuses those aligned pages (cached>0, %64==0).
                if W and W > 0:
                    wn = h._niah_needle(W, idx)
                    wp = h._make_niah_prompt(W, seed=args.seed_base + idx, needle=wn)
                    ws, _ = needle_logical_span(tok, wp, wn)
                    if ws:
                        sink.set_active_trial(f"L{W}-warm-i{idx}", idx, ws)
                        try:
                            _generate_decode(base, wp, 2)
                        finally:
                            sink.clear_active_trial()
                rec = issue(P, needle, f"L{L}-bnd-i{idx}", idx)

            elif arm == "partial":
                # PRODUCTION partial-page hit (cached ~2752, ~63%): warm a ~2752-token
                # PREFIX of P (needle is at depth<=0.65*L4096 ~<=2662 tokens, INSIDE the
                # prefix), then request prefix+tail. The radix reuses floor(2752/64)
                # aligned pages + recomputes the partial page; recall measured on the
                # reused region.
                prefix_text, n_pref = prefix_to_tokens(tok, P, args.partial_prefix_tokens)
                if needle not in prefix_text:
                    print(
                        f"[edge:{arm}] WARNING: needle not in {n_pref}-tok prefix for i{idx}; skipping"
                    )
                    continue
                wn_span, _ = needle_logical_span(tok, prefix_text, needle)
                if wn_span:
                    sink.set_active_trial(f"L{L}-pwarm-i{idx}", idx, wn_span)
                    try:
                        _generate_decode(base, prefix_text, 2)
                    finally:
                        sink.clear_active_trial()
                rec = issue(prefix_text + PARTIAL_TAIL, needle, f"L{L}-prt-i{idx}", idx)

            elif arm == "nearfull":
                # NEAR-FULL reuse (cached ~4288, ~98%) — the R26 mechanic, recorded as the
                # out-of-contract value-affecting characterization (NOT a gate input).
                ws, _ = needle_logical_span(tok, P, needle)
                if ws:
                    sink.set_active_trial(f"L{L}-nwarm-i{idx}", idx, ws)
                    try:
                        _generate_decode(base, P, 2)
                    finally:
                        sink.clear_active_trial()
                rec = issue(P + PARTIAL_TAIL, needle, f"L{L}-nfl-i{idx}", idx)

            else:  # evict
                # Eviction/recompute: warm the prefix, flush_cache, then re-request P ->
                # recompute. cached MUST fall to 0; recall must equal cold.
                ws, _ = needle_logical_span(tok, P, needle)
                if ws:
                    sink.set_active_trial(f"L{L}-ewarm-i{idx}", idx, ws)
                    try:
                        _generate_decode(base, P, 2)
                    finally:
                        sink.clear_active_trial()
                fr = requests.post(f"{base}/flush_cache", timeout=120)
                if fr.status_code != 200:
                    print(f"[edge:{arm}] WARNING: flush_cache status={fr.status_code} for i{idx}")
                rec = issue(P, needle, f"L{L}-evc-i{idx}", idx)

            if rec is not None:
                rec["arm"] = arm
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                issued += 1

    print(f"[edge:{arm}] issued {issued}/{args.num} measured trials at L{L} "
          f"(warmup_length={W}, partial_prefix_tokens={args.partial_prefix_tokens}, "
          f"page={ps}) ({time.time()-t0:.1f}s)")
    print(f"[edge:{arm}] per-request index -> {args.out}")
    print(f"[edge:{arm}] oracle records -> {sink_path}")


if __name__ == "__main__":
    main()
