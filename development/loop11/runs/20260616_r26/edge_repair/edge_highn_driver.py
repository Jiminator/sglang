"""R26 GATE C — ROBUST-N edge probe (length 4096, page 64), the original ±0.5pp-clean
contract. Repairs the R25 thin-sample fail-open (single prompt, n=312 layer/step records
of ONE needle gave +0.641/−0.961pp within-prompt noise).

Statistical unit = the NEEDLE. For each of N>=128 distinct NIAH needles we measure
recall@2048 under each radix edge variant and compare PAIRED to the cold/fresh control
(same needle id). This reuses the EXACT proven cache-engaging niah path:
  * prompt construction h._niah_needle / h._make_niah_prompt (loop7 helpers),
  * the L1024 same-seed warmup that seeds the shared radix prefix (the proven mechanism
    of niah_oracle_sweep / R25 l4096_highn: each L4096-iN reuses the L1024-iN aligned
    prefix pages → cached_tokens in [320,640]),
  * the DS recall-oracle sink keyed by request_id (set_active_trial / clear_active_trial),
  * per-request meta_info["cached_tokens"] recorded to PROVE engagement from raw data.

No new production code; this driver lives entirely under the run dir.

This driver runs ONE arm per invocation (selected by --arm), because the cold/fresh
control needs a radix-OFF server (cached==0) while the on/reuse variants need a
radix-ON server. The runner boots OFF first (cold), tears down, then boots ON and
runs boundary+partial+eviction in the SAME server. analyze pairs them by request_id.

Per-arm request_id scheme (so analyze can split + pair):
  cold     -> "L4096-cold-iN"     (OFF server, cached MUST be 0)
  boundary -> "L4096-bnd-iN"      (ON server, full page-aligned reuse, cached>0 %64==0)
  partial  -> "L4096-prt-iN"      (ON server, mid-page divergence, aligned pages reused)
  evict    -> "L4096-evc-iN"      (ON server, flush_cache→recompute, cached MUST fall to 0)

The needle (and thus the recall it drives) is IDENTICAL across arms for a given iN
(same seed, same needle, same prompt P), so per-needle recall is directly comparable.
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

# The partial-page tail: a short, fixed clause appended so the divergence point falls
# mid-page (its token count is NOT a multiple of 64), forcing the radix cache to reuse
# floor(shared/64) aligned pages and recompute the partial page. SAME string as the R25
# p3 driver's partial case (kept identical for continuity).
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
    ap.add_argument("--arm", required=True, choices=["cold", "boundary", "partial", "evict"])
    ap.add_argument("--length", type=int, default=4096)
    ap.add_argument("--num", type=int, default=144)
    ap.add_argument("--decode-steps", type=int, default=4)
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--warmup-length", type=int, default=1024)
    ap.add_argument("--page-size", type=int, default=64)
    ap.add_argument("--out", required=True, help="per-request index jsonl (with cached_tokens)")
    args = ap.parse_args()

    base = os.environ.get("DS_BASE_URL", "http://127.0.0.1:30000")

    trial_file = os.environ.get("SGLANG_DS_RECALL_ORACLE_TRIAL_FILE") or sink.default_trial_file()
    sink_path = os.environ.get("SGLANG_DS_RECALL_ORACLE_PATH") or sink.default_sink_path()
    os.environ["SGLANG_DS_RECALL_ORACLE_TRIAL_FILE"] = trial_file
    print(f"[edge:{args.arm}] trial_file={trial_file}")
    print(f"[edge:{args.arm}] sink={sink_path}")

    # Truncate the sink so each arm starts clean (the runner copies it out per arm).
    try:
        open(sink_path, "w").close()
    except OSError as e:
        print(f"[edge:{args.arm}] WARNING: could not truncate sink {sink_path}: {e}")

    tok = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_FILE)

    L = args.length
    W = args.warmup_length
    ps = args.page_size
    arm = args.arm
    tag = {"cold": "cold", "boundary": "bnd", "partial": "prt", "evict": "evc"}[arm]

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
                # COLD/FRESH control: radix-OFF server → cached MUST be 0. No warmup.
                rec = issue(P, needle, f"L{L}-cold-i{idx}", idx)

            elif arm == "boundary":
                # Full page-aligned reuse: warmup the L1024-iN same-seed prefix, then the
                # measured L4096-iN reuses those aligned pages (cached>0, %64==0).
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
                # Partial-page hit: warmup the L4096-iN prompt itself, then re-request it
                # with a short appended tail so the shared prefix diverges MID-PAGE — the
                # radix cache reuses floor(shared/64) aligned pages and recomputes the
                # partial page. The needle lives inside the shared prefix (depth<=0.65),
                # so recall is measured on the reused region.
                ws, _ = needle_logical_span(tok, P, needle)
                if ws:
                    sink.set_active_trial(f"L{L}-pwarm-i{idx}", idx, ws)
                    try:
                        _generate_decode(base, P, 2)
                    finally:
                        sink.clear_active_trial()
                rec = issue(P + PARTIAL_TAIL, needle, f"L{L}-prt-i{idx}", idx)

            else:  # evict
                # Eviction/recompute: warm the prefix, flush_cache (confirm cached→0 on a
                # tiny probe), then re-request P → recompute. cached MUST fall to 0; recall
                # must equal cold (deterministic recompute == no stale slot).
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
          f"(warmup_length={W}, page={ps}) ({time.time()-t0:.1f}s)")
    print(f"[edge:{arm}] per-request index -> {args.out}")
    print(f"[edge:{arm}] oracle records -> {sink_path}")


if __name__ == "__main__":
    main()
