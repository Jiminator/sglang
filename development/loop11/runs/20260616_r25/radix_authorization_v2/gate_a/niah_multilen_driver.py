"""GATE A multi-length recall driver — radix OFF vs ON, recall@2048.

A thin generalization of the PROVEN single-length cache-engaging driver
(development/loop11/runs/20260616_r25/l4096_highn/niah_highn_driver.py): it sweeps
several measured lengths and, for EACH measured trial, first issues a SHORTER
SAME-SEED warmup so the radix-ON arm reuses a cached shared-prefix (the exact
mechanism the n=144 L4096 run used to prove cache engagement). The ONLY thing this
file adds vs niah_oracle_sweep.py is recording the server-reported
meta_info["cached_tokens"] PER REQUEST so cache engagement is PROVEN from raw data
per length (owner rule #1), not inferred.

No new production code. Lives entirely under the run dir. Reuses the identical
prompt construction (h._niah_needle / h._make_niah_prompt) and the cross-process
trial sink (sink.set_active_trial) of the proven niah_oracle_sweep path.

For a measured length L at index N (seed = seed_base + N):
  * warmup length W(L) = max(min_warmup, L // warmup_div) is run FIRST at the SAME
    seed. _make_niah_prompt draws filler from random.Random(seed); the word stream
    depends only on the seed, so the L<W>-iN prompt's leading filler is a prefix of
    the L<L>-iN prompt's leading filler — an identical radix prefix the longer
    measured prompt then reuses. Warmup records land under L<W> and are NOT measured.
  * the measured L-iN trial then hits that cached prefix (cached_tokens > 0 on the
    radix-ON boot; == 0 on the --disable-radix-cache control).

Off and on are two boots over the IDENTICAL measured prompt set (same seeds/needles),
paired by request_id downstream.
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

TOKENIZER_FILE = os.environ.get(
    "DS_TOKENIZER_FILE",
    "/cluster-storage/models/deepseek-ai/DeepSeek-V3.2/tokenizer.json",
)


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
        timeout=900,
    )
    return r.json()["meta_info"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", type=int, nargs="+", default=[1024, 4096, 16384])
    ap.add_argument("--num", type=int, default=128)
    ap.add_argument("--decode-steps", type=int, default=4)
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument(
        "--warmup-div",
        type=int,
        default=4,
        help="Per-measured-length warmup length = max(min_warmup, L // warmup-div), "
        "run at the SAME seed immediately before each measured trial to pre-seed the "
        "shared radix prefix (the proven cache-engagement mechanism).",
    )
    ap.add_argument("--min-warmup", type=int, default=256)
    ap.add_argument("--out", required=True, help="per-request index jsonl (cached_tokens)")
    args = ap.parse_args()

    base = os.environ.get("DS_BASE_URL", "http://127.0.0.1:30000")

    trial_file = os.environ.get("SGLANG_DS_RECALL_ORACLE_TRIAL_FILE") or sink.default_trial_file()
    sink_path = os.environ.get("SGLANG_DS_RECALL_ORACLE_PATH") or sink.default_sink_path()
    os.environ["SGLANG_DS_RECALL_ORACLE_TRIAL_FILE"] = trial_file
    print(f"[multilen] trial_file={trial_file}")
    print(f"[multilen] sink={sink_path}")
    print(f"[multilen] lengths={args.lengths} num={args.num} warmup_div={args.warmup_div} "
          f"min_warmup={args.min_warmup}")

    try:
        open(sink_path, "w").close()
    except OSError as e:
        print(f"[multilen] WARNING: could not truncate sink {sink_path}: {e}")

    tok = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_FILE)

    def run_one(length, idx, fh, is_warmup):
        needle = h._niah_needle(length, idx)
        prompt = h._make_niah_prompt(length, seed=args.seed_base + idx, needle=needle)
        span, ntok = needle_logical_span(tok, prompt, needle)
        if not span:
            print(f"[multilen] WARNING: empty needle span L{length} idx={idx}; skipping")
            return None
        req_id = f"L{length}-i{idx}"
        sink.set_active_trial(req_id, idx, span)
        try:
            meta = _generate_decode(base, prompt, args.decode_steps)
        finally:
            sink.clear_active_trial()
        fh.write(json.dumps({
            "request_id": req_id,
            "length_words": length,
            "warmup": is_warmup,
            "needle_span_start": span[0],
            "needle_span_end": span[-1],
            "offline_tokens": ntok,
            "server_tokens": meta.get("prompt_tokens"),
            "cached_tokens": meta.get("cached_tokens"),
            "token_match": (meta.get("prompt_tokens") == ntok),
        }) + "\n")
        fh.flush()
        return req_id

    issued = {L: 0 for L in args.lengths}
    t0 = time.time()
    with open(args.out, "w") as fh:
        for L in args.lengths:
            W = max(args.min_warmup, L // args.warmup_div)
            tL = time.time()
            for idx in range(args.num):
                # warmup at a shorter SAME-seed length pre-seeds the shared prefix.
                if W < L:
                    run_one(W, idx, fh, is_warmup=True)
                rid = run_one(L, idx, fh, is_warmup=False)
                if rid is not None:
                    issued[L] += 1
            print(f"[multilen] L{L} (warmup L{W}): issued {issued[L]}/{args.num} "
                  f"measured ({time.time()-tL:.1f}s)", flush=True)
    print(f"[multilen] total measured {sum(issued.values())} across {args.lengths} "
          f"({time.time()-t0:.1f}s)")
    print(f"[multilen] per-request index -> {args.out}")


if __name__ == "__main__":
    main()
