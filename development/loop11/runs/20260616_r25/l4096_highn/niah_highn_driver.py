"""L4096 high-n recall driver — CHARACTERIZATION ONLY.

Reuses the EXACT cache-engaging NIAH oracle path that the R25 p1_recall serving
run used (development/loop7/niah_oracle_sweep.py helpers: same prompt
construction h._niah_needle / h._make_niah_prompt, same logical-span mapping,
same cross-process trial registration sink.set_active_trial). The ONLY addition
vs niah_oracle_sweep.py is that this driver also records, per request, the
server-reported meta_info["cached_tokens"] so cache engagement is PROVEN from raw
data (owner rule #1) rather than inferred.

No new production code. This file lives entirely under the run dir.

Per request it issues a raw /generate forcing `decode-steps` decode forwards
(ignore_eos), exactly like the proven sweep. The DS recall oracle (server, eager)
then writes per-(trial,layer,decode_step) recall@K records into the sink keyed by
request_id; the per-request cached_tokens index lets us (a) prove the ON arm hit
cache and the OFF control did not, and (b) match the off/on populations by id.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import requests

# Anchor to the repo root by walking up until we find test/manual + python
# (robust to this driver's directory depth).
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
    """Force `decode_steps` decode forwards; return full meta_info dict."""
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
    return r.json()["meta_info"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=4096)
    ap.add_argument("--num", type=int, default=144)
    ap.add_argument("--decode-steps", type=int, default=4)
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument(
        "--warmup-length",
        type=int,
        default=1024,
        help="Pre-seed the radix prefix with this shorter length at the SAME seed per "
        "index (0 disables). L<warmup>-iN shares seed=seed_base+N with L<length>-iN, so "
        "its leading filler is an identical prefix the longer prompt then reuses. This is "
        "the exact cache-engagement mechanism of the proven niah_oracle_sweep (it ran "
        "L1024 before L4096). Warmup records land under L<warmup> and are NOT measured.",
    )
    ap.add_argument("--out", required=True, help="per-request index jsonl (with cached_tokens)")
    args = ap.parse_args()

    base = os.environ.get("DS_BASE_URL", "http://127.0.0.1:30000")

    trial_file = os.environ.get("SGLANG_DS_RECALL_ORACLE_TRIAL_FILE") or sink.default_trial_file()
    sink_path = os.environ.get("SGLANG_DS_RECALL_ORACLE_PATH") or sink.default_sink_path()
    os.environ["SGLANG_DS_RECALL_ORACLE_TRIAL_FILE"] = trial_file
    print(f"[highn] trial_file={trial_file}")
    print(f"[highn] sink={sink_path}")

    # Truncate the sink so the measured run starts clean.
    try:
        open(sink_path, "w").close()
    except OSError as e:
        print(f"[highn] WARNING: could not truncate sink {sink_path}: {e}")

    tok = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_FILE)

    L = args.length
    W = args.warmup_length
    issued = []
    t0 = time.time()

    def run_one(length, idx, fh, is_warmup):
        needle = h._niah_needle(length, idx)
        prompt = h._make_niah_prompt(length, seed=args.seed_base + idx, needle=needle)
        span, ntok = needle_logical_span(tok, prompt, needle)
        if not span:
            print(f"[highn] WARNING: needle span empty for L{length} idx={idx}; skipping")
            return None
        req_id = f"L{length}-i{idx}"
        sink.set_active_trial(req_id, idx, span)
        try:
            meta = _generate_decode(base, prompt, args.decode_steps)
        finally:
            sink.clear_active_trial()
        ptoks = meta.get("prompt_tokens")
        cached = meta.get("cached_tokens")
        fh.write(json.dumps({
            "request_id": req_id,
            "length_words": length,
            "warmup": is_warmup,
            "needle_span_start": span[0],
            "needle_span_end": span[-1],
            "offline_tokens": ntok,
            "server_tokens": ptoks,
            "cached_tokens": cached,
            "token_match": (ptoks == ntok),
        }) + "\n")
        fh.flush()
        return req_id

    with open(args.out, "w") as fh:
        for idx in range(args.num):
            # Warmup pre-seeds the shared prefix at the SAME seed (proven mechanism).
            if W and W > 0:
                run_one(W, idx, fh, is_warmup=True)
            rid = run_one(L, idx, fh, is_warmup=False)
            if rid is not None:
                issued.append(rid)
    print(f"[highn] issued {len(issued)}/{args.num} measured trials at L{L} "
          f"(warmup_length={W}) ({time.time()-t0:.1f}s)")
    print(f"[highn] per-request index -> {args.out}")
    print(f"[highn] oracle records -> {sink_path}")


if __name__ == "__main__":
    main()
