"""ExpB probe — DSA-native default cold-vs-warm logprob/continuation parity.

NO new production code: probes purely at the OUTPUT/LOGPROB level via the standard
server /generate API with return_logprob. GREEDY decode (temperature 0, top_k 1)
so any divergence is numeric, not sampling. Records, per pass:
  * cached_tokens (meta_info.cached_tokens) — the radix-hit indicator,
  * step-0 output-token top-k logprobs (input_top_logprobs / output_top_logprobs),
  * per-step (output) token ids + token logprobs for the first N decode steps,
  * the greedy continuation token id sequence.

Usage: expB_probe.py --tag B1cold --outdir <dir> [--steps 16]
The prompt is the SAME deterministic ~6090-token text as the DS cold/warm probe
(dec9_determinism q2_driver), reconstructed here and sha256-verified.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys
import requests

BASE = os.environ.get("DS_BASE_URL", "http://127.0.0.1:30000")
EXPECT_SHA = "86165d70d90b26b0ccaa27eececa4d2ef2264a8ade6d3e9c80402a310b1a416b"


def build_prompt():
    base_sentence = "The quick brown fox jumps over the lazy dog near the riverbank at dawn. "
    n_sent = max(1, 3800 // 12)
    long_prefix = "".join(f"[{i}] {base_sentence}" for i in range(n_sent))
    return long_prefix + " Summarize the passage above in one sentence."


def generate(prompt, steps, top_logprobs_num):
    r = requests.post(
        f"{BASE}/generate",
        json={
            "text": prompt,
            "sampling_params": {
                "max_new_tokens": int(steps),
                "temperature": 0.0,
                "top_k": 1,
            },
            "return_logprob": True,
            # NOTE: do NOT set logprob_start_len=0 — forcing full-prompt input
            # logprobs disables radix prefix caching for the request (the warm
            # send then recomputes the whole prompt -> cached_tokens=0). We only
            # need OUTPUT-token logprobs + step-0 top-k, which are returned with
            # the default logprob_start_len, so the warm send gets a real hit.
            "top_logprobs_num": int(top_logprobs_num),
        },
        timeout=600,
    )
    r.raise_for_status()
    return r.json()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--top-logprobs-num", type=int, default=20)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    prompt = build_prompt()
    sha = hashlib.sha256(prompt.encode()).hexdigest()
    assert sha == EXPECT_SHA, f"prompt sha mismatch {sha} != {EXPECT_SHA}"

    resp = generate(prompt, args.steps, args.top_logprobs_num)
    meta = resp.get("meta_info", {})

    # output_token_logprobs: list of [logprob, token_id, token_text]
    otl = meta.get("output_token_logprobs") or []
    out_steps = [
        {"step": i, "logprob": e[0], "token_id": e[1], "token_text": e[2]}
        for i, e in enumerate(otl)
    ]
    continuation_ids = [e[1] for e in otl]

    # step-0 top-k logprobs over the vocab for the FIRST generated token.
    # output_top_logprobs_val/idx OR output_top_logprobs as [[lp, tid, txt],...]
    step0_topk = None
    otlp = meta.get("output_top_logprobs")
    if otlp:
        step0 = otlp[0]
        step0_topk = [{"logprob": e[0], "token_id": e[1]} for e in step0]
    else:
        v = meta.get("output_top_logprobs_val")
        idx = meta.get("output_top_logprobs_idx")
        if v and idx:
            step0_topk = [{"logprob": lp, "token_id": ti} for lp, ti in zip(v[0], idx[0])]

    rec = {
        "tag": args.tag,
        "prompt_sha256": sha,
        "prompt_chars": len(prompt),
        "prompt_tokens": meta.get("prompt_tokens"),
        "completion_tokens": meta.get("completion_tokens"),
        "cached_tokens": meta.get("cached_tokens"),
        "finish_reason": meta.get("finish_reason"),
        "n_steps_captured": len(out_steps),
        "continuation_token_ids": continuation_ids,
        "per_step": out_steps,
        "step0_topk_logprobs": step0_topk,
        "text": (resp.get("text") or "")[:200],
    }
    out = os.path.join(args.outdir, f"{args.tag}.json")
    with open(out, "w") as fh:
        json.dump(rec, fh, indent=2)
    print(f"[expB] {args.tag}: cached_tokens={rec['cached_tokens']} "
          f"prompt_tokens={rec['prompt_tokens']} steps={rec['n_steps_captured']} -> {out}")
    print(f"  continuation_ids[:8]={continuation_ids[:8]}")
    if out_steps:
        print(f"  step0: id={out_steps[0]['token_id']} logprob={out_steps[0]['logprob']!r}")


if __name__ == "__main__":
    main()
