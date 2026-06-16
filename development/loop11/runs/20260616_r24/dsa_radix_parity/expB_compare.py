"""ExpB compare — cold vs warm logprob bit-identity & continuation divergence.

Reads two probe JSONs and reports:
  * cached_tokens for each,
  * max |Δ logprob| over the matched first-N steps (per-step output token logprob),
  * step-0 max |Δ logprob| over the matched top-k vocab entries,
  * first divergent greedy continuation token step (or None if identical),
  * bit-identical verdict.
"""
from __future__ import annotations
import argparse, json, os, sys


def load(p):
    with open(p) as fh:
        return json.load(fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="cold/fresh probe json")
    ap.add_argument("--b", required=True, help="warm/second probe json")
    ap.add_argument("--label", default="cold_vs_warm")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    a = load(args.a)
    b = load(args.b)

    # continuation divergence
    ca = a["continuation_token_ids"]
    cb = b["continuation_token_ids"]
    first_div = None
    for i, (x, y) in enumerate(zip(ca, cb)):
        if x != y:
            first_div = i
            break
    cont_identical = (ca == cb)

    # per-step output-token logprob delta (matched steps)
    max_dlp = 0.0
    max_dlp_step = None
    pa = {s["step"]: s for s in a["per_step"]}
    pb = {s["step"]: s for s in b["per_step"]}
    per_step_delta = []
    for st in sorted(set(pa) & set(pb)):
        la = pa[st]["logprob"]
        lb = pb[st]["logprob"]
        same_tok = pa[st]["token_id"] == pb[st]["token_id"]
        d = abs(la - lb) if (la is not None and lb is not None) else None
        per_step_delta.append({
            "step": st, "tok_a": pa[st]["token_id"], "tok_b": pb[st]["token_id"],
            "same_token": same_tok, "logprob_a": la, "logprob_b": lb, "abs_delta": d,
        })
        if d is not None and d > max_dlp:
            max_dlp = d
            max_dlp_step = st

    # step-0 top-k vocab logprob delta (matched on token_id)
    step0_max = 0.0
    ta = {e["token_id"]: e["logprob"] for e in (a.get("step0_topk_logprobs") or [])}
    tb = {e["token_id"]: e["logprob"] for e in (b.get("step0_topk_logprobs") or [])}
    common = set(ta) & set(tb)
    for tid in common:
        d = abs(ta[tid] - tb[tid])
        if d > step0_max:
            step0_max = d
    step0_topk_bit_identical = (len(common) > 0 and step0_max == 0.0
                                and set(ta) == set(tb))

    logprobs_bit_identical = (max_dlp == 0.0 and step0_max == 0.0
                              and cont_identical)

    result = {
        "label": args.label,
        "a_tag": a["tag"], "b_tag": b["tag"],
        "a_cached_tokens": a["cached_tokens"], "b_cached_tokens": b["cached_tokens"],
        "a_prompt_tokens": a["prompt_tokens"], "b_prompt_tokens": b["prompt_tokens"],
        "prompt_sha256": a["prompt_sha256"],
        "continuation_identical": cont_identical,
        "first_divergent_step": first_div,
        "max_abs_delta_logprob_per_step": max_dlp,
        "max_abs_delta_logprob_step": max_dlp_step,
        "step0_topk_max_abs_delta_logprob": step0_max,
        "step0_topk_token_set_identical": (set(ta) == set(tb)),
        "step0_topk_bit_identical": step0_topk_bit_identical,
        "logprobs_bit_identical": logprobs_bit_identical,
        "per_step_delta": per_step_delta,
        "a_continuation_ids": ca,
        "b_continuation_ids": cb,
    }
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)

    print(f"\n=== ExpB compare {args.label} ===")
    print(f"  {a['tag']} cached_tokens={a['cached_tokens']}  {b['tag']} cached_tokens={b['cached_tokens']}")
    print(f"  continuation identical: {cont_identical}  first_divergent_step: {first_div}")
    print(f"  max |Δ logprob| per-step: {max_dlp!r} (step {max_dlp_step})")
    print(f"  step-0 top-k max |Δ logprob|: {step0_max!r}  token-set identical: {set(ta)==set(tb)}")
    print(f"  LOGPROBS BIT-IDENTICAL: {logprobs_bit_identical}")
    print(f"  -> {args.out}")


if __name__ == "__main__":
    main()
