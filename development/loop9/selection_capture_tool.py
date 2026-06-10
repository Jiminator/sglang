"""Loop-9 M0 selection-capture tool — freeze and check the production selector output.

Drives a fixed deterministic workload against a live DS server booted with the
config-borne ``selection_capture`` diagnostic (CUDA graph ON — this is the
production graph-replay selector path), then verifies and freezes the per-rank
per-(layer, decode-step) dumps the TP workers wrote:

  run     issue the fixed workload (sequential raw /generate, temperature 0,
          ignore_eos, forced decode steps) and snapshot the dump files.
  verify  fail-closed checks on a snapshot: expected file set present, indices
          + valid_lengths BIT-IDENTICAL across all TP ranks (the hard cross-rank
          gate), output contract on rank 0 (ascending, -1 padded, lengths
          consistent, in-range), run-to-run identity across passes. Emits a
          digest JSON (per-step SHA256) suitable for committing as the frozen
          baseline fingerprint.
  diff    exact selected-index diff between two snapshots (rank 0) — the
          attribution diagnostic for landed changes; never a pass/fail gate.

The dump files come from
``sglang.srt.layers.attention.double_sparsity.selection_capture`` (one
``rank{R}_step{S:05d}.pt`` per rank per decode step, fields: step, bs,
seq_lens, indices ``[num_layers, bs, max_top_k]`` int32, lengths
``[num_layers, bs]`` int32).
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import shutil
import sys
import time

import requests
import torch

_WORDS = (
    "alpha", "bridge", "carbon", "delta", "ember", "forest", "granite",
    "harbor", "island", "juniper", "kernel", "lantern", "meadow", "nickel",
    "orchard", "pebble", "quartz", "ridge", "summit", "timber", "upland",
    "valley", "willow", "zephyr",
)

# Fixed workload: word counts chosen so the tokenized prompts straddle the
# top_k=2048 selection budget — one dense (< budget) and three genuinely
# sparse (> budget) requests. Seeds fix the text bit-exactly.
_DEFAULT_WORD_COUNTS = (400, 2200, 4600, 9400)
_PROMPT_SEED_BASE = 11


def build_prompt(num_words: int, seed: int) -> str:
    rng = random.Random(seed)
    words = [_WORDS[rng.randrange(len(_WORDS))] for _ in range(num_words)]
    return "Recite the following field notes verbatim:\n" + " ".join(words)


def _default_dump_dir() -> str:
    from sglang.srt.layers.attention.double_sparsity.selection_capture import (
        selection_capture_dir,
    )

    return selection_capture_dir()


def _wipe_dump_dir(dump_dir: str) -> int:
    n = 0
    for f in glob.glob(os.path.join(dump_dir, "rank*_step*.pt")):
        os.remove(f)
        n += 1
    return n


def _generate(base: str, prompt: str, decode_steps: int) -> dict:
    # The first output token comes from the prefill forward; DS selection runs
    # only in decode, so request decode_steps + 1 tokens.
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
    r.raise_for_status()
    return r.json()


def cmd_run(args: argparse.Namespace) -> int:
    dump_dir = args.dump_dir or _default_dump_dir()
    os.makedirs(args.out, exist_ok=True)
    word_counts = [int(w) for w in args.words]
    prompts = [
        build_prompt(w, _PROMPT_SEED_BASE + i) for i, w in enumerate(word_counts)
    ]
    meta = {
        "base_url": args.base_url,
        "decode_steps": int(args.decode_steps),
        "word_counts": word_counts,
        "prompt_sha256": [hashlib.sha256(p.encode()).hexdigest() for p in prompts],
        "passes": [],
    }
    for pass_idx in range(int(args.repeat)):
        pass_dir = os.path.join(args.out, f"pass{pass_idx}")
        os.makedirs(pass_dir, exist_ok=True)
        wiped = _wipe_dump_dir(dump_dir)
        print(f"[selcap-run] pass {pass_idx}: wiped {wiped} stale dump file(s)")
        pass_meta = {"requests": []}
        for i, prompt in enumerate(prompts):
            t0 = time.time()
            resp = _generate(args.base_url, prompt, args.decode_steps)
            mi = resp.get("meta_info", {})
            pass_meta["requests"].append(
                {
                    "prompt_words": word_counts[i],
                    "prompt_tokens": mi.get("prompt_tokens"),
                    "completion_tokens": mi.get("completion_tokens"),
                    "latency_s": round(time.time() - t0, 2),
                }
            )
            print(
                f"[selcap-run]   prompt {i}: {word_counts[i]}w -> "
                f"{mi.get('prompt_tokens')} tokens ({time.time()-t0:.1f}s)"
            )
        time.sleep(1.0)  # let the final post-forward dumps land on disk
        files = sorted(glob.glob(os.path.join(dump_dir, "rank*_step*.pt")))
        for f in files:
            shutil.move(f, os.path.join(pass_dir, os.path.basename(f)))
        pass_meta["dump_files"] = len(files)
        meta["passes"].append(pass_meta)
        print(f"[selcap-run] pass {pass_idx}: snapshot {len(files)} files -> {pass_dir}")
    with open(os.path.join(args.out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[selcap-run] meta -> {os.path.join(args.out, 'meta.json')}")
    return 0


def _load_pass(pass_dir: str):
    """Return {step: {rank: record}} for one snapshot directory."""
    out: dict = {}
    for f in sorted(glob.glob(os.path.join(pass_dir, "rank*_step*.pt"))):
        name = os.path.basename(f)
        rank = int(name.split("_")[0][len("rank"):])
        rec = torch.load(f, map_location="cpu", weights_only=True)
        out.setdefault(int(rec["step"]), {})[rank] = rec
    return out


def _sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()


def _contract_errors(rec: dict, max_top_k: int) -> list:
    """Output-contract violations for one record (rank 0)."""
    errs = []
    idx, lens = rec["indices"], rec["lengths"]
    num_layers, bs, k = idx.shape
    if k != max_top_k:
        errs.append(f"index width {k} != max_top_k {max_top_k}")
    seq_lens = rec.get("seq_lens") or [None] * bs
    for layer in range(num_layers):
        for b in range(bs):
            v = int(lens[layer, b])
            row = idx[layer, b]
            if not (0 <= v <= k):
                errs.append(f"layer {layer} row {b}: valid_length {v} out of [0,{k}]")
                continue
            head, tail = row[:v], row[v:]
            if v > 1 and not bool((head[1:] > head[:-1]).all()):
                errs.append(f"layer {layer} row {b}: prefix not strictly ascending")
            if v and (int(head.min()) < 0):
                errs.append(f"layer {layer} row {b}: negative index in valid prefix")
            sl = seq_lens[b]
            if v and sl is not None and int(head.max()) >= int(sl):
                errs.append(
                    f"layer {layer} row {b}: index {int(head.max())} >= seq_len {sl}"
                )
            if tail.numel() and not bool((tail == -1).all()):
                errs.append(f"layer {layer} row {b}: padding tail not all -1")
            if sl is not None and v > int(sl):
                errs.append(f"layer {layer} row {b}: valid_length {v} > seq_len {sl}")
    return errs


def cmd_verify(args: argparse.Namespace) -> int:
    pass_dirs = sorted(glob.glob(os.path.join(args.run_dir, "pass*")))
    if not pass_dirs:
        pass_dirs = [args.run_dir]
    problems = []
    digest = {"run_dir": args.run_dir, "ranks": int(args.ranks), "passes": []}
    pass_step_shas = []
    for pass_dir in pass_dirs:
        by_step = _load_pass(pass_dir)
        if not by_step:
            problems.append(f"{pass_dir}: no dump files")
            continue
        steps = sorted(by_step)
        if args.expected_steps and len(steps) != int(args.expected_steps):
            problems.append(
                f"{pass_dir}: {len(steps)} steps != expected {args.expected_steps}"
            )
        pass_digest = {"dir": pass_dir, "steps": []}
        shas = []
        for step in steps:
            ranks = by_step[step]
            if len(ranks) != int(args.ranks):
                problems.append(
                    f"{pass_dir} step {step}: {sorted(ranks)} ranks present, "
                    f"expected {args.ranks}"
                )
                continue
            r0 = ranks[0]
            for rank in sorted(ranks):
                rec = ranks[rank]
                if not torch.equal(rec["indices"], r0["indices"]) or not torch.equal(
                    rec["lengths"], r0["lengths"]
                ):
                    problems.append(
                        f"{pass_dir} step {step}: rank {rank} selection != rank 0 "
                        "(cross-rank bit-identity violated)"
                    )
            errs = _contract_errors(r0, int(args.max_top_k))
            if errs:
                problems.extend(f"{pass_dir} step {step}: {e}" for e in errs[:10])
            sha_i, sha_l = _sha(r0["indices"]), _sha(r0["lengths"])
            shas.append((sha_i, sha_l))
            pass_digest["steps"].append(
                {
                    "step": step,
                    "bs": r0["bs"],
                    "seq_lens": r0.get("seq_lens"),
                    "num_layers": list(r0["indices"].shape)[0],
                    "indices_sha256": sha_i,
                    "lengths_sha256": sha_l,
                }
            )
        digest["passes"].append(pass_digest)
        pass_step_shas.append(shas)
    # Run-to-run identity across passes (aligned by step order within the pass).
    if len(pass_step_shas) > 1:
        ref = pass_step_shas[0]
        for i, shas in enumerate(pass_step_shas[1:], start=1):
            if shas != ref:
                problems.append(
                    f"pass {i} selection differs from pass 0 "
                    "(run-to-run determinism violated)"
                )
    digest["verdict"] = "PASS" if not problems else "FAIL"
    digest["problems"] = problems
    if args.digest:
        os.makedirs(os.path.dirname(args.digest) or ".", exist_ok=True)
        with open(args.digest, "w") as fh:
            json.dump(digest, fh, indent=2)
        print(f"[selcap-verify] digest -> {args.digest}")
    if problems:
        print(f"[selcap-verify] FAIL ({len(problems)} problem(s)):")
        for p in problems[:30]:
            print(f"  - {p}")
        return 1
    n_steps = sum(len(p["steps"]) for p in digest["passes"])
    print(
        f"[selcap-verify] PASS: {len(digest['passes'])} pass(es), {n_steps} steps, "
        f"{args.ranks} ranks bit-identical, contract clean"
    )
    return 0


def cmd_diff(args: argparse.Namespace) -> int:
    a = _load_pass(args.a)
    b = _load_pass(args.b)
    steps_a, steps_b = sorted(a), sorted(b)
    if len(steps_a) != len(steps_b):
        print(f"[selcap-diff] FAIL: step counts differ ({len(steps_a)} vs {len(steps_b)})")
        return 1
    total_rows = 0
    diff_rows = 0
    per_step = []
    for sa, sb in zip(steps_a, steps_b):
        ra = a[sa][min(a[sa])]
        rb = b[sb][min(b[sb])]
        if ra["indices"].shape != rb["indices"].shape:
            print(
                f"[selcap-diff] FAIL: step {sa}/{sb} shapes differ "
                f"{list(ra['indices'].shape)} vs {list(rb['indices'].shape)}"
            )
            return 1
        rows_equal = (ra["indices"] == rb["indices"]).all(dim=-1) & (
            ra["lengths"] == rb["lengths"]
        )
        n = rows_equal.numel()
        d = int((~rows_equal).sum())
        total_rows += n
        diff_rows += d
        entry = {"step_a": sa, "step_b": sb, "rows": n, "rows_differing": d}
        if d:
            # Per differing row: how many of the top_k positions moved.
            diff_pos = (
                (ra["indices"] != rb["indices"]).sum(dim=-1)[~rows_equal]
            )
            entry["positions_moved_max"] = int(diff_pos.max())
            entry["positions_moved_mean"] = round(float(diff_pos.float().mean()), 2)
        per_step.append(entry)
    out = {
        "a": args.a,
        "b": args.b,
        "total_layer_rows": total_rows,
        "layer_rows_differing": diff_rows,
        "fraction_differing": round(diff_rows / max(1, total_rows), 6),
        "per_step": per_step,
    }
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"[selcap-diff] report -> {args.out}")
    print(
        f"[selcap-diff] {diff_rows}/{total_rows} (layer,row) selections differ "
        f"({out['fraction_differing']*100:.4f}%) — diagnostic only"
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="drive the fixed workload and snapshot dumps")
    p.add_argument("--base-url", default=os.environ.get("DS_BASE_URL", "http://127.0.0.1:30000"))
    p.add_argument("--out", required=True, help="snapshot output directory")
    p.add_argument("--dump-dir", default=None, help="override the worker dump dir")
    p.add_argument("--decode-steps", type=int, default=8)
    p.add_argument("--repeat", type=int, default=2)
    p.add_argument("--words", type=int, nargs="+", default=list(_DEFAULT_WORD_COUNTS))
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("verify", help="fail-closed checks + digest emission")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--ranks", type=int, default=8)
    p.add_argument("--expected-steps", type=int, default=None)
    p.add_argument("--max-top-k", type=int, default=2048)
    p.add_argument("--digest", default=None, help="write the digest JSON here")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("diff", help="exact index diff between two snapshots (diagnostic)")
    p.add_argument("--a", required=True, help="baseline snapshot (pass dir)")
    p.add_argument("--b", required=True, help="candidate snapshot (pass dir)")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_diff)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
