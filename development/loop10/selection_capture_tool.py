"""Loop-10 selection-capture tool — bs-1 AND op-point gates, diff as HARD gate.

Extends the loop-9 tool (development/loop9/selection_capture_tool.py) with the
op-point harness and gate-grade comparisons. Subcommands:

  run         loop-9 bs-1 fixed sequential workload (identical prompts/seeds,
              so digests stay comparable to the frozen loop-9 R1 digest).
  run-op      op-point workload: N concurrent requests (default 29) with
              ~4k-token prompts decoded lock-step under CUDA-graph replay.
              Requests are submitted in a fixed order with a deterministic
              stagger so the decode batch row order is reproducible.
  verify      loop-9 fail-closed checks (cross-rank bit-identity, output
              contract, run-to-run identity) PLUS bucket-identity recording
              (raw_bs, padded_bs, selector_width, graph_key, replay_path,
              max_real_seq_len) and optional hard requirements:
              --require-raw-bs N, --require-replay, --require-padded-bs N.
              The digest JSON always lands before a failing exit.
  diff        exact selected-index diff between two snapshot pass dirs.
              --fail-on-diff promotes it to a hard gate (exit 1 on any
              differing row) — the loop-10 exact-change gate mode.
  diff-digest digest-vs-digest comparison (per-step indices/lengths SHA256 +
              shapes). HARD by construction: any SHA or structure mismatch
              exits 1. Identity-field changes also fail unless explicitly
              declared via --allow-identity-change FIELD (fail-closed).

Dump records come from
``sglang.srt.layers.attention.double_sparsity.selection_capture`` (one
``rank{R}_step{S:05d}.pt`` per rank per decode step). Loop-10 records carry
the bucket identity fields; loop-9-era records lack them, and comparisons
against loop-9 digests fall back to SHA/shape equality only.
"""

from __future__ import annotations

import argparse
import concurrent.futures
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

# bs-1 fixed workload — MUST stay bit-identical to the loop-9 tool so the
# fresh bs-1 digest is comparable to the frozen loop-9 R1 digest.
_DEFAULT_WORD_COUNTS = (400, 2200, 4600, 9400)
_PROMPT_SEED_BASE = 11

# Op-point workload: seeds disjoint from the bs-1 workload; ~3070 words
# tokenize to ~4090 tokens with this tokenizer (loop-9 meta: 1.33 tok/word),
# mirroring the frozen Case-1 ISL-4096 regime.
_OP_PROMPT_SEED_BASE = 211
_OP_DEFAULT_WORDS = 3070

_IDENTITY_FIELDS = (
    "raw_bs",
    "padded_bs",
    "selector_width",
    "graph_key",
    "replay_path",
)


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
        timeout=1800,
    )
    r.raise_for_status()
    return r.json()


def _snapshot_pass(dump_dir: str, pass_dir: str) -> int:
    time.sleep(1.0)  # let the final post-forward dumps land on disk
    files = sorted(glob.glob(os.path.join(dump_dir, "rank*_step*.pt")))
    for f in files:
        shutil.move(f, os.path.join(pass_dir, os.path.basename(f)))
    return len(files)


def cmd_run(args: argparse.Namespace) -> int:
    dump_dir = args.dump_dir or _default_dump_dir()
    os.makedirs(args.out, exist_ok=True)
    word_counts = [int(w) for w in args.words]
    prompts = [
        build_prompt(w, _PROMPT_SEED_BASE + i) for i, w in enumerate(word_counts)
    ]
    meta = {
        "mode": "bs1-sequential",
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
        n = _snapshot_pass(dump_dir, pass_dir)
        pass_meta["dump_files"] = n
        meta["passes"].append(pass_meta)
        print(f"[selcap-run] pass {pass_idx}: snapshot {n} files -> {pass_dir}")
    with open(os.path.join(args.out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[selcap-run] meta -> {os.path.join(args.out, 'meta.json')}")
    return 0


def cmd_run_op(args: argparse.Namespace) -> int:
    dump_dir = args.dump_dir or _default_dump_dir()
    os.makedirs(args.out, exist_ok=True)
    conc = int(args.concurrency)
    words = int(args.op_words)
    prompts = [build_prompt(words, _OP_PROMPT_SEED_BASE + i) for i in range(conc)]
    stagger_s = float(args.stagger_ms) / 1000.0
    meta = {
        "mode": "op-point-concurrent",
        "base_url": args.base_url,
        "concurrency": conc,
        "decode_steps": int(args.decode_steps),
        "words_per_prompt": words,
        "stagger_ms": float(args.stagger_ms),
        "prompt_sha256": [hashlib.sha256(p.encode()).hexdigest() for p in prompts],
        "passes": [],
    }
    for pass_idx in range(int(args.repeat)):
        pass_dir = os.path.join(args.out, f"pass{pass_idx}")
        os.makedirs(pass_dir, exist_ok=True)
        wiped = _wipe_dump_dir(dump_dir)
        print(f"[selcap-run-op] pass {pass_idx}: wiped {wiped} stale dump file(s)")
        t0 = time.time()
        results: list = [None] * conc
        with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as ex:
            futures = {}
            for i, prompt in enumerate(prompts):
                futures[ex.submit(_generate, args.base_url, prompt, args.decode_steps)] = i
                # Deterministic admission order: fixed submission order with a
                # stagger long enough to dominate HTTP/scheduler jitter.
                time.sleep(stagger_s)
            errors = []
            for fut in concurrent.futures.as_completed(futures):
                i = futures[fut]
                try:
                    results[i] = fut.result()
                except Exception as exc:  # record, then fail after snapshot
                    errors.append(f"request {i}: {exc}")
        elapsed = time.time() - t0
        pass_meta = {
            "requests": [
                {
                    "prompt_tokens": (r or {}).get("meta_info", {}).get("prompt_tokens"),
                    "completion_tokens": (r or {})
                    .get("meta_info", {})
                    .get("completion_tokens"),
                }
                for r in results
            ],
            "errors": errors,
            "elapsed_s": round(elapsed, 2),
        }
        n = _snapshot_pass(dump_dir, pass_dir)
        pass_meta["dump_files"] = n
        meta["passes"].append(pass_meta)
        print(
            f"[selcap-run-op] pass {pass_idx}: {conc} requests in {elapsed:.1f}s, "
            f"snapshot {n} files -> {pass_dir}"
        )
        if errors:
            # Snapshot + meta land before the failing exit so the evidence of
            # a rejected/failed request is durable.
            with open(os.path.join(args.out, "meta.json"), "w") as fh:
                json.dump(meta, fh, indent=2)
            print(f"[selcap-run-op] FAIL: {len(errors)} request error(s):")
            for e in errors[:10]:
                print(f"  - {e}")
            return 1
    with open(os.path.join(args.out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[selcap-run-op] meta -> {os.path.join(args.out, 'meta.json')}")
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


def _step_identity(rec: dict) -> dict:
    """Bucket-identity fields of one record (None-filled for loop-9 records)."""
    ident = {f: rec.get(f) for f in _IDENTITY_FIELDS}
    ident["max_real_seq_len"] = rec.get("max_real_seq_len")
    return ident


def cmd_verify(args: argparse.Namespace) -> int:
    pass_dirs = sorted(glob.glob(os.path.join(args.run_dir, "pass*")))
    if not pass_dirs:
        pass_dirs = [args.run_dir]
    problems = []
    digest = {"run_dir": args.run_dir, "ranks": int(args.ranks), "passes": []}
    requirements = {
        "require_raw_bs": args.require_raw_bs,
        "require_replay": bool(args.require_replay),
        "require_padded_bs": args.require_padded_bs,
    }
    digest["requirements"] = requirements
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
        if args.min_steps and len(steps) < int(args.min_steps):
            problems.append(
                f"{pass_dir}: {len(steps)} steps < required minimum {args.min_steps}"
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
            ident = _step_identity(r0)
            if args.require_raw_bs is not None and ident["raw_bs"] != int(
                args.require_raw_bs
            ):
                problems.append(
                    f"{pass_dir} step {step}: raw_bs {ident['raw_bs']} != "
                    f"required {args.require_raw_bs}"
                )
            if args.require_replay and not ident["replay_path"]:
                problems.append(
                    f"{pass_dir} step {step}: replay_path is "
                    f"{ident['replay_path']} (graph replay required; eager or "
                    "untagged path detected)"
                )
            if args.require_padded_bs is not None and ident["padded_bs"] != int(
                args.require_padded_bs
            ):
                problems.append(
                    f"{pass_dir} step {step}: padded_bs {ident['padded_bs']} != "
                    f"required {args.require_padded_bs}"
                )
            sha_i, sha_l = _sha(r0["indices"]), _sha(r0["lengths"])
            shas.append((sha_i, sha_l))
            entry = {
                "step": step,
                "bs": r0["bs"],
                "seq_lens": r0.get("seq_lens"),
                "num_layers": list(r0["indices"].shape)[0],
                "indices_sha256": sha_i,
                "lengths_sha256": sha_l,
            }
            entry.update(ident)
            pass_digest["steps"].append(entry)
        # Uniform identity across the pass (one captured variant serving the
        # whole steady-state workload) — recorded always, enforced when any
        # hard requirement was requested.
        idents = [
            {f: s.get(f) for f in _IDENTITY_FIELDS} for s in pass_digest["steps"]
        ]
        uniform = all(i == idents[0] for i in idents) if idents else False
        pass_digest["identity_uniform"] = uniform
        if idents and (
            args.require_replay
            or args.require_raw_bs is not None
            or args.require_padded_bs is not None
        ):
            if not uniform:
                problems.append(
                    f"{pass_dir}: bucket identity not uniform across steps: "
                    f"{[dict(i) for i in idents[:4]]}..."
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
        "fail_on_diff": bool(args.fail_on_diff),
        "per_step": per_step,
    }
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"[selcap-diff] report -> {args.out}")
    if args.fail_on_diff and diff_rows:
        print(
            f"[selcap-diff] FAIL: {diff_rows}/{total_rows} (layer,row) selections "
            "differ — exact-change gate violated"
        )
        return 1
    print(
        f"[selcap-diff] {diff_rows}/{total_rows} (layer,row) selections differ "
        f"({out['fraction_differing']*100:.4f}%)"
        + ("" if args.fail_on_diff else " — diagnostic only")
    )
    return 0


def _digest_steps(digest: dict) -> list:
    """Flatten a digest into [(pass_idx, step_entry)] in pass/step order."""
    out = []
    for pi, p in enumerate(digest.get("passes", [])):
        for s in p.get("steps", []):
            out.append((pi, s))
    return out


def cmd_diff_digest(args: argparse.Namespace) -> int:
    with open(args.a) as fh:
        da = json.load(fh)
    with open(args.b) as fh:
        db = json.load(fh)
    allow = set(args.allow_identity_change or [])
    sa, sb = _digest_steps(da), _digest_steps(db)
    problems = []
    if len(sa) != len(sb):
        problems.append(f"step counts differ: {len(sa)} vs {len(sb)}")
    n_sha_mismatch = 0
    identity_changes: dict = {}
    for (pa, ea), (pb, eb) in zip(sa, sb):
        tag = f"pass{pa}/step{ea.get('step')} vs pass{pb}/step{eb.get('step')}"
        for key in ("bs", "num_layers"):
            if ea.get(key) != eb.get(key):
                problems.append(f"{tag}: {key} differs ({ea.get(key)} vs {eb.get(key)})")
        for key in ("indices_sha256", "lengths_sha256"):
            if ea.get(key) != eb.get(key):
                n_sha_mismatch += 1
                problems.append(f"{tag}: {key} mismatch")
        for f in (*_IDENTITY_FIELDS, "max_real_seq_len"):
            va, vb = ea.get(f), eb.get(f)
            if va is None or vb is None:
                continue  # loop-9-era digest without identity fields
            if va != vb:
                identity_changes.setdefault(f, set()).add((str(va), str(vb)))
                if f not in allow:
                    problems.append(
                        f"{tag}: identity field '{f}' changed ({va} -> {vb}) "
                        "without --allow-identity-change declaration"
                    )
    out = {
        "a": args.a,
        "b": args.b,
        "steps_compared": min(len(sa), len(sb)),
        "sha_mismatches": n_sha_mismatch,
        "identity_changes": {
            f: sorted(list(v)) for f, v in identity_changes.items()
        },
        "allowed_identity_changes": sorted(allow),
        "verdict": "PASS" if not problems else "FAIL",
        "problems": problems,
    }
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"[selcap-diff-digest] report -> {args.out}")
    if problems:
        print(f"[selcap-diff-digest] FAIL ({len(problems)} problem(s)):")
        for p in problems[:30]:
            print(f"  - {p}")
        return 1
    print(
        f"[selcap-diff-digest] PASS: {out['steps_compared']} steps bit-identical"
        + (
            f" (declared identity changes: {sorted(allow)})"
            if allow and identity_changes
            else ""
        )
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="bs-1 fixed sequential workload (loop-9 identical)")
    p.add_argument("--base-url", default=os.environ.get("DS_BASE_URL", "http://127.0.0.1:30000"))
    p.add_argument("--out", required=True, help="snapshot output directory")
    p.add_argument("--dump-dir", default=None, help="override the worker dump dir")
    p.add_argument("--decode-steps", type=int, default=8)
    p.add_argument("--repeat", type=int, default=2)
    p.add_argument("--words", type=int, nargs="+", default=list(_DEFAULT_WORD_COUNTS))
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("run-op", help="op-point concurrent workload (graph replay)")
    p.add_argument("--base-url", default=os.environ.get("DS_BASE_URL", "http://127.0.0.1:30000"))
    p.add_argument("--out", required=True, help="snapshot output directory")
    p.add_argument("--dump-dir", default=None, help="override the worker dump dir")
    p.add_argument("--concurrency", type=int, default=29)
    p.add_argument("--decode-steps", type=int, default=12)
    p.add_argument("--repeat", type=int, default=2)
    p.add_argument("--op-words", type=int, default=_OP_DEFAULT_WORDS)
    p.add_argument("--stagger-ms", type=float, default=100.0)
    p.set_defaults(func=cmd_run_op)

    p = sub.add_parser("verify", help="fail-closed checks + identity-tagged digest")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--ranks", type=int, default=8)
    p.add_argument("--expected-steps", type=int, default=None)
    p.add_argument("--min-steps", type=int, default=None)
    p.add_argument("--max-top-k", type=int, default=2048)
    p.add_argument("--digest", default=None, help="write the digest JSON here")
    p.add_argument("--require-raw-bs", type=int, default=None,
                   help="hard-require this raw batch size on every step")
    p.add_argument("--require-replay", action="store_true",
                   help="hard-require the graph-replay path on every step")
    p.add_argument("--require-padded-bs", type=int, default=None,
                   help="hard-require this padded batch size on every step")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("diff", help="exact index diff between two snapshot pass dirs")
    p.add_argument("--a", required=True, help="baseline snapshot (pass dir)")
    p.add_argument("--b", required=True, help="candidate snapshot (pass dir)")
    p.add_argument("--out", default=None)
    p.add_argument("--fail-on-diff", action="store_true",
                   help="exit 1 on any differing row (exact-change hard gate)")
    p.set_defaults(func=cmd_diff)

    p = sub.add_parser(
        "diff-digest",
        help="digest-vs-digest SHA comparison (hard gate; identity fail-closed)",
    )
    p.add_argument("--a", required=True, help="baseline digest JSON")
    p.add_argument("--b", required=True, help="candidate digest JSON")
    p.add_argument("--out", default=None)
    p.add_argument(
        "--allow-identity-change",
        nargs="+",
        default=None,
        help="identity fields whose change is explicitly declared "
        f"(subset of {list(_IDENTITY_FIELDS) + ['max_real_seq_len']})",
    )
    p.set_defaults(func=cmd_diff_digest)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
