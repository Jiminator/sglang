"""Offline DS-vs-DSA accuracy gate (MMLU + NIAH) — sequential collect, then compare.

GLM-5.1 cannot run two TP=8 servers at once on 8 GPUs (weights > 140 GB/rank), and
cannot run at TP=4 (weights ~2x exceed a single H200), so the paired in-process
harness (`test/manual/test_double_sparsity_v32.py`, which needs DS_BASE_URL AND
DSA_BASE_URL live together) cannot drive the DS-vs-DSA accuracy comparison. This
module is the sequential path:

    AC12_MODE=collect AC12_SIDE=dsa AC12_BASE_URL=http://127.0.0.1:30000 \
        python development/loop8/accuracy_gate.py        # boot DSA-native, score, write artifact
    # shut DSA down, boot DS with the 256 mask, then:
    AC12_MODE=collect AC12_SIDE=ds  AC12_BASE_URL=http://127.0.0.1:30000 \
        python development/loop8/accuracy_gate.py
    AC12_MODE=compare AC12_DSA_ARTIFACT=... AC12_DS_ARTIFACT=... \
        python development/loop8/accuracy_gate.py        # offline, no server

`collect` reuses the TUNED scoring from `test_double_sparsity_v32.py` (the
`_parse_mmlu_letter` two-tier parser, the deterministic NIAH prompt-gen / recall
scorer, and the MMLU 5-shot prompt + example loader) and the harness within-budget
`(1024,1536)` / beyond-budget `(4096,16384,65536)` length sets, so a side's score
matches the paired gate. It records per-test served counts, first errors, and the
per-NIAH-length max `usage.prompt_tokens`, plus the server op-point from
`/get_server_info`.

`compare` is pure (operates on the two artifact dicts) and FAILS CLOSED: it requires
both sides to have actually served every mandatory request (MMLU `served==total`,
no errors; within-budget NIAH `served==num_prompts`, no first_error, and
`max_prompt_tokens <= index_topk` so the within-budget premise holds), and the
stable server op-point to match except the intended DS differences. A gate that
"passes" when every request failed is worse than no gate. Thresholds mirror the
paired gate: MMLU DS within 1.0 pp of DSA (mandatory); within-budget NIAH DS within
5.0 pp of DSA (mandatory); beyond-budget NIAH is characterization-only.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import urllib.request
from typing import Any, Dict, List, Optional

SCHEMA = "ac12_accuracy_side_v3"
MMLU_TOLERANCE_PP = 1.0
NIAH_WITHIN_BUDGET_TOLERANCE_PP = 5.0

# The stable server op-point both sides must share is NOT a hand whitelist (a whitelist
# silently drops any launch flag not listed, and a `da.get(k) != sa.get(k)` compare
# treats `None == None` as agreement when both artifacts omit a field). We reuse the
# EXACT projection `benchmark_compare.py` derives from `dataclasses.fields(ServerArgs)`
# — the full stable launch-arg set, minus the DS-only knobs (legitimately different) and
# the recorded-not-matched keys (`random_seed`, `mem_fraction_static`) — and require
# every locked Option-B field present + non-null on both sides. Sharing the source means
# a new sglang launch flag is auto-protected and the two gates cannot drift. See
# `_bench_compare()`.

# The exact production landing mask the DS column must have served (AC-3 artifact). The
# gate proves BOTH the path and the recorded content hash so a wrong/stale mask cannot
# pass. Overridable via AC12_EXPECTED_DS_MASK_PATH / AC12_EXPECTED_DS_MASK_SHA256.
EXPECTED_DS_MASK_PATH = "/models/glm51-fp8-channel-mask-s256.safetensors"
EXPECTED_DS_MASK_SHA256 = (
    "35155ac46ad79fa82e531138434ff35708e2d8c2932889323a21a455342a9b00"
)


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


# ----- compare (pure; offline; fail-closed) ------------------------------


class GateError(ValueError):
    """Raised when the two artifacts are incomparable or a mandatory gate fails."""


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise GateError(msg)


def _pct(hits: int, total: int) -> float:
    return 100.0 * hits / total if total else 0.0


_BENCH_COMPARE = None


def _bench_compare():
    """Load development/benchmark_compare.py once and reuse its stable-launch-arg
    projection so the accuracy gate and the SLO comparator cannot drift."""
    global _BENCH_COMPARE
    if _BENCH_COMPARE is None:
        import importlib.util
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "benchmark_compare.py"))
        spec = importlib.util.spec_from_file_location("_ac12_bench_compare", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_ac12_bench_compare"] = mod
        spec.loader.exec_module(mod)
        _BENCH_COMPARE = mod
    return _BENCH_COMPARE


# The accuracy gate permits ONLY the DS knobs and the DS-vs-DSA memory-reservation
# difference to differ — NOT random_seed. Unlike the throughput sweep (which records
# seed but cannot match it across sequential trials), accuracy is greedy/deterministic
# and this loop mandates seed parity for the final evidence, so a seed difference must
# fail closed. We still reuse benchmark_compare's ServerArgs-derived field universe +
# DS-only set so a new launch flag is auto-protected and the two gates cannot drift.
_ACCURACY_RECORDED_NOT_MATCHED = frozenset({"mem_fraction_static"})


def _comparable_op_point(server_info: Dict[str, Any]) -> Dict[str, Any]:
    """Project a /get_server_info dict onto the stable ServerArgs launch fields that
    must match across DSA and DS — the full ServerArgs set minus the DS-only knobs and
    the memory-reservation difference. Dynamic telemetry is dropped because it is not a
    ServerArgs field."""
    bc = _bench_compare()
    excluded = bc._DS_ONLY_SERVER_ARG_KEYS | _ACCURACY_RECORDED_NOT_MATCHED
    keys = (server_info.keys() & bc._AC11_STABLE_LAUNCH_ARG_KEYS) - excluded
    return {k: server_info[k] for k in keys}


def _check_op_point(dsa: Dict[str, Any], ds: Dict[str, Any], *,
                    expected_mask_path: str, expected_mask_sha256: Optional[str]) -> None:
    """Fail closed unless the full stable server op-point matches (only DS knobs may
    differ) AND the DS column proves it served the expected production mask."""
    da, sa = dsa.get("server_info") or {}, ds.get("server_info") or {}
    _require(bool(da) and bool(sa), "missing server_info in one or both artifacts")
    _require(da.get("enable_double_sparsity") is False, "dsa side has enable_double_sparsity != False")
    _require(sa.get("enable_double_sparsity") is True, "ds side has enable_double_sparsity != True")

    dproj, sproj = _comparable_op_point(da), _comparable_op_point(sa)
    # Every locked Option-B launch field must be present + non-null on BOTH sides, so a
    # missing field cannot pass as `None == None` (reuse benchmark_compare's locked set).
    locked = _bench_compare()._AC11_OPTION_B_LOCKED_FIELDS
    for side, proj in (("dsa", dproj), ("ds", sproj)):
        missing = sorted(f for f in locked if proj.get(f) is None)
        _require(not missing,
                 f"{side} server op-point missing/null locked field(s) {missing} "
                 "(cannot prove the operating point — fail closed)")
    # Compare the UNION of stable keys so a field present on one side only, or differing
    # (including launch flags outside any hand whitelist, e.g. dtype), fails closed.
    mismatches = [f"{k}: dsa={dproj.get(k)!r} ds={sproj.get(k)!r}"
                  for k in (dproj.keys() | sproj.keys()) if dproj.get(k) != sproj.get(k)]
    _require(not mismatches, "server op-point mismatch (only DS knobs may differ): "
             + "; ".join(sorted(mismatches)))

    # Provenance: the DS column must prove it served the EXACT expected mask — both the
    # path and the recorded content hash (a non-empty-but-wrong path must fail closed).
    ds_mask = _ds_mask_path(sa)
    _require(ds_mask == expected_mask_path,
             f"ds double_sparsity_config.channel_mask_path={ds_mask!r} != expected "
             f"{expected_mask_path!r} (cannot prove the 256-sample GLM mask — fail closed)")
    if expected_mask_sha256:
        ds_sha = sa.get("_ds_channel_mask_sha256")
        _require(ds_sha == expected_mask_sha256,
                 f"ds channel-mask content sha256={ds_sha!r} != expected "
                 f"{expected_mask_sha256!r} (wrong/unrecorded mask — fail closed)")


def _check_mmlu_served(side: str, mm: Dict[str, Any]) -> None:
    total = int(mm.get("total", 0))
    _require(total > 0, f"{side} MMLU total is zero (fail closed)")
    _require(int(mm.get("served", -1)) == total,
             f"{side} MMLU served={mm.get('served')} != total={total} (requests failed — fail closed)")
    _require(not mm.get("first_error"),
             f"{side} MMLU had a request error: {mm.get('first_error')!r} (fail closed)")


def _check_niah_within(side: str, e: Dict[str, Any], index_topk: int) -> None:
    L = e.get("length_words")
    n = int(e.get("num_prompts", 0))
    _require(n > 0, f"{side} NIAH L={L} num_prompts zero (fail closed)")
    _require(int(e.get("served", -1)) == n,
             f"{side} NIAH L={L} served={e.get('served')} != num_prompts={n} (fail closed)")
    _require(not e.get("first_error"),
             f"{side} NIAH L={L} request error: {e.get('first_error')!r} (fail closed)")
    _require(not e.get("usage_missing", True),
             f"{side} NIAH L={L} usage.prompt_tokens missing — within-budget premise unprovable (fail closed)")
    mpt = e.get("max_prompt_tokens")
    _require(isinstance(mpt, int) and mpt <= index_topk,
             f"{side} NIAH L={L} max_prompt_tokens={mpt} not <= index_topk={index_topk} "
             "(not within budget — fail closed)")


def compare(dsa: Dict[str, Any], ds: Dict[str, Any], *,
            expected_ds_mask_path: str = EXPECTED_DS_MASK_PATH,
            expected_ds_mask_sha256: Optional[str] = EXPECTED_DS_MASK_SHA256) -> Dict[str, Any]:
    """Compare a DSA-native and a DS accuracy artifact. Fail closed.

    Validates schema, the full stable op-point (only DS knobs differ; locked fields
    required present), the exact expected DS mask (path + content sha), the same prompt
    set (run_id + per-test hashes + index_topk + length sets), and that both sides
    actually served every mandatory request, before applying the thresholds.
    """
    for name, art in (("dsa", dsa), ("ds", ds)):
        _require(isinstance(art, dict), f"{name} artifact is not a dict")
        _require(art.get("schema") == SCHEMA, f"{name} artifact schema != {SCHEMA!r}")
    _require(dsa.get("side") == "dsa", "dsa artifact side != 'dsa'")
    _require(ds.get("side") == "ds", "ds artifact side != 'ds'")
    _require(dsa.get("run_id") and dsa.get("run_id") == ds.get("run_id"),
             f"run_id mismatch: dsa={dsa.get('run_id')!r} ds={ds.get('run_id')!r}")
    index_topk = int(dsa.get("index_topk", -1))
    _require(index_topk == int(ds.get("index_topk", -2)),
             f"index_topk mismatch: dsa={dsa.get('index_topk')} ds={ds.get('index_topk')}")
    _check_op_point(dsa, ds, expected_mask_path=expected_ds_mask_path,
                    expected_mask_sha256=expected_ds_mask_sha256)

    # MMLU (mandatory): both sides fully served, same example set, DS within tol.
    dmm, smm = dsa.get("mmlu") or {}, ds.get("mmlu") or {}
    _require(bool(dmm) and bool(smm), "missing mmlu block")
    _require(dmm.get("prompt_set_hash") and dmm.get("prompt_set_hash") == smm.get("prompt_set_hash"),
             "MMLU prompt-set hash mismatch (different example set/prompts)")
    _check_mmlu_served("dsa", dmm)
    _check_mmlu_served("ds", smm)
    dsa_mmlu = _pct(int(dmm["hits"]), int(dmm["total"]))
    ds_mmlu = _pct(int(smm["hits"]), int(smm["total"]))
    mmlu_delta = dsa_mmlu - ds_mmlu
    mmlu_pass = abs(mmlu_delta) <= MMLU_TOLERANCE_PP

    # NIAH within-budget (mandatory): both fully served + within budget + within tol.
    d_in = {int(e["length_words"]): e for e in (dsa.get("niah_within_budget") or [])}
    s_in = {int(e["length_words"]): e for e in (ds.get("niah_within_budget") or [])}
    _require(d_in and set(d_in) == set(s_in),
             f"NIAH within-budget length set mismatch: dsa={sorted(d_in)} ds={sorted(s_in)}")
    niah_rows: List[Dict[str, Any]] = []
    niah_pass = True
    for L in sorted(d_in):
        de, se = d_in[L], s_in[L]
        _require(de.get("prompt_set_hash") and de.get("prompt_set_hash") == se.get("prompt_set_hash"),
                 f"NIAH prompt-set hash mismatch at length {L}")
        _check_niah_within("dsa", de, index_topk)
        _check_niah_within("ds", se, index_topk)
        dr, sr = _pct(int(de["hits"]), int(de["num_prompts"])), _pct(int(se["hits"]), int(se["num_prompts"]))
        delta = dr - sr
        ok = abs(delta) <= NIAH_WITHIN_BUDGET_TOLERANCE_PP
        niah_pass = niah_pass and ok
        niah_rows.append({"length_words": L, "dsa_recall_pct": round(dr, 2),
                          "ds_recall_pct": round(sr, 2), "delta_pp": round(delta, 2),
                          "within_tolerance": ok})

    # Beyond-budget NIAH: must be present + comparable; characterization only (never gates).
    d_be = {int(e["length_words"]): e for e in (dsa.get("niah_beyond_budget") or [])}
    s_be = {int(e["length_words"]): e for e in (ds.get("niah_beyond_budget") or [])}
    _require(d_be and set(d_be) == set(s_be),
             f"NIAH beyond-budget length set mismatch/empty: dsa={sorted(d_be)} ds={sorted(s_be)}")
    beyond_rows = []
    for L in sorted(d_be):
        de, se = d_be[L], s_be[L]
        _require(de.get("prompt_set_hash") and de.get("prompt_set_hash") == se.get("prompt_set_hash"),
                 f"NIAH beyond-budget prompt-set hash mismatch at length {L}")
        dn, sn = int(de.get("num_prompts", 0)), int(se.get("num_prompts", 0))
        _require(dn > 0 and dn == sn,
                 f"NIAH beyond-budget num_prompts mismatch/zero at length {L}: dsa={dn} ds={sn}")
        # Recall over num_prompts (unserved prompts count as misses — matches the
        # paired harness and the collect artifact's recall_pct). served/errors are
        # kept as separate admission telemetry, NOT the recall denominator.
        dr = _pct(int(de.get("hits", 0)), dn)
        sr = _pct(int(se.get("hits", 0)), sn)
        beyond_rows.append({"length_words": L, "num_prompts": dn,
                            "dsa_recall_pct": round(dr, 2), "ds_recall_pct": round(sr, 2),
                            "delta_pp": round(dr - sr, 2),
                            "dsa_served": de.get("served"), "ds_served": se.get("served")})

    return {
        "run_id": dsa["run_id"], "index_topk": index_topk,
        "ds_channel_mask_path": _ds_mask_path((ds.get("server_info") or {})),
        "ds_channel_mask_sha256": (ds.get("server_info") or {}).get("_ds_channel_mask_sha256"),
        "mmlu": {"dsa_pct": round(dsa_mmlu, 2), "ds_pct": round(ds_mmlu, 2),
                 "delta_pp": round(mmlu_delta, 2), "tolerance_pp": MMLU_TOLERANCE_PP,
                 "pass": mmlu_pass},
        "niah_within_budget": {"tolerance_pp": NIAH_WITHIN_BUDGET_TOLERANCE_PP,
                               "rows": niah_rows, "pass": niah_pass},
        "niah_beyond_budget_characterization": beyond_rows,
        "mandatory_pass": bool(mmlu_pass and niah_pass),
    }


# ----- collect (reuses the tuned harness scoring; needs one live server) --


def _load_harness():
    import importlib.util

    path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "test", "manual",
                     "test_double_sparsity_v32.py")
    )
    spec = importlib.util.spec_from_file_location("_ac12_harness", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_ac12_harness"] = mod
    spec.loader.exec_module(mod)
    return mod


def _default_mmlu_data_dir() -> str:
    # Mirror the paired harness default so collect is runnable without AC12_MMLU_DATA_DIR.
    return os.environ.get("AC12_MMLU_DATA_DIR") or os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "benchmark", "mmlu", "data"))


def _read_mask_content_sha256(path: str) -> Optional[str]:
    """Read `__metadata__.content_sha256` from a safetensors header without loading
    tensors (the value `save_channel_mask` stamps + the bind-time gate validated).
    Returns None if the file is unreadable/malformed — compare then fails closed."""
    try:
        with open(path, "rb") as fh:
            n = int.from_bytes(fh.read(8), "little")
            header = json.loads(fh.read(n).decode("utf-8"))
        meta = header.get("__metadata__") or {}
        return meta.get("content_sha256")
    except (OSError, ValueError, TypeError):
        return None


def _fetch_server_info(base_url: str) -> Dict[str, Any]:
    """Fetch the full stable ServerArgs launch projection from /get_server_info, plus
    the DS channel-mask content hash for provenance (fail closed)."""
    req = urllib.request.Request(base_url.rstrip("/") + "/get_server_info")
    with urllib.request.urlopen(req, timeout=30) as r:
        info = json.loads(r.read().decode())
    # Record the full stable ServerArgs launch projection (the same field universe
    # benchmark_compare.py protects) so compare can prove the whole op-point; dynamic
    # /get_server_info telemetry is dropped because it is not a ServerArgs field.
    stable = _bench_compare()._AC11_STABLE_LAUNCH_ARG_KEYS
    server_info = {k: info[k] for k in info.keys() & stable}
    mask_path = _ds_mask_path(server_info)
    if mask_path:
        server_info["_ds_channel_mask_sha256"] = _read_mask_content_sha256(mask_path)
    return server_info


def _ds_mask_path(server_info: Dict[str, Any]) -> Optional[str]:
    """Extract channel_mask_path from a DS side's double_sparsity_config (str or dict)."""
    cfg = server_info.get("double_sparsity_config")
    if isinstance(cfg, str):
        try:
            cfg = json.loads(cfg)
        except (ValueError, TypeError):
            return None
    if isinstance(cfg, dict):
        return cfg.get("channel_mask_path")
    return None


def _collect_niah(H, base_url, lengths, num, max_new, seed) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for L in lengths:
        needles = [H._niah_needle(L, i) for i in range(num)]
        prompts = [H._make_niah_prompt(L, seed=seed + i, needle=needles[i]) for i in range(num)]
        attempts = [H._generate_attempt(base_url, p, max_new_tokens=max_new, use_chat=True)
                    for p in prompts]
        served, first_err = H._summarize_attempts(attempts)
        hits = H._niah_recall_hits(needles, [a.text for a in attempts])
        served_tokens = [a.prompt_tokens for a in attempts if a.ok and a.prompt_tokens is not None]
        usage_missing = any(a.ok and a.prompt_tokens is None for a in attempts)
        rows.append({
            "length_words": L, "num_prompts": num, "served": served, "hits": hits,
            "recall_pct": round(_pct(hits, num), 2),
            "max_prompt_tokens": max(served_tokens) if served_tokens else None,
            "usage_missing": usage_missing, "first_error": first_err,
            "prompt_set_hash": _sha("|".join(prompts)),
        })
    return rows


def collect(base_url: str, side: str, *, out_path: Optional[str] = None) -> Dict[str, Any]:
    if side not in ("dsa", "ds"):
        raise GateError(f"AC12_SIDE must be 'dsa' or 'ds', got {side!r}")
    H = _load_harness()
    Q = H.TestDoubleSparsityV32Quality
    index_topk = H._env_int("AC12_INDEX_TOPK", 2048)
    niah_num = H._env_int("AC12_NIAH_NUM_PROMPTS", 20)
    niah_max_new = H._env_int("AC12_NIAH_MAX_NEW_TOKENS", 64)
    niah_seed = H._env_int("AC12_NIAH_SEED", 1234)
    within = list(Q.NIAH_WITHIN_BUDGET_LENGTHS)
    beyond = list(Q.NIAH_CHARACTERIZATION_LENGTHS)
    server_info = _fetch_server_info(base_url)

    niah_within = _collect_niah(H, base_url, within, niah_num, niah_max_new, niah_seed)
    niah_beyond = _collect_niah(H, base_url, beyond, niah_num, niah_max_new, niah_seed)

    # MMLU 5-shot (raw /generate), reusing the harness loader + tuned parser.
    data_dir = _default_mmlu_data_dir()
    subjects_env = os.environ.get("AC12_MMLU_SUBJECTS", "all")
    max_examples = H._env_int("AC12_MMLU_NUM_EXAMPLES", 200)
    dev_dir, test_dir = H._ensure_mmlu_data_dir(data_dir)
    subjects = None if subjects_env == "all" else subjects_env.split(",")
    examples, _per_subject = H._load_mmlu_examples(dev_dir, test_dir, subjects=subjects,
                                                   max_examples=max_examples)
    mmlu_hits = mmlu_served = 0
    mmlu_first_err: Optional[str] = None
    mmlu_prompts: List[str] = []
    for ex in examples:
        prompt = H._make_mmlu_5shot_prompt(ex["dev"], ex["subject"], ex["row"])
        mmlu_prompts.append(prompt)
        att = H._generate_attempt(base_url, prompt, max_new_tokens=4, use_chat=False)
        if att.ok:
            mmlu_served += 1
            pred = H._parse_mmlu_letter(att.text.strip())
            if pred is not None and pred == str(ex["row"][5]).strip().upper():
                mmlu_hits += 1
        elif mmlu_first_err is None:
            mmlu_first_err = att.error

    run_id = _sha(json.dumps({
        "index_topk": index_topk, "within": within, "beyond": beyond,
        "niah_num": niah_num, "niah_seed": niah_seed,
        "mmlu_subjects": sorted(subjects or ["all"]), "mmlu_total": len(examples),
        "mmlu_prompt_set_hash": _sha("|".join(mmlu_prompts)),
    }, sort_keys=True))
    artifact = {
        "schema": SCHEMA, "side": side, "run_id": run_id, "index_topk": index_topk,
        "base_url": base_url, "server_info": server_info,
        "mmlu": {"hits": mmlu_hits, "total": len(examples), "served": mmlu_served,
                 "first_error": mmlu_first_err, "num_examples": max_examples,
                 "prompt_set_hash": _sha("|".join(mmlu_prompts))},
        "niah_within_budget": niah_within,
        "niah_beyond_budget": niah_beyond,
    }
    out_path = out_path or os.environ.get("AC12_OUTPUT_PATH") or os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "results",
                     f"ac12_accuracy_{side}_{run_id}.json"))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:   # fail-closed: no try/except
        json.dump(artifact, fh, indent=2)
    print(f"wrote {out_path} (run_id={run_id} mmlu={mmlu_hits}/{len(examples)} served={mmlu_served})")
    return artifact


def main(argv: List[str]) -> int:
    mode = os.environ.get("AC12_MODE", "")
    if mode == "collect":
        collect(os.environ["AC12_BASE_URL"], os.environ.get("AC12_SIDE", ""))
        return 0
    if mode == "compare":
        with open(os.environ["AC12_DSA_ARTIFACT"]) as f:
            dsa = json.load(f)
        with open(os.environ["AC12_DS_ARTIFACT"]) as f:
            ds = json.load(f)
        verdict = compare(
            dsa, ds,
            expected_ds_mask_path=os.environ.get("AC12_EXPECTED_DS_MASK_PATH", EXPECTED_DS_MASK_PATH),
            expected_ds_mask_sha256=os.environ.get("AC12_EXPECTED_DS_MASK_SHA256", EXPECTED_DS_MASK_SHA256),
        )
        print(json.dumps(verdict, indent=2))
        return 0 if verdict["mandatory_pass"] else 1
    print("set AC12_MODE=collect (AC12_SIDE, AC12_BASE_URL) or "
          "AC12_MODE=compare (AC12_DSA_ARTIFACT, AC12_DS_ARTIFACT)", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
