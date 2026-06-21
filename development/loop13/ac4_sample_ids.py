#!/usr/bin/env python3
"""AC-4 per-arm GSM8K sample IDs/order (offline, re-derived from the stock loader).

`simple_eval_gsm8k.GSM8KEval` selects examples DETERMINISTICALLY (no random/seed/shuffle): it loads the
fixed-order test.jsonl, uses the first `num_shots` lines as few-shot context, then evaluates the slice
`lines[num_shots : num_shots+num_examples]` in order. So every arm that ran the same (num_shots,
num_examples) config used the IDENTICAL ordered example set. This reproduces and persists that set —
the line index in test.jsonl plus a sha256 of each question — so every per-arm GSM8K row is reproducible.

Configs: dense = 5-shot / 200 (eval lines [5:205]); sparse = 24-shot / 150 (eval lines [24:174]).
Writes evidence/gsm8k_sample_ids.json. CPU-only; fail-closed if the loader/order cannot be reproduced.
"""
import hashlib
import json
import os
import sys

from sglang.test.simple_eval_gsm8k import GSM8K_URL
from sglang.utils import download_and_cache_file, read_jsonl

HERE = os.path.dirname(os.path.abspath(__file__))


def _qsha(line):
    return hashlib.sha256(line["question"].encode("utf-8")).hexdigest()[:16]


def derive(lines, num_shots, num_examples):
    few = list(range(num_shots))
    eval_lines = lines[num_shots:][:num_examples]
    eval_idx = list(range(num_shots, num_shots + len(eval_lines)))
    return {
        "num_shots": num_shots,
        "num_examples": num_examples,
        "few_shot_line_indices": few,
        "eval_line_indices": [eval_idx[0], eval_idx[-1]] if eval_idx else [],
        "n_eval": len(eval_lines),
        # ordered (line_index, question_sha16) — the authoritative per-example identity/order
        "eval_examples": [{"line": i, "q_sha16": _qsha(l)} for i, l in zip(eval_idx, eval_lines)],
    }


def main():
    filename = download_and_cache_file(GSM8K_URL)
    file_sha = hashlib.sha256(open(filename, "rb").read()).hexdigest()
    lines = list(read_jsonl(filename))

    dense = derive(lines, 5, 200)
    sparse = derive(lines, 24, 150)
    report = {
        "ac": "AC-4 per-arm GSM8K sample IDs/order",
        "source": GSM8K_URL,
        "test_jsonl_sha256": file_sha,
        "total_lines": len(lines),
        "selection_rule": ("stock simple_eval_gsm8k.GSM8KEval: deterministic, no seed/shuffle — few-shot = "
                           "first num_shots lines; eval = lines[num_shots:num_shots+num_examples] in order. "
                           "ALL arms with the same config used this identical ordered set."),
        "dense_5shot_200": dense,
        "sparse_24shot_150": sparse,
        "applies_to_arms": "every GSM8K arm in evidence/meta/arms (dsa, dsa_noradix, production_ds, "
                           "ref_faithful, ref_cosine, ref_cosine_noinc, ds_reduce_fp32, ds_forced_all, ds_anchor_*)",
    }
    out = os.path.join(HERE, "evidence", "gsm8k_sample_ids.json")
    json.dump(report, open(out, "w"), indent=2)
    print(json.dumps({"test_jsonl_sha256": file_sha, "total_lines": len(lines),
                      "dense": {"eval_line_indices": dense["eval_line_indices"], "n": dense["n_eval"]},
                      "sparse": {"eval_line_indices": sparse["eval_line_indices"], "n": sparse["n_eval"]}},
                     indent=2))
    print("wrote", out)
    if dense["n_eval"] != 200 or sparse["n_eval"] != 150:
        print(f"FAIL: expected 200 dense / 150 sparse, got {dense['n_eval']}/{sparse['n_eval']} "
              f"(dataset changed or load failed)", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
