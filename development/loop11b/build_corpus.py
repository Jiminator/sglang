#!/usr/bin/env python3
"""Rebuild the deterministic Pile-val calibration corpus for GLM-5.1-FP8 channel-mask
calibration.

In-order stream of the ``mit-han-lab/pile-val-backup`` validation split; keep the first
300 documents with >= 1500 characters; collapse all whitespace so each doc is exactly one
line. In-order streaming is deterministic, so re-running reproduces the same file:

    sha256(calib_corpus_pileval.txt) = 46d72075ed1d64831cf6777cf328b85ac025bb76cf36cc87675bfe5afa6535f0

(this matches the byte-identical corpus the earlier DeepSeek-V3.2 calibration used). The
corpus blob itself is gitignored (under artifacts/); it is committed-by-reference via this
reproducer + the recorded sha256 in the calibration provenance.

Requires: pip install datasets zstandard
Usage:    python development/loop11b/build_corpus.py [OUT_PATH]
"""
import os
import sys

from datasets import load_dataset

OUT = sys.argv[1] if len(sys.argv) > 1 else "development/loop11b/artifacts/calib_corpus_pileval.txt"
MIN_CHARS = 1500
TARGET_DOCS = 300
SCAN_CAP = 20000

ds = load_dataset("mit-han-lab/pile-val-backup", split="validation", streaming=True)
docs = []
scanned = 0
for ex in ds:
    scanned += 1
    text = (ex.get("text") or "").strip()
    if len(text) >= MIN_CHARS:
        docs.append(" ".join(text.split()))
        if len(docs) >= TARGET_DOCS:
            break
    if scanned > SCAN_CAP:
        break

# The default OUT lives under the gitignored artifacts/ dir, absent on a clean
# checkout — create the parent dir so the documented default command works.
os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
with open(OUT, "w") as fh:
    for doc in docs:
        fh.write(doc + "\n")
print(f"scanned={scanned} kept={len(docs)} -> {OUT}")
