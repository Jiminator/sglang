"""Diagnostic — does the DSA-native default radix cache EVER produce a cache hit?
Sends short / medium / the exact 6090-tok ExpB prompt, each TWICE, and reports
cached_tokens on each send. Distinguishes a prompt-specific miss from a general
DSA-default 'never caches' behavior. Standard /generate API only."""
import json, os, requests
BASE = os.environ.get("DS_BASE_URL", "http://127.0.0.1:30000")

def gen(text, n=4):
    r = requests.post(f"{BASE}/generate", json={
        "text": text, "sampling_params": {"max_new_tokens": n, "temperature": 0}}, timeout=300)
    m = r.json()["meta_info"]
    return m.get("cached_tokens"), m.get("prompt_tokens")

short = "The quick brown fox jumps over the lazy dog. " * 6 + "Tell a short story."
med = "The capital city facts and figures vary widely. " * 120
base = "The quick brown fox jumps over the lazy dog near the riverbank at dawn. "
long = "".join(f"[{i}] {base}" for i in range(316)) + " Summarize the passage above in one sentence."

for label, txt in [("short", short), ("medium", med), ("long6090", long)]:
    c1, p1 = gen(txt)
    c2, p2 = gen(txt)
    print(f"{label:9s} send1: cached={c1} ptok={p1}   send2: cached={c2} ptok={p2}   "
          f"{'HIT' if (c2 or 0) > 0 else 'MISS'}")
