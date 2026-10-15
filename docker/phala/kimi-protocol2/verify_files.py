"""Fail closed until the seven product blobs are explicitly frozen."""
import hashlib
import json
from pathlib import Path
import re
import sys

root = Path("/sgl-workspace/sglang")
manifest = json.loads(Path(sys.argv[1]).read_text())
assert len(manifest) == 7
for relative, expected in manifest.items():
    target = root / relative
    assert target.resolve().is_relative_to(root), relative
    if expected is None:
        assert not target.exists(), "Base unexpectedly contains " + relative
    else:
        assert isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected), "Unfrozen manifest: " + relative
        assert hashlib.sha256(target.read_bytes()).hexdigest() == expected, "Source bytes differ: " + relative
print("Seven source paths verified")
