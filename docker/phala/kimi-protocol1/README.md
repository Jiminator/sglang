# Kimi-K3 protocol repair image

This compatibility release derives from the immutable SGLang 0.5.21 mcfix3
image and replaces eight reviewed Python source files plus one tested XGrammar
wheel. The complete engine source retains mcfix3 commit
`e9a40ece22f8539d434ccc138336019ddbf2e06c` and adds the protocol implementation
at `dfbdb81c0711575763d5ed26d04b0dd7e498faf0`. Product hashes and dependency
provenance are fixed in [release-inputs.json](release-inputs.json).
The development commit stores LF blobs while its qualified checkout contains
seven CRLF files and the new LF encoding module. The release packaging commit preserves those exact qualified product
bytes with per-file `-text` attributes, including when consumed by Git archive.

The repair keeps native grammar ownership through thinking, tools and normal
response boundaries; preserves JSON-string tool values and history; validates
strict final tool output; rejects unsupported schema roots explicitly; and
avoids copying the entire accepted-token history during rollback. It does not
enable a performance candidate or change engine/PIG concurrency, model, KV,
Mooncake, routing or deployment credentials.

## Build

Check out source tag `phala-v0.5.21-3-k3protocol1`. Download its existing tested
wheel Release asset into the build context's `wheels/` directory:

```bash
mkdir -p wheels
gh release download phala-v0.5.21-3-k3protocol1 \
  --repo Phala-Network/sglang \
  --pattern 'xgrammar-0.2.6rc2+phala.ecmadot.20261014.p1-cp312-cp312-linux_x86_64.whl' \
  --dir wheels
docker build --build-arg ENGINE_REVISION="$(git rev-parse HEAD)" \
  -f docker/phala/kimi-protocol1/Dockerfile \
  -t ghcr.io/phala-network/sglang:v0.5.21-phala-3-k3protocol1 .
```

The Dockerfile pins the base digest, checks the original eight source paths
(including absence of the new `encoding_k3.py`), installs the exact wheel without
dependency resolution, and checks the final product hashes. Source is installed
in the immutable image; no runtime source mount or startup installation is used.
The three [XGrammar deltas](xgrammar/) apply in their listed order to public
`mlc-ai/xgrammar` v0.2.6rc2, commit
`8d9679cc4b2b7eb314f7840fbf3989e2a989f880`: JSON Schema dot semantics, the
official 0.2.7 parallel-tool API backport, then the distribution identity.
Use the Release wheel; rebuilding the dependency would create a new untested
binary. Preserve its Apache-2.0 upstream license and attribution.

## Qualification and limits

The unchanged protocol files and tested p1 dependency passed 809 CPU tests with
782 subtests, 48 HTTP cases and the 60-case K3 direct-backend live matrix. Twelve
synthetic-tokenizer skips remain; their real-tokenizer counterparts ran.
Final image and production/public entrypoint acceptance are separate stages
recorded by the image Release receipt.

Loose tools remain loose unless the caller or server requests strict parameter
constraints. Unsupported root combinations/dependencies return explicit 400s;
the release does not claim full support for every JSON Schema keyword, improved
natural-traffic success rate, or E2E throughput gains. Shared v0.5.20 `main` and
the newer 0.5.21 GGUF maintenance composition are separate inputs and are not
merged into this compatibility release.
