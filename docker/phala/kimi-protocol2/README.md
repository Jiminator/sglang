# Kimi feedback compatibility image

This recipe extends the accepted K3 protocol1 image with the seven qualified
schema, parser and response lifecycle files in `product-files.json`, plus the
already tested XGrammar p3 wheel. The base is pinned to
`sha256:31d5b201dd88b46210b13cee50d87d328aff2c41432beb5e897a5d478b3874f4`.
It preserves SGLang 0.5.21, Mooncake and all other installed dependencies.

`release-inputs.json` binds the qualified product commit, exact source bytes,
native source tree, complete native delta and qualification-receipt hashes.
The build verifies the base paths, the wheel SHA256 and the resulting product
paths. It uses `pip --no-deps --no-index`; it does not rebuild the wheel.
The native `LICENSE` and `NOTICE` are retained under `xgrammar/`.

Place the exact wheel in the source root's `wheels/` directory:

```text
xgrammar-0.2.6rc2+phala.feedback.20261015.p3-cp312-cp312-linux_x86_64.whl
SHA256 5235045af2d459f260d4e85fb3632631de858d91bbeec17adc2e75f21992956a
```

From the clean, committed full-engine checkout on the authorized builder:

```sh
sh docker/phala/kimi-protocol2/build.sh
```

Use the existing v0.5.21 compatibility-maintenance release exception documented
in the serving-patches version README. The shared v0.5.20 integration and GGUF
selection remain separate. The source version is
`phala-v0.5.21-3-k3protocol2`; the image version is
`ghcr.io/phala-network/sglang:v0.5.21-phala-3-k3protocol2`.
Do not replace an already published version.

Before registry publication, read back the exact remote source tag and its
published Release (including the same wheel asset). Qualify the actual final
image ID without source, package or virtualenv mounts: direct HF CLI, SGLang
CLI, installed source/native identity, affected feedback regressions and
Mooncake master/store package compatibility. Source-only evidence does not
qualify the final image, GPU inference or production. Publish once, read back
the immutable digest and use that same reference for backend/downloader and
Mooncake master/store.
