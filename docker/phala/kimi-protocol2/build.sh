#!/bin/sh
set -eu
revision=$(git rev-parse HEAD)
test -z "$(git status --porcelain --untracked-files=no)"
test -f wheels/xgrammar-0.2.6rc2+phala.feedback.20261015.p3-cp312-cp312-linux_x86_64.whl
docker image inspect ghcr.io/phala-network/sglang:v0.5.21-phala-3-k3protocol1@sha256:31d5b201dd88b46210b13cee50d87d328aff2c41432beb5e897a5d478b3874f4 >/dev/null
docker build --pull=false --network=none \
  --build-arg "ENGINE_REVISION=$revision" \
  -t ghcr.io/phala-network/sglang:v0.5.21-phala-3-k3protocol2 \
  -f docker/phala/kimi-protocol2/Dockerfile .
