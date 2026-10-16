# Preparation only: p6 source/wheel and the complete engine commit must be frozen first.
FROM ghcr.io/phala-network/sglang:v0.5.21-phala-3-k3protocol3-ref4@sha256:dea82d95eacb380041e465735d32056cda0d145866cec04381d6dd46c5039447
ARG ENGINE_REVISION
ARG XGRAMMAR_SOURCE_REVISION=d645e9d0dbaf6cc85c2ca27acc8f9b4a4b83cbdc
ARG XGRAMMAR_WHEEL_SHA256=ee7d54fb8b77e712e8d5e8b9f7aee59f4ead18409d6f00a4c0fb5e63ca5b7832
LABEL org.opencontainers.image.source="https://github.com/Phala-Network/sglang" \
      org.opencontainers.image.revision="${ENGINE_REVISION}" \
      org.opencontainers.image.version="v0.5.21-phala-3-k3protocol3-ref5" \
      io.phala.sglang.revision="${ENGINE_REVISION}" \
      io.phala.xgrammar.revision="${XGRAMMAR_SOURCE_REVISION}" \
      io.phala.xgrammar.version="0.2.6rc2+phala.feedback.20261017.p6" \
      io.phala.xgrammar.serialization.version="v21-phala-unordered-constraints-v3" \
      io.phala.xgrammar.wheel.sha256="${XGRAMMAR_WHEEL_SHA256}"
COPY wheels/xgrammar-0.2.6rc2+phala.feedback.20261017.p6-cp312-cp312-linux_x86_64.whl /tmp/
RUN test ${#ENGINE_REVISION} -eq 40 && test ${#XGRAMMAR_SOURCE_REVISION} -eq 40 && test ${#XGRAMMAR_WHEEL_SHA256} -eq 64 \
    && echo "${XGRAMMAR_WHEEL_SHA256}  /tmp/xgrammar-0.2.6rc2+phala.feedback.20261017.p6-cp312-cp312-linux_x86_64.whl" | sha256sum --check - \
    && /opt/sglang/bin/python3 -m pip install --no-deps --no-index --no-cache-dir --force-reinstall /tmp/xgrammar-0.2.6rc2+phala.feedback.20261017.p6-cp312-cp312-linux_x86_64.whl \
    && rm /tmp/xgrammar-0.2.6rc2+phala.feedback.20261017.p6-cp312-cp312-linux_x86_64.whl
