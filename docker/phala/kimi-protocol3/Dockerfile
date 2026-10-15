FROM ghcr.io/phala-network/sglang:v0.5.21-phala-3-k3protocol1@sha256:31d5b201dd88b46210b13cee50d87d328aff2c41432beb5e897a5d478b3874f4
ARG ENGINE_REVISION
LABEL org.opencontainers.image.source="https://github.com/Phala-Network/sglang" org.opencontainers.image.revision="${ENGINE_REVISION}" org.opencontainers.image.version="v0.5.21-phala-3-k3protocol3"
COPY docker/phala/kimi-protocol3/ /tmp/k3-release/
RUN /opt/sglang/bin/python3 /tmp/k3-release/verify_files.py /tmp/k3-release/base-files.json
COPY python/sglang/ /sgl-workspace/sglang/python/sglang/
COPY ["wheels/xgrammar-0.2.6rc2+phala.feedback.20261016.p5-cp312-cp312-linux_x86_64.whl", "/tmp/k3-release/xgrammar-0.2.6rc2+phala.feedback.20261016.p5-cp312-cp312-linux_x86_64.whl"]
RUN /opt/sglang/bin/python3 -m pip install --no-deps --no-index /tmp/k3-release/xgrammar-0.2.6rc2+phala.feedback.20261016.p5-cp312-cp312-linux_x86_64.whl
COPY native/ /tmp/k3-native/
RUN /opt/sglang/bin/python3 /tmp/k3-release/install_native.py && /opt/sglang/bin/python3 /tmp/k3-release/verify_files.py /tmp/k3-release/product-files.json && /opt/sglang/bin/python3 /tmp/k3-release/verify_native.py && rm -rf /tmp/k3-native
