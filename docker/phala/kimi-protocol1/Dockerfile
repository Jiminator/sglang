FROM ghcr.io/phala-network/sglang:v0.5.21-phala-3-mcfix3@sha256:f077dab5c1d77930e4c117b609bd8bab8bee5216eb585e7f682ed0f6e3a23e13
ARG ENGINE_REVISION
LABEL org.opencontainers.image.source="https://github.com/Phala-Network/sglang" \
      org.opencontainers.image.revision="${ENGINE_REVISION}" \
      org.opencontainers.image.version="v0.5.21-phala-3-k3protocol1" \
      io.phala.sglang.revision="${ENGINE_REVISION}"
COPY docker/phala/kimi-protocol1/base-files.json docker/phala/kimi-protocol1/product-files.json docker/phala/kimi-protocol1/verify_files.py /tmp/k3-protocol/
RUN /opt/sglang/bin/python3 /tmp/k3-protocol/verify_files.py /tmp/k3-protocol/base-files.json
COPY python/sglang/srt/constrained/grammar_manager.py /sgl-workspace/sglang/python/sglang/srt/constrained/grammar_manager.py
COPY python/sglang/srt/constrained/reasoner_grammar_backend.py /sgl-workspace/sglang/python/sglang/srt/constrained/reasoner_grammar_backend.py
COPY python/sglang/srt/constrained/xgrammar_backend.py /sgl-workspace/sglang/python/sglang/srt/constrained/xgrammar_backend.py
COPY python/sglang/srt/entrypoints/openai/chat_encoding.py /sgl-workspace/sglang/python/sglang/srt/entrypoints/openai/chat_encoding.py
COPY python/sglang/srt/entrypoints/openai/encoding_k3.py /sgl-workspace/sglang/python/sglang/srt/entrypoints/openai/encoding_k3.py
COPY python/sglang/srt/entrypoints/openai/serving_chat.py /sgl-workspace/sglang/python/sglang/srt/entrypoints/openai/serving_chat.py
COPY python/sglang/srt/function_call/kimik3_detector.py /sgl-workspace/sglang/python/sglang/srt/function_call/kimik3_detector.py
COPY python/sglang/srt/function_call/kimik3_structural_tag.py /sgl-workspace/sglang/python/sglang/srt/function_call/kimik3_structural_tag.py
COPY wheels/xgrammar-0.2.6rc2+phala.ecmadot.20261014.p1-cp312-cp312-linux_x86_64.whl /tmp/k3-protocol/xgrammar-0.2.6rc2+phala.ecmadot.20261014.p1-cp312-cp312-linux_x86_64.whl
RUN echo "e80259e5837dd5f8c21cac0b95906def48b59bcb27705697733d438ce7acd36b  /tmp/k3-protocol/xgrammar-0.2.6rc2+phala.ecmadot.20261014.p1-cp312-cp312-linux_x86_64.whl" | sha256sum -c - \
    && /opt/sglang/bin/python3 -m pip install --no-cache-dir --no-deps --no-index /tmp/k3-protocol/xgrammar-0.2.6rc2+phala.ecmadot.20261014.p1-cp312-cp312-linux_x86_64.whl \
    && /opt/sglang/bin/python3 /tmp/k3-protocol/verify_files.py /tmp/k3-protocol/product-files.json \
    && rm /tmp/k3-protocol/xgrammar-0.2.6rc2+phala.ecmadot.20261014.p1-cp312-cp312-linux_x86_64.whl
