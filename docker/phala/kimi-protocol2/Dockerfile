FROM ghcr.io/phala-network/sglang:v0.5.21-phala-3-k3protocol1@sha256:31d5b201dd88b46210b13cee50d87d328aff2c41432beb5e897a5d478b3874f4
ARG ENGINE_REVISION
LABEL org.opencontainers.image.source="https://github.com/Phala-Network/sglang" \
      org.opencontainers.image.revision="${ENGINE_REVISION}" \
      org.opencontainers.image.version="v0.5.21-phala-3-k3protocol2" \
      io.phala.sglang.revision="${ENGINE_REVISION}"
COPY docker/phala/kimi-protocol2/base-files.json docker/phala/kimi-protocol2/product-files.json docker/phala/kimi-protocol2/verify_files.py /tmp/k3-feedback/
RUN /opt/sglang/bin/python3 /tmp/k3-feedback/verify_files.py /tmp/k3-feedback/base-files.json
COPY python/sglang/srt/constrained/reasoner_grammar_backend.py /sgl-workspace/sglang/python/sglang/srt/constrained/reasoner_grammar_backend.py
COPY python/sglang/srt/constrained/kimik3_json_schema.py /sgl-workspace/sglang/python/sglang/srt/constrained/kimik3_json_schema.py
COPY python/sglang/srt/constrained/json_schema_validation.py /sgl-workspace/sglang/python/sglang/srt/constrained/json_schema_validation.py
COPY python/sglang/srt/function_call/kimik3_structural_tag.py /sgl-workspace/sglang/python/sglang/srt/function_call/kimik3_structural_tag.py
COPY python/sglang/srt/entrypoints/openai/serving_chat.py /sgl-workspace/sglang/python/sglang/srt/entrypoints/openai/serving_chat.py
COPY python/sglang/srt/entrypoints/openai/serving_responses.py /sgl-workspace/sglang/python/sglang/srt/entrypoints/openai/serving_responses.py
COPY python/sglang/srt/parser/reasoning_parser.py /sgl-workspace/sglang/python/sglang/srt/parser/reasoning_parser.py
COPY wheels/xgrammar-0.2.6rc2+phala.feedback.20261015.p3-cp312-cp312-linux_x86_64.whl /tmp/k3-feedback/xgrammar-0.2.6rc2+phala.feedback.20261015.p3-cp312-cp312-linux_x86_64.whl
RUN echo "5235045af2d459f260d4e85fb3632631de858d91bbeec17adc2e75f21992956a  /tmp/k3-feedback/xgrammar-0.2.6rc2+phala.feedback.20261015.p3-cp312-cp312-linux_x86_64.whl" | sha256sum -c - \
    && /opt/sglang/bin/python3 -m pip install --no-cache-dir --no-deps --no-index /tmp/k3-feedback/xgrammar-0.2.6rc2+phala.feedback.20261015.p3-cp312-cp312-linux_x86_64.whl \
    && /opt/sglang/bin/python3 /tmp/k3-feedback/verify_files.py /tmp/k3-feedback/product-files.json \
    && rm /tmp/k3-feedback/xgrammar-0.2.6rc2+phala.feedback.20261015.p3-cp312-cp312-linux_x86_64.whl
