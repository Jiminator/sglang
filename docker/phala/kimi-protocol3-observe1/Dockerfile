FROM ghcr.io/phala-network/sglang:v0.5.21-phala-3-k3protocol3@sha256:e646efb46f7d5a3983de69202a6bd0f67894b994e7d3433144445f2531c177b2
ARG ENGINE_REVISION
LABEL org.opencontainers.image.source="https://github.com/Phala-Network/sglang" \
      org.opencontainers.image.revision="${ENGINE_REVISION}" \
      org.opencontainers.image.version="v0.5.21-phala-3-k3protocol3-observe1" \
      io.phala.sglang.revision="${ENGINE_REVISION}"
COPY python/sglang/srt/entrypoints/openai/serving_base.py /sgl-workspace/sglang/python/sglang/srt/entrypoints/openai/serving_base.py
COPY python/sglang/srt/constrained/base_grammar_backend.py /sgl-workspace/sglang/python/sglang/srt/constrained/base_grammar_backend.py
COPY python/sglang/srt/managers/scheduler.py /sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py
