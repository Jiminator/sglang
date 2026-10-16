FROM ghcr.io/phala-network/sglang:v0.5.21-phala-3-k3protocol3-observe1@sha256:d8101eb762398688f7528bf52352829d21c376a697239b60e96db30c2ce583d1
ARG ENGINE_REVISION
LABEL org.opencontainers.image.source="https://github.com/Phala-Network/sglang" \
      org.opencontainers.image.revision="${ENGINE_REVISION}" \
      org.opencontainers.image.version="v0.5.21-phala-3-k3protocol3-stream1" \
      io.phala.sglang.revision="${ENGINE_REVISION}"
COPY python/sglang/srt/entrypoints/openai/serving_chat.py /sgl-workspace/sglang/python/sglang/srt/entrypoints/openai/serving_chat.py
