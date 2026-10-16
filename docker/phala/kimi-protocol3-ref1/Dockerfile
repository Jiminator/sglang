FROM ghcr.io/phala-network/sglang:v0.5.21-phala-3-k3protocol3-stream1@sha256:118dee8b7ed9cc35cbd854bc315ff46b25fec7069a9a2b7b1e034816affe3d20
ARG ENGINE_REVISION
LABEL org.opencontainers.image.source="https://github.com/Phala-Network/sglang" \
      org.opencontainers.image.revision="${ENGINE_REVISION}" \
      org.opencontainers.image.version="v0.5.21-phala-3-k3protocol3-ref1" \
      io.phala.sglang.revision="${ENGINE_REVISION}"
COPY python/sglang/srt/constrained/json_schema_intersections.py /sgl-workspace/sglang/python/sglang/srt/constrained/json_schema_intersections.py
COPY python/sglang/srt/constrained/json_schema_validation.py /sgl-workspace/sglang/python/sglang/srt/constrained/json_schema_validation.py
