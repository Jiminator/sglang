FROM ghcr.io/phala-network/sglang:v0.5.21-phala-3-k3protocol3-ref2@sha256:59cf8ba0a8d3aa45dd9592f3aa9de8264da9484d9bc524313ba7308c2be0fa04
ARG ENGINE_REVISION
LABEL org.opencontainers.image.source="https://github.com/Phala-Network/sglang" \
      org.opencontainers.image.revision="${ENGINE_REVISION}" \
      org.opencontainers.image.version="v0.5.21-phala-3-k3protocol3-ref3" \
      io.phala.sglang.revision="${ENGINE_REVISION}"
COPY python/sglang/srt/function_call/kimik3_structural_tag.py /sgl-workspace/sglang/python/sglang/srt/function_call/kimik3_structural_tag.py
COPY python/sglang/srt/constrained/json_schema_intersections.py /sgl-workspace/sglang/python/sglang/srt/constrained/json_schema_intersections.py
