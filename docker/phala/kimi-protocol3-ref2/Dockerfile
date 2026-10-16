FROM ghcr.io/phala-network/sglang:v0.5.21-phala-3-k3protocol3-ref1@sha256:0a89ec2a3644f7787eac6838b23adc80880d7b8a51735d0d44604bde72ed087c
ARG ENGINE_REVISION
LABEL org.opencontainers.image.source="https://github.com/Phala-Network/sglang" \
      org.opencontainers.image.revision="${ENGINE_REVISION}" \
      org.opencontainers.image.version="v0.5.21-phala-3-k3protocol3-ref2" \
      io.phala.sglang.revision="${ENGINE_REVISION}"
COPY python/sglang/srt/function_call/kimik3_structural_tag.py /sgl-workspace/sglang/python/sglang/srt/function_call/kimik3_structural_tag.py
