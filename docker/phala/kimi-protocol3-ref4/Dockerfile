FROM ghcr.io/phala-network/sglang:v0.5.21-phala-3-k3protocol3-ref3@sha256:b05b00447d4b2fcc8e99f48e862917bafefcabf27fe3d90105a98ed218a87054
ARG ENGINE_REVISION
LABEL org.opencontainers.image.source="https://github.com/Phala-Network/sglang" \
      org.opencontainers.image.revision="${ENGINE_REVISION}" \
      org.opencontainers.image.version="v0.5.21-phala-3-k3protocol3-ref4" \
      io.phala.sglang.revision="${ENGINE_REVISION}"
COPY python/sglang/srt/function_call/function_call_parser.py /sgl-workspace/sglang/python/sglang/srt/function_call/function_call_parser.py
COPY python/sglang/srt/function_call/kimik3_constraint_diagnostics.py /sgl-workspace/sglang/python/sglang/srt/function_call/kimik3_constraint_diagnostics.py
COPY python/sglang/srt/constrained/base_grammar_backend.py /sgl-workspace/sglang/python/sglang/srt/constrained/base_grammar_backend.py
COPY python/sglang/srt/constrained/reasoner_grammar_backend.py /sgl-workspace/sglang/python/sglang/srt/constrained/reasoner_grammar_backend.py
COPY python/sglang/srt/constrained/xgrammar_backend.py /sgl-workspace/sglang/python/sglang/srt/constrained/xgrammar_backend.py
