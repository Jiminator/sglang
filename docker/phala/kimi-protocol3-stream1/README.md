# Kimi buffered tool stream progress

Kimi's parser can buffer an entire tool call while model tokens continue to
arrive. After the first model event, a tool-enabled Kimi chat stream now sends
an SSE comment at most once every 15 seconds when completion tokens advance
without a data event. A stalled generator sends no comment. There is no timer
task or early HTTP success response.

The change preserves JSON payloads, usage, finish reasons, first-error status,
and existing cancellation cleanup. It changes one Python product file and
inherits dependencies, native binaries, and loading behavior from the pinned
observe1 runtime. SSE clients must ignore comments as required by the protocol.

CPU regressions cover buffered progress, other models, visible content, no new
tokens, terminal events, pre-header failures, cancellation, and multiple choices.
The previous runtime's actual synthetic long tool request was silent for 72.5
seconds after its first event while generation continued. That observation
does not by itself explain historical gateway failures at approximately 125
seconds, or prove a production performance improvement.

Production scope is node101 only. Published source and image identities, final
image checks, real-chain acceptance, and natural observation remain separate
release gates. Formal deployment must not use fastsafetensors.
