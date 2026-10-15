# Protocol3 bounded diagnostics candidate

This candidate adds fixed rejection locations and on-demand grammar accounting to the pinned protocol3 image. It does not fix the remaining unclassified production HTTP400 responses, prove a memory plateau, or change schema support, cache limits, dependencies, native binaries, model parameters, or loading mode.

The product change is `2d2b61b5a597bbcf537d64eb2c4b104b9170e24a`, based on protocol3 source `83b5988844dc7e7e6dd907ef4c5553d8d3c3c303`. The image copies only the three changed Python files. Build with `ENGINE_REVISION` set to the complete committed source containing this recipe; do not substitute a mutable tag for the base digest.

Base OpenAI error responses log only a fixed category, numeric status, an allowlisted source label, and line number. They do not log exception text or request fields. The existing Responses-specific override and errors after streaming starts are not all covered by that log. Response bodies and method signatures are unchanged.

`server_info` internal state adds `grammar_budget`, tagged with the reporting `tp_rank`. This is the existing control-response rank, not a collective snapshot of all TP ranks. A busy accounting lock or unavailable budget returns `available: false`; do not interpret that as zero. The snapshot reads fixed counters and lengths without walking schemas or compiled contexts.

- `approx_cache_bytes` is the existing cache charge, not RSS or the total native allocation.
- `pending_input_bytes` covers retained input keys, not all compiled results or live matchers.
- Hit/miss count lookups; compile counters count single-flight work. Finished includes exceptions. These are not HTTP request success/error denominators.
- Compile queue seconds cover executor waiting for work that actually starts. Compile wall seconds sum dispatch wall time, including failures, and exclude later TP synchronization and KV prefetch.
- Counters survive cache reset and restart with the backend process. No per-token metrics or new collective communication are added.

The candidate passed 160 isolated CPU checks with the pinned protocol3 image and the three-file overlay: error privacy/response contracts, cache ownership/single-flight/cancellation/reset, preserved PP behavior, nonblocking reads, actual internal-state serialization and environment/capacity contracts. These are source checks; final-image verification and the authorized 101-only rollout remain separate gates. Production stability and Redpill error-rate acceptance remain open.
