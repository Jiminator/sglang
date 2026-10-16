# Kimi reference and length constraints

Maintenance candidate on the immutable protocol3-stream1 image. It retains the
existing native libraries, loading mode, diagnostics and buffered tool stream
progress fix. Only two installed Python schema-normalization files change.

The bounded reference reducer now intersects minLength, maxLength and type along
local reference chains, preserves nullable/non-string alternatives, and removes
integer minLength=0 as a no-op. Its derived-definition cache includes every new
constraint. Unsupported combinations still fail explicitly; this is not general
JSON Schema intersection support. There is no added token-loop work or cache.

Synthetic CPU reproduction: seven failures on the previous installed image.
Candidate checks: 54 passed plus eight subtests, with independent jsonschema and
native grammar agreement, actual Kimi json-string encoding, legacy dialects,
reference chains, invalid inputs and immutability. Final image and production
acceptance are separate gates; this evidence does not prove the exact customer
schema or resolve all observed memory, image-fetch and transport failures.

Build with this Dockerfile and ENGINE_REVISION set to the frozen source commit.
Formal deployment must not use fastsafetensors. Only node 101 is authorized for
this rollout; preserve the existing node credential and serving configuration.
