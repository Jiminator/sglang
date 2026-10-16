# Kimi protocol3-ref5 native repair

This recipe derives from the immutable ref4 image and replaces only XGrammar p5 with the focused p6 partial-hash repair. Product Python remains the exact ref4 subtree; model and runtime parameters are unchanged.

`release-inputs.json` fixes the native source and wheel. The complete native source is reproduced from the fixed upstream commit using the existing protocol3 p5 source patch, followed by `xgrammar/p5-to-p6.patch`. Existing protocol3 LICENSE/NOTICE remain applicable. The wheel is a separate build input and Release asset, not a tracked source binary.

Build only after the frozen native candidate passes independent dynamic qualification. Build the service image once with network disabled and the complete engine commit as `ENGINE_REVISION`; do not rebuild native in the image or install at runtime. Final tests must use installed code without product or wheel overlays.

P6 changes serialized cache identity to v3. P5 compiled masks must be rejected and rebuilt from original grammar. See [the ref4 regression notice](RISK-REF4-943.md). Earlier ref4 test results do not establish that this new defect is absent.
