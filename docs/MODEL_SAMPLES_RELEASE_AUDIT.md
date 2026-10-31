# Reusable sample contract release audit

Status: v4 contract **not yet fully conformant**. Sample schema, remote integrity,
manifest discovery, and representative public-API inference passed. The seven speech
recognition workers reuse whisper-standard.mp3, whose upstream source and redistribution
license are not recorded. Resolve that provenance before publishing the package.

- 61/61 workers expose nonempty samples through the public WebAI API without inference initialization.
- All 36 unique fixture/asset URLs were verified against local bytes and SHA-1.
- Missing MediaPipe face/hand fixtures were uploaded unchanged to the public B2 benchmark prefix.
- Every advertised precision/device pair still has a saved passing benchmark result.
- Existing matrices were reused: artifact revisions, runtime versions, fixture bytes, quality gates,
  and advertised support were unchanged. Input-schema normalization only reflects existing behavior.
- Public-API quality smoke tests passed: MiniLM q8/wasm (semantic margin 0.744), ORMBG q8/wasm
  (mask MAE 0.000208, IoU 0.999459), Moonshine Tiny q4/wasm (WER 0.209).
- Memory release and retained cache were verified for those three configurations; test cleanup
  disposes models and removes only the tested model precision.
- BERT Small PII Detection and Sweep Next Edit remain unsupported: no saved quality-passing pair.
  Other failed combinations remain recorded in their unchanged benchmark matrices.

Checks: all 61 worker syntax checks and template syntax, library typecheck/build, sample preparation
unit tests, remote sample contract audit, test-app lint/build, LocalAI Tools lint/typecheck/build/tests,
JSON parse validation, git diff --check, and npm pack --dry-run. Test-app lint has one pre-existing
Next.js no-img-element warning in vision-to-text/page.tsx.

Reports: benchmarks/model-samples-contract-audit.json and benchmarks/model-samples-public-api.json.
The contract audit separates sample functionality (`passed`) from full fixture provenance
(`fullFixtureProvenancePassed`); both must be true for a fully conformant release.
