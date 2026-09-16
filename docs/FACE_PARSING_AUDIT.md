# Face Parsing v4 contract audit

Audit date: 2026-09-15.

**Final status: v4 contract: conformant.**

The worker was checked field-by-field against `docs/WORKER_CONTRACT.md` and the Whisper Tiny reference. It has a correlated boot/handshake and single request handler, legacy capabilities plus a versioned manifest, separate provider/source/artifact provenance and immutable revisions, the upstream non-commercial research/education `LicenseRef` and authoritative pinned URL, exact Transformers.js 4.2.0 runtime, typed input and native output, exact artifact and complete-download byte counts, validation, download progress, standard result envelope, explicit non-stream support, and disposal.

The full isolated-browser matrix covers all three repository artifacts on WASM and WebGPU. Five of six cases passed the semantic quality gate. FP16/WASM failed ONNX session creation, remains preserved with its full error in `benchmarks/face-parsing.json`, and is not advertised. Recommended configuration: **FP16/WebGPU**; it provides approximately half the FP32 weight download and comparable measured inference speed. Q8 passed on both devices but was very slow on this test machine.

The local/public fixture is the licensed 1024×683 Pexels portrait, 78,124 bytes, SHA-1 `38212739a4a4f0be25d84863cf991d7fb6c6d818`. The checked-in expected result verifies original-size binary masks, known unique labels, fixture-specific visible regions, plausible coverage, and non-degenerate foreground. This is a semantic regression gate, not pixel-annotated ground truth.

The public WebAI audit passed with FP16/WebGPU and verified manifest discovery, public download with progress, cache detection, standard native/runtime output, typed masks, invalid-input rejection, unsupported-stream rejection, clear-memory behavior, cached reinitialization, model-specific cache deletion, and preservation of another model's cache.

Release gates passed: worker syntax check; `npm run typecheck`; `npm run build`; `npm run lint --prefix test-app` with one unrelated pre-existing warning; `npm run build --prefix test-app`; six-case benchmark and allow-list/cleanup validation; fixture size/checksum and public URL validation; public API audit; `git diff --check`; and `npm pack --dry-run --json --cache /tmp/webai-npm-cache`.
