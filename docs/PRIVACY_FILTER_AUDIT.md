# OpenAI Privacy Filter v4 contract audit

Audit date: 2026-09-15.

**Final status: v4 contract: conformant.** Recommended configuration: **Q4/WebGPU**.

The worker and manifest were checked against `docs/WORKER_CONTRACT.md` and Whisper Tiny. They include correlated boot/handshake and operations, one error boundary, exact Transformers.js 4.2.0, immutable source/artifact revisions, authoritative Apache-2.0 license and provider URLs, typed input/config/output, exact external-data weight totals and estimated complete downloads, validation, progress, standard native/runtime envelope, explicit non-stream support, and disposal.

The isolated-browser matrix tested all five published artifacts on WASM and WebGPU. Q4/WebGPU and Q4F16/WebGPU passed. Q4, Q4F16, and Q8 failed on WASM because the required quantized gather kernel was unavailable; Q8/WebGPU ran but failed the checked-in PII quality gate; FP16 and FP32 failed on both backends with out-of-memory errors. All failures remain in `benchmarks/privacy-filter.json` and are excluded from advertised support.

The safe fictional model-card fixture is public and local, 899 bytes, SHA-1 `39b9188d3224c759479738aa156538e01a283414`. The gate requires the expected person/email spans at score >= 0.9, finite scores and valid offsets. It is a regression test, not a complete privacy evaluation.

The public WebAI Q4/WebGPU audit passed manifest, download/progress, inference quality and envelope, invalid-input and streaming rejection, clear-memory/reinit, model cache deletion, and preservation of another model's cache. Its approximately 917 MB weights were not retained by the isolated Chrome Cache API profile, so this limitation is explicit in the manifest.
