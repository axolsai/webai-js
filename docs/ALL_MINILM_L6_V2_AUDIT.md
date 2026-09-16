# All MiniLM L6 v2 — final v4 audit

Worker: `model-workers-v4/all-minilm-l6-v2.worker.js`. Test page: `/minilm-embeddings`.

## Contract

Compared against `WORKER_CONTRACT.md`, the template, and Whisper Tiny: the module-worker boot handshake, capability discovery, correlated request/progress/error envelopes, required operations, and standard generated result envelope are preserved. Embeddings do not advertise streaming; the public API rejects it. Validation rejects empty texts, invalid pooling/normalization, and unknown configuration options before inference. Replaced models and `clearMemory` dispose the pipeline; errors propagate through the single message-handler boundary. No telemetry is added.

The manifest separates original Sentence Transformers provider/source metadata from Xenova's immutable ONNX artifact revision, includes authoritative provider and Apache-2.0 license URLs, describes English/truncation/accuracy/safety limitations, and documents normalized 384-dimensional output and accepted configuration. All eight actual artifacts have exact weight and complete-download sizes. The pinned Transformers.js 4.2.0 initializes and runs in module workers on the tested Chrome/macOS environment.

## Benchmark and recommendation

Full report: `benchmarks/all-minilm-l6-v2.json`; early-case evidence rerun: `benchmarks/all-minilm-l6-v2-evidence.json`; public API: `benchmarks/all-minilm-l6-v2-public-api.json`.

Sixteen isolated cases: fourteen pass semantic retrieval, finite/distinct vector, shape, and normalization gates. FP16/WASM and Q4F16/WASM fail ONNX session creation and are not advertised. Their WebGPU variants pass; FP32, INT8, UINT8, Q8, Q4, and BNB4 pass both devices. Each case records load/inference/total durations, embeddings/second, a performance grade, disposal, and model-scoped cache deletion counts. Successful full vectors and complete failures are retained, and all sixteen compact results are embedded in the worker. Timing is hardware/browser-specific, not a universal accuracy or performance claim.

Recommended and verified through the public WebAI API: **INT8/WASM**, mean pooling and normalization. Exact weights: 22,972,370 bytes; estimated complete first download: 23,916,680 bytes. Public API produced three 384-dimensional vectors in 48.9 ms with a relevant-versus-unrelated cosine margin of 0.74045. Download progress, downloaded-state detection, invalid input rejection, unsupported streaming, clear/reinitialization, and scoped deletion preserving another cache were verified.

The local redistributable English retrieval fixture is 407 bytes, SHA1 `8beb837ef1dfd6c35b4a100d7d947f414f03c0f6`; its immutable public B2 copy matches. This small fixture is a smoke quality gate, not broad retrieval evaluation. Artifact/runtime/fixture/quality/browser invalidation conditions are documented for reruns.

## Release gates

- Module worker syntax check and public-worker mirror comparison.
- `npm run typecheck` and `npm run build`.
- `npm run lint --prefix test-app` and `npm run build --prefix test-app` (lint retains one unrelated pre-existing image warning; zero errors).
- `npm run benchmark:minilm --prefix test-app`, the six-case evidence rerun, and `node test-app/scripts/finalize-minilm-benchmark.mjs`.
- `npm run audit:minilm --prefix test-app`.
- JSON validation, advertised-support/matrix/embedded-result comparison, and local/public fixture checksum validation.
- `git diff --check` and `npm pack --dry-run --json`.

Final status: **v4 contract: conformant**. No failed or untested precision/device pair is advertised.
