# BGE Small v1.5 — English and Chinese final v4 audits

Workers: `bge-small-en-v1.5.worker.js` and `bge-small-zh-v1.5.worker.js` under `model-workers-v4/`. Playground: `/bge-embeddings`, with language-specific examples, model-state reset on selection, download-first controls and raw request/result JSON.

## Contract comparison

Both workers were compared against every requirement in `WORKER_CONTRACT.md`, the worker template and Whisper Tiny. Boot handshake, manifest discovery/legacy capabilities, one correlated message handler/error boundary, all required operations, progress/errors and the public native-result/runtime envelope are preserved. Validation rejects empty/non-string texts, wrong pooling/normalization and unknown configuration options before inference. Initialization/generation are guarded, replaced and download-only pipelines are disposed, and clear-memory errors propagate. No telemetry, public manager/queue refactoring or streaming capability was added. Unsupported streaming is explicitly rejected by the public API.

Each manifest separates original BAAI identity/provider/MIT license URL and source revision/update date from Xenova's pinned artifact revision/update date. Display name, description, intended use and concrete language/truncation/accuracy/bias/safety limitations are present. All eight actual artifacts have exact weight and estimated complete-download sizes. Transformers.js 4.2.0 was proven to initialize in a module Web Worker before implementation. The original BGE source recommends normalized CLS pooling; that setting is used, rather than assuming mean pooling from a converter example. Defaults derive from co-located manifest schemas. Callers supply retrieval instructions on queries only.

English produces 384-dimensional vectors; Chinese produces 512-dimensional vectors. Both use a 512-token input limit and expose one normalized vector per input string inside the normal WebAI result envelope.

## Matrix evidence

Full reports: `benchmarks/bge-small-en-v1.5.json` and `benchmarks/bge-small-zh-v1.5.json`. Each retains all 16 isolated-browser cases, full native vectors, finite/shape/norm/distinctness checks, expected retrieval margin >=0.1, complete failure errors, raw load/inference/total duration, embeddings/second and performance grade, disposal and model-specific cold/after-case cache cleanup with deleted-entry counts. All compact cases are embedded in each manifest, and advertised precision/device pairs exactly equal measured quality passes.

Each model passes 14/16 cases. FP16/WASM and Q4F16/WASM fail ONNX session graph creation and are excluded. FP16/Q4F16 pass WebGPU; FP32, INT8, UINT8, Q8, Q4 and BNB4 pass both backends. English half-precision WebGPU variants pass this smoke gate but show substantially reduced retrieval margin relative to FP32 (about 0.156/0.169 versus 0.357); they are not recommended. Fixture correctness is not a claim of broad retrieval accuracy or interchangeable precision behavior.

## Verified recommendations and fixtures

Both recommended configurations are **INT8/WASM**, CLS pooling with L2 normalization:

- English: weights 33,760,831 bytes; complete estimated download 34,704,909 bytes, including 944,078 shared bytes. Public API: three 384d vectors in 81.1 ms, retrieval margin 0.3562.
- Chinese: weights 23,903,394 bytes; complete estimated download 24,453,267 bytes, including 549,873 shared bytes. Public API: three 512d vectors in 78.8 ms, retrieval margin 0.3886.

The WebAI-authored MIT retrieval fixtures are retained locally and uploaded to stable public B2 assets. English is 606 bytes, SHA1 `2f3cc9dccba6004ab0061011297a862c5bda1634`; Chinese is 596 bytes, SHA1 `2365b808a87dc8e3e77199ae98192506c7826988`. Source/license/preprocessing and fixture size/checksum are recorded in the manifests; local/public bytes match. Artifact/runtime/fixture/quality/backend invalidation conditions are recorded for reruns.

Public API reports, `benchmarks/bge-small-en-v1.5-public-api.json` and `benchmarks/bge-small-zh-v1.5-public-api.json`, verify manifest discovery, recommendation support, downloads/progress, cached reinitialization, native shape/normalization/quality and runtime, invalid input/config rejection, unsupported streaming, memory clearing and scoped cache deletion preserving another model's sentinel.

## Passed release gates

- Module syntax checks for both workers.
- `npm run typecheck` and `npm run build`.
- `npm run lint --prefix test-app` and `npm run build --prefix test-app` (one unrelated existing image warning; zero errors).
- Full 16-case matrices for each worker through `benchmark-embedding.mjs`, reproducible with `npm run benchmark:bge-small-en --prefix test-app` and `npm run benchmark:bge-small-zh --prefix test-app`.
- `node test-app/scripts/finalize-bge-reports.mjs`.
- `npm run audit:bge-small --prefix test-app`.
- `node test-app/scripts/validate-bge-small.mjs`: JSON/provenance metadata, 16 embedded results per model, support/quality/cleanup equivalence, exact sizes, public API evidence, mirrored workers and local/public fixture checksums.
- `git diff --check` and `npm pack --dry-run --json`; package verification includes both workers and their full/API reports.

Final status for **both workers: v4 contract: conformant**. No failed or unverified precision/device combination is advertised.
