# BGE Reranker, DETR and VitMatte onboarding audit

Audit date: 2026-09-15. All three workers were compared with WORKER_CONTRACT.md and the Whisper Tiny adapter.

Verified: boot handshake and legacy capabilities; manifest version 1.0; separate original-provider and artifact provenance; provider/license URLs; exact Transformers.js 4.2.0; input/config/output documentation; correlated requests, errors and download progress; one message handler; standard generated result shape; explicit no-stream capabilities; adapter-local inference; disposal and model-specific cache cleanup in every benchmark case; exact artifact weights and complete estimated download bytes; performance and quality results; benchmark-passing-only device allowlists.

| Worker | Recommended configuration | Matrix | Public API |
| --- | --- | --- | --- |
| bge-reranker-base | q8 / wasm | 12 of 16 passed | Full public API audit passed, 230 ms inference |
| detr-resnet-50 | q4 / webgpu | 12 of 14 passed | Full public API audit passed, 263 ms inference |
| vitmatte-small-composition-1k | fp32 / webgpu | 4 of 4 passed | Full public API audit passed, 512 ms inference |

BGE FP16 and Q4F16 fail WASM initialization and fail the ranking quality gate on WebGPU. DETR FP16 and Q4F16 fail WASM initialization. All failures remain in reports and are excluded from advertised support. VitMatte's quantized and WASM cases passed but were very slow on this machine. Timings are fixture-specific, not universal performance claims.

Release checks passed: syntax checks for all three workers; `npm run typecheck`; `npm run build`; `npm run lint --prefix test-app` (one pre-existing image warning); `npm run build --prefix test-app`; benchmark JSON parsing and cleanup/allowlist validation; `git diff --check`; `npm pack --dry-run --json --cache /tmp/webai-npm-cache`; full BGE, DETR and VitMatte benchmark matrices; `node test-app/scripts/audit-model-public-api.mjs`.

Protocol/manifest contract checks pass. The VitMatte reference mask and deterministic trimap were published to the public B2 fixture prefix, verified with HTTP 200 responses, and recorded with their exact byte sizes and SHA-1 checksums in the worker and report. Its refreshed matrix consumes the checked-in immutable trimap. The quality result remains explicitly described as a MODNet-reference regression test, not independently labeled accuracy evidence.

The public audit covers manifest discovery, download progress, recommended inference, native output and standard runtime envelope, invalid-input rejection, unsupported-stream rejection, clear-memory behavior, reinitialization, cache deletion, and preservation of another model's cache. DETR and VitMatte weights persisted and were detected. BGE inference passed, but its smallest tested weight artifact (about 279 MB) was not retained by the isolated Chrome Cache API profile; the worker manifest and audit report explicitly disclose that affected sessions will redownload it.

Final status for all three workers: **v4 contract: conformant**.
