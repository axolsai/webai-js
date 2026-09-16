# Moonshine Base English — final v4 audit

Worker: `model-workers-v4/moonshine-base-en.worker.js`; playground: `/moonshine`.
The WebAI ID preserves the requested English suffix; the browser artifact is `onnx-community/moonshine-base-ONNX`.

## Contract review

Compared field-by-field with `WORKER_CONTRACT.md`, the worker template and Whisper Tiny's implementation. The worker preserves boot handshake, legacy capabilities plus manifest, one correlated message handler/error boundary, required operations, correlated progress/errors and standard generated result/runtime envelope. It validates typed finite 16 kHz mono audio and configuration before inference, rejects unknown configuration, disposes replaced pipelines and download-only instances, and propagates disposal/load errors. Busy lifecycle checks cover initialization, inference and clear-memory. It does not advertise streaming, timestamps or diarization. No telemetry or manager/queue changes were added.

Original Useful Sensors/Moonshine AI source/provider, authoritative MIT license URL, source revision/date and ONNX repository/revision/date are separate. The artifact is pinned to `b1e9b6aae3c3c7298f10c3798393fdf38e8fbbad`. Transformers.js 4.2.0 initialization was proven in a module worker before implementation. Manifest descriptions cover English-only use, audio-quality/accuracy failures, silence/long-recording hallucinations and safety limitations. Model defaults derive from the co-located schema. Actual encoder and merged-decoder artifacts determine all eight candidate dtype families and their exact weight bytes; shared assets total 3,898,689 bytes.

## Matrix and public API

`benchmarks/moonshine-base-en.json` retains all 16 isolated-browser cases, native transcripts, complete failures, WER/repetition quality checks, load/inference/total timing, real-time factor and performance grade, model disposal and repository-scoped cache deletion counts. All sixteen compact cases are embedded in the manifest; advertised runtime support exactly matches passes.

Nine cases pass: FP32, Q4 and BNB4 on WASM/WebGPU, plus INT8, UINT8 and Q8 on WebGPU. Seven fail initialization and are excluded: FP16 and Q4F16 on both devices; INT8, UINT8 and Q8 on WASM. FP16/Q4F16 expose invalid graphs/subgraphs; 8-bit WASM variants lack required quantization scales. Failures remain in the report rather than being silently retried under another dtype.

Recommended configuration: **Q4/WebGPU**, `max_new_tokens:256`. Exact weights: 97,537,620 bytes; estimated complete first download: 101,436,309 bytes. The standard 29.49225-second recording ran in 859.1 ms in the matrix (RTF 0.0291, excellent), with WER 0.1212 and no repetition degeneration. These figures describe one browser/hardware and a small regression fixture, not general ASR accuracy guarantees.

`benchmarks/moonshine-base-en-public-api.json` verifies the recommendation through public WebAI APIs: manifest discovery, correlated download progress, cached-state detection/reinitialization, native transcript and runtime envelope, invalid input/config rejection, unsupported streaming rejection, clear memory and model-scoped deletion preserving another model's cache. API inference took 843.4 ms with the same WER.

The existing project standard Whisper fixture/reference is reused unchanged, not newly generated. Its local and stable public B2 MP3 copies match: 471,925 bytes, SHA1 `5faad1507b5d6bc61a39ca3184db5a7b554a7479`. The local MP3/reference are retained for offline reproduction; preprocessing and artifact/runtime/quality/fixture/backend invalidation conditions are recorded.

## Passed release gates

- `node --input-type=module --check < model-workers-v4/moonshine-base-en.worker.js`.
- `npm run typecheck` and `npm run build`.
- `npm run lint --prefix test-app` and `npm run build --prefix test-app` (one unrelated pre-existing image warning, zero errors).
- Full 16-case `benchmark-vision-adapter.mjs` run, reproducible with `npm run benchmark:moonshine-base --prefix test-app`.
- `npm run audit:moonshine-base --prefix test-app`.
- JSON, exact sizes, 16 embedded-case comparisons, advertised-pass equivalence, cleanup evidence, worker mirror and local/public fixture checksum checks.
- `git diff --check` and `npm pack --dry-run --json`; the package contains the worker and full/API reports.

Final status: **v4 contract: conformant**. Recommended Q4/WebGPU is verified; tested failures are not advertised.
