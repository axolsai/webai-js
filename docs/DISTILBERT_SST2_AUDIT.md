# DistilBERT SST-2 English sentiment — final v4 audit

Worker: `model-workers-v4/distilbert-base-uncased-finetuned-sst-2-english.worker.js`.
Test UI: `/sentiment` (download first, then inference; raw input and output JSON).

## Contract review

Compared with every requirement in `WORKER_CONTRACT.md`, the template, and Whisper Tiny. The module-worker handshake, legacy capabilities and manifest discovery, single correlated handler/error boundary, standard result envelope, progress, required operations, validation, replaced-model disposal, and clear-memory lifecycle are preserved. Streaming is not advertised and the public API rejects it. There is no telemetry or infrastructure change.

Hugging Face's original source is separately identified from Xenova's ONNX artifact, with revisions, timestamps, original provider URL and authoritative Apache-2.0 license URL. The artifact revision is pinned to `0b6928efcb76139cae2c6881d49cda67fe119f42`; Transformers.js is pinned to 4.2.0 and its import was proven in a module worker before implementation. The actual configuration confirms NEGATIVE/POSITIVE label ordering and a 512-token limit. Metadata describes binary sentiment, intended uses, English/movie-review domain, truncation, neutral/sarcasm limitations and upstream bias/safety concerns. Config defaults are derived from the manifest schema. Output documents ranked softmax probabilities for both classes.

## Evidence

`benchmarks/distilbert-sst2.json` retains the complete 16-case matrix with expected-label accuracy/confidence checks, finite normalized probabilities, native output, errors, load/inference/total timing, texts/second and performance grades, disposal and model-scoped cache deletion counts. Fourteen cases pass. FP16/WASM and Q4F16/WASM fail ONNX session creation and are excluded from runtime support. Both pass on WebGPU; the other six precisions pass both devices. All sixteen compact results are embedded in the manifest.

Recommended public-API configuration: **INT8/WASM**, `max_length:512`. Exact weights: 67,370,466 bytes; estimated complete first download: 68,314,602 bytes (shared assets: 944,136 bytes). Public API produced POSITIVE in 43.7 ms with confidence 0.99977 and correctly classified the negative fixture at 0.99952. This is a small smoke correctness gate, not broad evaluation or universal performance evidence. Q8/WebGPU is valid but graded slow on this machine.

`benchmarks/distilbert-sst2-public-api.json` verifies manifest discovery, download progress, cache detection/reinitialization, positive and negative inference, standard runtime envelope, invalid input/config rejection, unsupported streaming, clear-memory behavior, and scoped cache deletion preserving another model's sentinel. The local/public B2 fixture is 486 bytes with SHA1 `226199feb31ba0ad6ffa6629585a76930b240a3a`; both copies match. Fixture source/license/preprocessing and benchmark invalidation conditions are documented.

## Passed release checks

- Worker module syntax check.
- `npm run typecheck` and `npm run build`.
- `npm run lint --prefix test-app` and `npm run build --prefix test-app` (one unrelated existing image warning, zero errors).
- Full `benchmark-vision-adapter.mjs` 16-case run, also available as `npm run benchmark:sentiment --prefix test-app`.
- `npm run audit:sentiment --prefix test-app`.
- `node test-app/scripts/validate-distilbert-sst2.mjs` (JSON, field metadata, embedded results, supported-pair equivalence, cleanup, public/local checksum and mirror comparison).
- `git diff --check` and `npm pack --dry-run --json`; package includes the worker and benchmark reports.

Final status: **v4 contract: conformant**. Only measured, quality-passing precision/device pairs are advertised.
