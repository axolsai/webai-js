# BGE Reranker Large v4 audit

Status: **v4 contract conformant**.

## Identity and artifacts

- Worker ID: `bge-reranker-large`
- Original provider/source: BAAI, `BAAI/bge-reranker-large`, revision `55611d7bca2a7133960a6d3b71e083071bbfc312`
- Browser artifacts: `Xenova/bge-reranker-large`, revision `3c4ff3c9420fb24ea62acd31e3884e09c8827f2a`
- Runtime: Transformers.js `4.2.0`
- License: MIT, linked to the authoritative FlagEmbedding license text
- Recommended configuration: `int8` on `wasm`; exact weights 561,674,733 bytes and estimated first download 583,843,468 bytes
- Fixture: immutable B2 `bge-m3.json`, 500 bytes, SHA-1 `75a7770d06fc312ddeb39987712a8b4bbe6f1e08`

The adapter uses an 8 MiB chunked Cache Storage implementation because Chrome cannot retain the
561 MB ONNX response as one cache entry. It remains adapter-local. The normal model key is retained,
so existing public download detection and model-scoped deletion work without manager changes.

## Browser matrix

All 16 precision/device candidates were run in isolated Chrome 151. Passing combinations are the
only combinations advertised by the worker:

- FP16: WebGPU
- INT8: WASM and WebGPU
- UINT8: WASM and WebGPU
- Q8: WASM and WebGPU
- Q4: WASM
- Q4F16: WebGPU
- BNB4: WASM

Tested but unsupported:

- FP32/WASM and FP32/WebGPU exceeded the 240-second bounded load window.
- FP16/WASM and Q4F16/WASM failed ONNX session creation.
- Q4/WebGPU and BNB4/WebGPU failed with unaligned-access runtime errors.

Every passing case produced finite, descending scores, ranked the France passage first, exceeded
the unrelated passage by the required sigmoid margin, disposed its model, and cleared only this
model's cache. Raw load/inference timings and performance grades are in
`benchmarks/bge-reranker-large.json`.

## Public API audit

The recommended INT8/WASM configuration passed through the public `WebAI` API. The audit verified
manifest discovery, correlated download progress, cache detection, cached reinitialization,
standard result/runtime envelopes, normalized and raw scores, `top_k`, invalid input/config
rejection, unsupported streaming rejection, `clearMemory`, model-scoped deletion, and preservation
of another model's cache. The measured public inference duration was 723 ms for two pairs.

## Release gates

- Worker module syntax check: passed
- Library typecheck: passed
- Library build: passed
- Test-app lint: passed (one pre-existing `<img>` warning outside this model)
- Test-app production build: passed
- 16-case benchmark matrix and quality gates: passed with failures preserved above
- Public `WebAI` audit: passed
- Benchmark/manifest JSON parsing: passed
- `git diff --check`: passed
- `npm pack --dry-run`: passed
