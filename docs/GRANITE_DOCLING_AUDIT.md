# Granite Docling 258M audit

Pinned Transformers.js 4.2.0 module-worker adapter; IBM source and ONNX artifact revisions recorded separately. Complete families: FP32, FP16, Q8, Q4, Q4F16. Partial INT8/UINT8/BNB4 families are not complete decoder configurations.

The ten-case matrix records three quality passes: FP32/WebGPU and Q4 on WASM/WebGPU. FP32/WASM fails with memory allocation; FP16/WASM fails graph creation; FP16/WebGPU, Q8 on both backends, and Q4F16 on both backends fail output quality or initialization. Failed cases remain in the full report.

The benchmark checks a bounded 256-token prefix of a real pinned document image, expected header token recall >=90%, coordinates and DocTags, and repetition rejection. It does not establish full-page table/formula accuracy. Q4 has a header OCR substitution and should be treated as accuracy-degraded. Verified recommendation: Q4/WebGPU (406,711,456 estimated complete download bytes; 5.71 seconds in the public API audit).

Both Q4 backends passed the public API/stream audit: exact streaming equality, full external-weight cache detection, model-scoped deletion, preservation of another cache, invalid-option rejection, disposal and cached reinitialization. Q4/WASM is very slow (173.3 seconds). FP32/WebGPU passes inference but fails full cache lifecycle verification because Chrome cannot put its large external weight responses into Cache API; it is not advertised. Only Q4/WASM and Q4/WebGPU are supported. V4 contract audit: conformant, with the documented bounded-prefix quality limitation.
