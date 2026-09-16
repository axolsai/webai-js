# RuBERT Tiny Toxicity v4 audit

**Final status: v4 contract: conformant.** Recommended configuration: **Q8/WASM**.

The artifact inventory found six ONNX variants. The full isolated-browser matrix tested FP32, INT8, UINT8, Q8, Q4, and BNB4 on both WASM and WebGPU. All twelve combinations loaded, inferred, passed the sourced two-example regression gate, disposed their models, and cleared only matching cache entries.

The local and public fixture is 700 bytes with SHA-1 `f136212c8f17ea8b87f6c567580ddae3ad924bcc`. It checks one explicitly toxic and one safe example from the pinned source model card. This small regression gate is not a substitute for a representative Russian moderation evaluation.

The manifest records original and artifact provenance separately, the pinned Transformers.js runtime, exact model and complete-download bytes, typed input/config/output schemas, intended uses, task-specific limitations, lifecycle operations, full candidate inventory, and compact results for every tested pair. Streaming is not advertised or implemented.

The public WebAI Q8/WASM audit passed manifest discovery, progress and cache detection, real inference and the standard result envelope, invalid-input and unsupported-stream rejection, clear-memory and cached reinitialization, model-only cache deletion, and preservation of another model's cache. The audited inference took 23.8 ms on the recorded host.
