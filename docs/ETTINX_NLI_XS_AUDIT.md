# EttinX NLI XS v4 contract audit

Audit date: 2026-09-15.

**Final status: v4 contract: conformant.** Recommended configuration: **Q8/WASM**.

The worker and manifest were checked against `docs/WORKER_CONTRACT.md` and Whisper Tiny. They include correlated boot/handshake and operations, one error boundary, exact Transformers.js 4.2.0, separate immutable source/artifact provenance, authoritative MIT license and provider URLs, typed pair input/config/output, exact weight and complete download sizes, validation, progress, softmax-normalized native output, explicit non-stream support, and disposal.

The full matrix tested all eight repository artifacts on WASM and WebGPU. Twelve of sixteen passed. FP16 and Q4F16 failed both backends because the exported graph has an invalid float16 cast and are excluded. An initial Q4/WASM network failure was rerun successfully and both attempts are identified in the report evidence. Q8/WASM provides an approximately 36 MB first download and excellent measured throughput.

The three-pair semantic fixture is public and local, 930 bytes, SHA-1 `18c133130582f80cfa13ca3aa915e686f9dda37c`. Its expected labels and probability margins passed. This is a focused semantic regression gate, not a replacement for full MNLI/SNLI evaluation.

The public WebAI Q8/WASM audit passed manifest discovery, download/progress and cache detection, inference quality and standard envelope, invalid-input and streaming rejection, clear-memory and cached reinitialization, model-only cache deletion, and preservation of another model's cache.
