# SegFormer B3 ADE20K v4 audit

**Final status: v4 contract: conformant.** Recommended configuration: **Q4F16/WebGPU**.

The full isolated-browser matrix tested all eight repository artifacts—FP32, FP16, INT8, UINT8, Q8, Q4, Q4F16, and BNB4—on WASM and WebGPU. All sixteen combinations completed inference, passed the checked-in semantic-mask quality gate, disposed the pipeline, and cleared only matching cache entries. Q4F16/WebGPU is recommended because it has the smallest complete first download (47,929,465 bytes) and the fastest measured inference (947.2 ms).

The standard fixture is the existing public and local Pexels portrait: 78,124 bytes, SHA-1 `38212739a4a4f0be25d84863cf991d7fb6c6d818`. The FP32 result identifies building, sky, and person with a complete non-overlapping mask assignment. This one-image regression gate detects output degradation but is not an ADE20K-wide accuracy evaluation.

The public WebAI Q4F16/WebGPU audit passed manifest discovery, progress and cache detection, native typed-mask output and the standard result envelope, invalid-input and unsupported-stream rejection, clear-memory and cached reinitialization, model-only cache deletion, and preservation of another model's cache.

The manifest identifies the source and browser artifact separately. The source is governed by the NVIDIA SegFormer non-commercial research/evaluation license; the worker uses `LicenseRef-NVIDIA-SegFormer-NonCommercial`, links the authoritative license, and describes the restriction prominently.
