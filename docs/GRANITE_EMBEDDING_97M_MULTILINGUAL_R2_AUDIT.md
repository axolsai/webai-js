# Granite Embedding 97M Multilingual R2 v4 audit

The worker pins the IBM source and ONNX Community artifact independently, records Apache-2.0 provenance, exact weight and complete-download sizes, Transformers.js 4.2.0, the public benchmark fixture, limitations, and tested browser support.

All eight repository variants were tested on WASM and WebGPU. Twelve of sixteen cases passed the multilingual semantic-quality gate. FP16 and Q4F16 fail ONNX session creation on both devices; those failures remain in the report and those precisions are not advertised. The verified recommendation is **INT8/WASM**: 123,174,716 bytes estimated complete first download, 97.8 ms for three fixture embeddings, and 30.67 embeddings/second.
