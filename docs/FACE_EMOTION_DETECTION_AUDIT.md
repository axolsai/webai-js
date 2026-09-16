# Face Emotion Detection v4 audit

The worker follows the v4 lifecycle and result contract and runs `onnx-community/face-emotion-detection-ONNX` through Transformers.js 4.2.0. The manifest separately pins the Apache-2.0 source and ONNX artifact provenance, exact artifact sizes, fixture provenance, limitations, and measured browser support.

The isolated browser matrix tested all eight repository artifacts on WASM and WebGPU. Fourteen of sixteen combinations passed the seven-label consistency gate. `int8/wasm` and the equivalent `q8/wasm` retained the reference top label but failed the minimum probability-margin gate, remain recorded in the benchmark report, and are not advertised. The verified recommendation is **q4f16/WebGPU** (49,785,624-byte complete first download; 96.3 ms measured inference; 10.38 images/second).

Facial expression classification does not reveal a person's internal emotion, intent, honesty, or mental health. The worker and test UI explicitly prohibit consequential uses.
