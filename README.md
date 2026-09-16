# WebAI.js

[![npm version](https://img.shields.io/npm/v/@axols/webai-js.svg)](https://www.npmjs.com/package/@axols/webai-js)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

WebAI.js runs AI models in a browser through a consistent JavaScript API. The library manages Web Workers, downloads, caching, request queues, cancellation, and result timing. Each model worker handles its own runtime, inputs, configuration, inference, and cleanup.

The current collection is in [`model-workers-v4/`](model-workers-v4/). It includes text, speech, image, and multimodal models powered by Transformers.js, LiteRT.js, LiteRT-LM, and MediaPipe.

## What it does

- Runs supported models on the user's device using WebAssembly or WebGPU, without an inference server.
- Provides one lifecycle for different tasks: create a worker, initialize a tested configuration, generate a result, and release resources.
- Lets an app inspect each worker's manifest to discover its inputs, options, output shape, provenance, license, supported precision/device pairs, and benchmark evidence.
- Reports download progress and returns a common result envelope while preserving each model's native output.
- Supports token streaming only for workers that explicitly advertise it.

The library does **not** include model weights. Workers normally fetch their pinned artifacts from the model provider or browser-artifact repository on first use. Inputs are processed locally by the worker, but model downloads still make network requests. Browser storage, memory, WebGPU availability, and model licenses vary by model and device.

## Install and try a worker

```bash
npm install @axols/webai-js
```

Serve the worker script from your app's public directory. For example, in a Next.js app:

```bash
mkdir -p public/workers
cp node_modules/@axols/webai-js/model-workers-v4/bge-reranker-large.worker.js public/workers/
```

Then use the worker from client-side code:

```js
import { WebAI } from "@axols/webai-js";

const ai = await WebAI.create({
  modelId: "bge-reranker-large",
  workerPath: "/workers/bge-reranker-large.worker.js",
});

try {
  // The manifest is available before the model is downloaded.
  console.log(ai.modelManifest);

  await ai.init({
    precision: "int8",
    device: "wasm",
    onDownloadProgress: (event) => console.log(event),
  });

  const { result, runtime } = await ai.generate({
    userInput: {
      query: "What is the capital of France?",
      documents: [
        "Paris is the capital of France.",
        "Whales live in the ocean.",
      ],
    },
    modelConfig: { top_k: 2, normalize_scores: true },
  });

  console.log(result.results); // Ranked passages, scores, logits, and original indices
  console.log(runtime.durationMs); // Inference time in milliseconds
} finally {
  await ai.terminate();
}
```

`int8`/`wasm` is this worker's verified recommended configuration, not a universal default. Its estimated first download is about 584 MB. Before choosing a configuration for another worker, inspect `ai.modelManifest.runtime.precisions` or its report in [`benchmarks/`](benchmarks/). Only benchmark-passing precision/device pairs are advertised.

## Model collection

The v4 directory contains 49 workers. These are examples of the tasks they cover; the [worker directory](model-workers-v4/) is the source of truth for exact model IDs.

| Task | Example workers | What they return |
| --- | --- | --- |
| Chat and text generation | `qwen3.5-0.8b`, `apertus-v1.1-0.5b`, Gemma 4 LiteRT-LM workers | Generated text; streaming where implemented |
| Embeddings and retrieval | `all-minilm-l6-v2`, `bge-small-en-v1.5`, `bge-m3`, `embeddinggemma-300m`, `granite-embedding-97m-multilingual-r2` | Vectors for similarity or search |
| Reranking | `bge-reranker-base`, `bge-reranker-large` | Query-document relevance rankings |
| Speech and audio | Whisper variants, Moonshine variants, `cohere-transcribe-03-2026`, `kokoro-82m-v1`, `wav2vec2-base-superb-er` | Transcripts, synthesized audio, or audio labels |
| Text classification | `distilbert-base-uncased-finetuned-sst-2-english`, `ettinx-nli-xs`, `rubert-tiny-toxicity`, `privacy-filter` | Task-specific labels, scores, or spans |
| Image understanding | `florence-2-base-ft`, `siglip-base-patch16-224`, `detr-resnet-50`, `depth-anything-v3-small` | Captions, similarity, detections, or depth |
| Segmentation and matting | `sam-vit-base`, `sam-vit-large`, `segformer-b3-ade20k`, `modnet`, `vitmatte-small-composition-1k` | Masks or alpha mattes |
| Landmarks and detection | MediaPipe face/hand workers, `yolox-m-litert`, `face-emotion-detection` | Landmarks, boxes, or classification scores |

Each worker has its own required input and output shape. Do not assume that a chat request, an image URL, and an audio buffer are interchangeable just because they use the same `generate()` method. Read the worker's manifest before building a UI for it.

## The v4 worker contract

Every v4 worker sends a startup handshake and answers correlated requests for capability discovery, initialization, download, generation, and memory cleanup. A `contractVersion: "1.0"` manifest describes:

- The original provider, source repository, artifact repository, update dates, SPDX license, and URLs.
- Accepted inputs, model/generation options, defaults, native output, and supported operations.
- Exact model-weight bytes, estimated complete first-download bytes, and tested precision/device support.
- A fixture, quality gate, performance results, failures, and conditions that invalidate the benchmark.

WebAI wraps a successful worker result as `{ result, runtime: { durationMs } }`. The worker's `result` content stays task-specific. Unsupported streaming calls reject instead of silently pretending to stream. Call `clearMemory()` to unload a model while keeping the instance, or `terminate()` when finished. `deleteDownloadedModel()` removes cached model artifacts.

The contract and integration instructions are in [`docs/WORKER_CONTRACT.md`](docs/WORKER_CONTRACT.md) and [`docs/ADDING_MODELS.md`](docs/ADDING_MODELS.md). New workers start from [`docs/worker-template.js`](docs/worker-template.js); [`whisper-tiny.worker.js`](model-workers-v4/whisper-tiny.worker.js) is the complete Transformers.js reference.

## Benchmarks and support claims

An artifact's filename alone does not establish support. Onboarding tests actual model loading and inference in isolated browsers for each candidate precision/backend pair, checks the output against a task-specific quality gate, records timings and failures, and tests the recommended configuration through the public `WebAI` API. Reports are kept in [`benchmarks/`](benchmarks/).

Results are tied to the pinned model artifact, runtime, fixture, quality gate, browser, and backend. They are evidence for the tested environment, not a guarantee of speed or quality on every device. Large models can take substantial time and storage to download; some combinations are intentionally absent from a worker's supported list because they failed loading, inference, or quality checks.

## Development

```bash
npm run typecheck
npm run build
npm pack --dry-run
```

The local, Git-ignored `test-app/` contains task-specific playgrounds and benchmark harnesses. When adding a model, follow the full release gate and final v4 audit in [`docs/ADDING_MODELS.md`](docs/ADDING_MODELS.md).

## License

WebAI.js is licensed under [Apache 2.0](LICENSE). Individual models have their own licenses; inspect each worker's `model.license` and `model.licenseUrl` before use.

Created by Peng Zhang. [Homepage](https://www.webai-js.com) · [Discussions](https://github.com/axols/webai-js/discussions) · [Discord](https://discord.gg/RkpAKgZC)
