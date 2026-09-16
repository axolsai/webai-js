import {
  env,
  pipeline,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0";
// Model identity and immutable artifact revision.
const ID = "paraphrase-multilingual-minilm-l12-v2";
const REPO = "Xenova/paraphrase-multilingual-MiniLM-L12-v2";
const REV = "2c4055b12046f11709e9df2c122e59ffbdc2f900";

// All discovered artifacts. SUPPORTED only contains benchmarked combinations.
const PRECISIONS = {
  q4f16: {
    weightsSize: 204777707,
    size: 236625303,
    modelKeys: ["model_q4f16.onnx"],
    dtype: "q4f16",
    candidateDevices: ["wasm", "webgpu"],
  },
  q4: {
    weightsSize: 398649249,
    size: 430496845,
    modelKeys: ["model_q4.onnx"],
    dtype: "q4",
    candidateDevices: ["wasm", "webgpu"],
  },
  q8: {
    weightsSize: 118308126,
    size: 150155722,
    modelKeys: ["model_quantized.onnx"],
    dtype: "q8",
    candidateDevices: ["wasm", "webgpu"],
  },
  fp16: {
    weightsSize: 235336673,
    size: 267184269,
    modelKeys: ["model_fp16.onnx"],
    dtype: "fp16",
    candidateDevices: ["wasm", "webgpu"],
  },
  fp32: {
    weightsSize: 470268510,
    size: 502116106,
    modelKeys: ["model.onnx"],
    dtype: "fp32",
    candidateDevices: ["wasm", "webgpu"],
  },
  int8: { weightsSize: 118054609, size: 149902205, modelKeys: ["model_int8.onnx"], dtype: "int8", candidateDevices: ["wasm", "webgpu"] },
  uint8: { weightsSize: 118054642, size: 149902238, modelKeys: ["model_uint8.onnx"], dtype: "uint8", candidateDevices: ["wasm", "webgpu"] },
  bnb4: { weightsSize: 397322601, size: 429170197, modelKeys: ["model_bnb4.onnx"], dtype: "bnb4", candidateDevices: ["wasm", "webgpu"] },
};
const SUPPORTED = {
  q4f16: { supportedDevices: ["webgpu"], ...PRECISIONS.q4f16 },
  q4: { supportedDevices: ["wasm", "webgpu"], ...PRECISIONS.q4 },
  q8: { supportedDevices: ["wasm", "webgpu"], ...PRECISIONS.q8 },
  fp16: { supportedDevices: ["webgpu"], ...PRECISIONS.fp16 },
  fp32: { supportedDevices: ["wasm", "webgpu"], ...PRECISIONS.fp32 },
  int8: { supportedDevices: ["wasm", "webgpu"], ...PRECISIONS.int8 },
  uint8: { supportedDevices: ["wasm", "webgpu"], ...PRECISIONS.uint8 },
  bnb4: { supportedDevices: ["wasm", "webgpu"], ...PRECISIONS.bnb4 },
};
const MANIFEST = {
  contractVersion: "1.0",
  model: {
    id: ID,
    displayName: "Paraphrase Multilingual MiniLM L12 v2",
    provider: "Sentence Transformers",
    providerUrl: "https://www.sbert.net/",
    license: "Apache-2.0",
    licenseUrl: "https://www.apache.org/licenses/LICENSE-2.0.txt",
    lastUpdated: "2026-01-28T10:02:26.000Z",
    sourceRepository: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    repository: REPO,
    artifactRevision: REV,
    artifactLastUpdated: "2025-07-22T00:07:45.000Z",
    task: "feature-extraction",
    description:
      "A multilingual sentence-transformer that produces normalized 384-dimensional embeddings for paraphrase detection, semantic similarity, clustering, and retrieval in the browser.",
    intendedUses: [
      "Multilingual and cross-lingual semantic search and retrieval.",
      "Clustering, similarity, and classification prototypes.",
    ],
    limitations: [
      "Similarity quality varies by language, domain, sentence length, and input quality across the model's 50-language training coverage.",
      "Semantic similarity is not factual verification and can encode training-data bias.",
      "Long inputs are truncated and should be chunked.",
      "Quantization can change ranking margins.",
    ],
  },
  runtime: {
    engine: { name: "Transformers.js", version: "4.2.0" },
    precisions: SUPPORTED,
  },
  input: {
    description: "One or more non-empty strings.",
    alternatives: [{ name: "texts", type: "string[]", required: true }],
  },
  config: {
    model: [
      {
        name: "pooling",
        type: "string",
        default: "mean",
        enum: ["mean"],
      },
      { name: "normalize", type: "boolean", default: true },
    ],
    generation: [],
  },
  output: {
    description: "One normalized 384-dimensional embedding per input.",
    fields: [{ name: "result", type: "number[][]" }],
    example: { result: [[0.01, -0.02]] },
  },
  operations: [
    "checkModelSupports",
    "init",
    "download",
    "generate",
    "clearMemory",
  ],
  benchmark: {
    report: "benchmarks/paraphrase-multilingual-minilm-l12-v2.json",
    testedAt: "2026-09-10T17:01:39.383Z",
    fixture: {
      file: "paraphrase-multilingual-minilm-l12-v2.json",
      url: "https://f005.backblazeb2.com/file/hubters-webai-public/assets/benchmarks/paraphrase-multilingual-minilm-l12-v2.json",
      size: 523,
      sha1: "0294528b8c4ca8006eadb7a4872cad2a5ce29416",
    },
    candidates: PRECISIONS,
    qualityGate:
      "Finite normalized 384d vectors, distinct rows, and cross-lingual paraphrase cosine exceeds an unrelated sentence by at least 0.1.",
    invalidatedBy: [
      "artifact revision change",
      "runtime version change",
      "fixture or quality gate change",
      "browser/backend change",
    ],
    results: [
      { precision: "q4f16", device: "wasm", status: "fail", failure: "ONNX session graph error" },
      { precision: "q4f16", device: "webgpu", status: "pass", runtime: { embeddingsPerSecond: 13.889, grade: "good" }, quality: { positive: 0.9634, negative: 0.0164, margin: 0.947 } },
      { precision: "q4", device: "wasm", status: "pass", runtime: { embeddingsPerSecond: 32.223, grade: "excellent" }, quality: { positive: 0.9633, negative: 0.0164, margin: 0.9469 } },
      { precision: "q4", device: "webgpu", status: "pass", runtime: { embeddingsPerSecond: 11.304, grade: "good" }, quality: { positive: 0.9633, negative: 0.0164, margin: 0.9469 } },
      { precision: "q8", device: "wasm", status: "pass", runtime: { embeddingsPerSecond: 37.975, grade: "excellent" }, quality: { positive: 0.96, negative: 0.0196, margin: 0.9405 } },
      { precision: "q8", device: "webgpu", status: "pass", runtime: { embeddingsPerSecond: 22.642, grade: "excellent" }, quality: { positive: 0.9598, negative: 0.0196, margin: 0.9402 } },
      { precision: "fp16", device: "wasm", status: "fail", failure: "ONNX session graph error" },
      { precision: "fp16", device: "webgpu", status: "pass", runtime: { embeddingsPerSecond: 36.32, grade: "excellent" }, quality: { positive: 0.9615, negative: 0.0106, margin: 0.9508 } },
      { precision: "fp32", device: "wasm", status: "pass", runtime: { embeddingsPerSecond: 33.937, grade: "excellent" }, quality: { positive: 0.9616, negative: 0.0105, margin: 0.9511 } },
      { precision: "fp32", device: "webgpu", status: "pass", runtime: { embeddingsPerSecond: 22.157, grade: "excellent" }, quality: { positive: 0.9616, negative: 0.0105, margin: 0.9511 } },
      { precision: "int8", device: "wasm", status: "pass", runtime: { embeddingsPerSecond: 37.879, grade: "excellent" }, quality: { positive: 0.9671, negative: 0.0156, margin: 0.9515 } },
      { precision: "int8", device: "webgpu", status: "pass", runtime: { embeddingsPerSecond: 20.979, grade: "excellent" }, quality: { positive: 0.9673, negative: 0.0149, margin: 0.9524 } },
      { precision: "uint8", device: "wasm", status: "pass", runtime: { embeddingsPerSecond: 39.37, grade: "excellent" }, quality: { positive: 0.9625, negative: -0.0003, margin: 0.9627 } },
      { precision: "uint8", device: "webgpu", status: "pass", runtime: { embeddingsPerSecond: 20.175, grade: "excellent" }, quality: { positive: 0.9615, negative: -0.0023, margin: 0.9638 } },
      { precision: "bnb4", device: "wasm", status: "pass", runtime: { embeddingsPerSecond: 37.879, grade: "excellent" }, quality: { positive: 0.9633, negative: 0.0236, margin: 0.9397 } },
      { precision: "bnb4", device: "webgpu", status: "pass", runtime: { embeddingsPerSecond: 13.375, grade: "good" }, quality: { positive: 0.9633, negative: 0.0236, margin: 0.9397 } },
    ],
  },
};
const state = { initializing: false, initialized: false, generating: false };
let extractor = null;
const caps = {
  supportedModes: ["webai"],
  supportedPrecisions: Object.keys(SUPPORTED),
  supportedPrecisionsDevicesMap: SUPPORTED,
  doesSupportStreamGeneration: false,
  externalInterrupt: true,
  workerVersion: "v4",
  transformersJsVersion: "4.2.0",
  cacheModelId: REPO,
};
const send = (requestId, type, data) =>
  self.postMessage({ requestId, type, data });
function configure(c = {}) {
  env.allowRemoteModels = c.allowRemoteModels ?? true;
  env.allowLocalModels = c.allowLocalModels ?? false;
  env.useBrowserCache = c.useBrowserCache ?? true;
  if (c.remoteHost) env.remoteHost = c.remoteHost;
  if (c.remotePathTemplate) env.remotePathTemplate = c.remotePathTemplate;
  if (c.localModelPath) env.localModelPath = c.localModelPath;
}
function validateInit(data) {
  const p = data?.precision,
    d = data?.device,
    s = SUPPORTED[p];
  if (!s) throw Error(`Unsupported precision: ${p}`);
  if (!s.supportedDevices.includes(d))
    throw Error(
      `Precision ${p} is only supported on: ${s.supportedDevices.join(", ")}`,
    );
  if (d === "webgpu" && !self.navigator?.gpu)
    throw Error("WebGPU is unavailable");
  return { p, d };
}
async function clear() {
  if (extractor?.dispose) await extractor.dispose();
  extractor = null;
  state.initialized = false;
}
async function load(data, id) {
  const { p, d } = validateInit(data);
  await clear();
  extractor = await pipeline("feature-extraction", REPO, {
    revision: REV,
    dtype: p,
    device: d,
    progress_callback: (v) => send(id, "downloadProgress", v),
  });
}
async function generate(data) {
  const texts = data?.userInput?.texts;
  if (
    !Array.isArray(texts) ||
    !texts.length ||
    texts.some((t) => typeof t !== "string" || !t.trim())
  )
    throw Error(
      "userInput.texts must be a non-empty array of non-empty strings",
    );
  const c = { pooling: "mean", normalize: true, ...data.modelConfig };
  if (c.pooling !== "mean") throw Error("pooling must be mean");
  if (typeof c.normalize !== "boolean")
    throw Error("normalize must be boolean");
  const tensor = await extractor(texts, c);
  return { result: tensor.tolist() };
}
self.addEventListener(
  "message",
  async ({ data: { requestId, type, data } = {} }) => {
    try {
      if (type === "checkModelSupports") {
        configure(data?.workerConfig);
        send(requestId, type, { ...caps, manifest: MANIFEST });
        return;
      }
      if (type === "download") {
        if (state.initializing || state.generating)
          throw Error("Worker is busy");
        const supported = SUPPORTED[data?.precision];
        if (!supported) throw Error(`Unsupported precision: ${data?.precision}`);
        const device = supported.supportedDevices.includes("wasm") ? "wasm" : supported.supportedDevices[0];
        await load({ ...data, device }, requestId);
        await clear();
        send(requestId, type, { status: "success" });
        return;
      }
      if (type === "init") {
        if (state.initializing || state.generating)
          throw Error("Worker is busy");
        state.initializing = true;
        try {
          await load(data, requestId);
          state.initialized = true;
          send(requestId, type, { status: "success" });
        } finally {
          state.initializing = false;
        }
        return;
      }
      if (type === "generate") {
        if (!state.initialized || !extractor)
          throw Error("Call init before generate");
        if (state.generating) throw Error("Generation already in progress");
        state.generating = true;
        try {
          send(requestId, "generated", {
            status: "success",
            result: await generate(data),
          });
        } finally {
          state.generating = false;
        }
        return;
      }
      if (type === "generateStream")
        throw Error("Stream generation is not supported for embedding models");
      if (type === "clearMemory") {
        if (state.generating)
          throw Error("Cannot clear memory while generation is active");
        await clear();
        send(requestId, type, { status: "success" });
        return;
      }
      throw Error(`Unknown operation: ${type}`);
    } catch (e) {
      send(requestId, "error", {
        message: e instanceof Error ? e.message : String(e),
        context: type,
      });
    }
  },
);
self.postMessage({
  type: "worker initialized",
  data: { success: true, workerVersion: "v4", modelId: ID },
});
