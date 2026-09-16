import {
  env,
  pipeline,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0";
// Model identity and immutable artifact revision.
const ID = "bge-m3";
const REPO = "Xenova/bge-m3";
const REV = "4de13258303883538bd53b696b452bf8099f0858";

// All discovered artifacts. SUPPORTED only contains benchmarked combinations.
const PRECISIONS = {
  q4f16: {
    weightsSize: 699932840,
    size: 722088453,
    modelKeys: ["model_q4f16.onnx"],
    dtype: "q4f16",
    candidateDevices: ["wasm", "webgpu"],
  },
  q4: {
    weightsSize: 1248237611,
    size: 1270393224,
    modelKeys: ["model_q4.onnx"],
    dtype: "q4",
    candidateDevices: ["wasm", "webgpu"],
  },
  q8: {
    weightsSize: 569694530,
    size: 591850143,
    modelKeys: ["model_quantized.onnx"],
    dtype: "q8",
    candidateDevices: ["wasm", "webgpu"],
  },
  fp16: {
    weightsSize: 1133992936,
    size: 1156148549,
    modelKeys: ["model_fp16.onnx"],
    dtype: "fp16",
    candidateDevices: ["wasm", "webgpu"],
  },
  fp32: {
    weightsSize: 2267427906,
    size: 2289583519,
    modelKeys: ["model.onnx"],
    dtype: "fp32",
    candidateDevices: ["wasm", "webgpu"],
  },
  int8: {
    weightsSize: 568456694,
    size: 590612307,
    modelKeys: ["model_int8.onnx"],
    dtype: "int8",
    candidateDevices: ["wasm", "webgpu"],
  },
  uint8: {
    weightsSize: 568456765,
    size: 590612378,
    modelKeys: ["model_uint8.onnx"],
    dtype: "uint8",
    candidateDevices: ["wasm", "webgpu"],
  },
  bnb4: {
    weightsSize: 1229364419,
    size: 1251520032,
    modelKeys: ["model_bnb4.onnx"],
    dtype: "bnb4",
    candidateDevices: ["wasm", "webgpu"],
  },
};
// Only combinations that passed loading, inference, and semantic quality checks.
const SUPPORTED = {
  q8: { supportedDevices: ["wasm", "webgpu"], ...PRECISIONS.q8 },
  int8: { supportedDevices: ["wasm", "webgpu"], ...PRECISIONS.int8 },
  uint8: { supportedDevices: ["wasm", "webgpu"], ...PRECISIONS.uint8 },
  q4f16: { supportedDevices: ["webgpu"], ...PRECISIONS.q4f16 },
  q4: { supportedDevices: ["wasm"], ...PRECISIONS.q4 },
  bnb4: { supportedDevices: ["wasm"], ...PRECISIONS.bnb4 },
  fp16: { supportedDevices: ["webgpu"], ...PRECISIONS.fp16 },
};
const MANIFEST = {
  contractVersion: "1.0",
  model: {
    id: ID,
    displayName: "BGE-M3 — Multilingual Embeddings",
    provider: "BAAI",
    providerUrl: "https://www.baai.ac.cn/",
    license: "MIT",
    licenseUrl: "https://opensource.org/license/mit/",
    lastUpdated: "2024-07-03T14:50:10.000Z",
    sourceRepository: "BAAI/bge-m3",
    repository: REPO,
    artifactRevision: REV,
    artifactLastUpdated: "2026-02-10T19:47:18.000Z",
    task: "feature-extraction",
    description:
      "A multilingual and multi-function retrieval model that produces normalized 1024-dimensional dense text embeddings in the browser.",
    intendedUses: [
      "Multilingual and cross-lingual dense semantic search and retrieval.",
      "Clustering, similarity, and classification prototypes.",
    ],
    limitations: [
      "Retrieval quality depends on language, domain, and input quality.",
      "This adapter exposes dense embeddings only; BGE-M3 lexical weights and multi-vector interaction outputs are not exposed.",
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
        default: "cls",
        enum: ["cls"],
      },
      { name: "normalize", type: "boolean", default: true },
    ],
    generation: [],
  },
  output: {
    description: "One normalized 1024-dimensional embedding per input.",
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
    report: "benchmarks/bge-m3.json",
    testedAt: "2026-09-10T10:58:00.000Z",
    fixture: {
      file: "bge-m3.json",
      url: "https://f005.backblazeb2.com/file/hubters-webai-public/assets/benchmarks/bge-m3.json",
      size: 500,
      sha1: "75a7770d06fc312ddeb39987712a8b4bbe6f1e08",
    },
    candidates: PRECISIONS,
    qualityGate:
      "Finite normalized 1024d vectors, distinct rows, and positive cosine exceeds negative by at least 0.1.",
    invalidatedBy: [
      "artifact revision change",
      "runtime version change",
      "fixture or quality gate change",
      "browser/backend change",
    ],
    results: [
      { precision: "q8", device: "wasm", status: "pass", runtime: { embeddingsPerSecond: 4.486, grade: "slow" } },
      { precision: "q8", device: "webgpu", status: "pass", runtime: { embeddingsPerSecond: 3.874, grade: "slow" } },
      { precision: "int8", device: "wasm", status: "pass", runtime: { embeddingsPerSecond: 4.496, grade: "slow" } },
      { precision: "int8", device: "webgpu", status: "pass", runtime: { embeddingsPerSecond: 3.566, grade: "slow" } },
      { precision: "uint8", device: "wasm", status: "pass", runtime: { embeddingsPerSecond: 4.416, grade: "slow" } },
      { precision: "uint8", device: "webgpu", status: "pass", runtime: { embeddingsPerSecond: 3.868, grade: "slow" } },
      { precision: "q4f16", device: "wasm", status: "fail", failure: "ONNX session graph error" },
      { precision: "q4f16", device: "webgpu", status: "pass", runtime: { embeddingsPerSecond: 5.848, grade: "slow" } },
      { precision: "q4", device: "wasm", status: "pass", runtime: { embeddingsPerSecond: 3.01, grade: "slow" } },
      { precision: "q4", device: "webgpu", status: "fail", failure: "WebGPU table index out of bounds" },
      { precision: "bnb4", device: "wasm", status: "pass", runtime: { embeddingsPerSecond: 3.743, grade: "slow" } },
      { precision: "bnb4", device: "webgpu", status: "fail", failure: "WebGPU table index out of bounds" },
      { precision: "fp16", device: "wasm", status: "fail", failure: "ONNX session graph error" },
      { precision: "fp16", device: "webgpu", status: "pass", runtime: { embeddingsPerSecond: 12.107, grade: "good" } },
      { precision: "fp32", device: "wasm", status: "fail", failure: "bounded browser timeout" },
      { precision: "fp32", device: "webgpu", status: "fail", failure: "bounded browser timeout" },
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
  const c = { pooling: "cls", normalize: true, ...data.modelConfig };
  if (c.pooling !== "cls") throw Error("pooling must be cls");
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
        await load(data, requestId);
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
