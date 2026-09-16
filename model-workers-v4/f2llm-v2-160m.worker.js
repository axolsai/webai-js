import {
  env,
  pipeline,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0";
// Model identity and immutable artifact revision.
const ID = "f2llm-v2-160m";
const REPO = "onnx-community/F2LLM-v2-160M-ONNX";
const REV = "7202b3cd72a7f11b4bea90d6c41b67423281360b";

// All discovered artifacts. SUPPORTED only contains benchmarked combinations.
const PRECISIONS = {
  q4f16: {
    weightsSize: 229850384,
    size: 245734089,
    modelKeys: ["model_q4f16.onnx"],
    dtype: "q4f16",
    candidateDevices: ["wasm", "webgpu"],
  },
  q4: {
    weightsSize: 428093722,
    size: 443977427,
    modelKeys: ["model_q4.onnx"],
    dtype: "q4",
    candidateDevices: ["wasm", "webgpu"],
  },
  q8: {
    weightsSize: 159815176,
    size: 175698881,
    modelKeys: ["model_quantized.onnx"],
    dtype: "q8",
    candidateDevices: ["wasm", "webgpu"],
  },
  fp16: {
    weightsSize: 318898839,
    size: 334782544,
    modelKeys: ["model_fp16.onnx"],
    dtype: "fp16",
    candidateDevices: ["wasm", "webgpu"],
  },
  fp32: {
    weightsSize: 637103196,
    size: 652986901,
    modelKeys: ["model.onnx"],
    dtype: "fp32",
    candidateDevices: ["wasm", "webgpu"],
  },
};
const SUPPORTED = {
  q8: { supportedDevices: ["wasm", "webgpu"], ...PRECISIONS.q8 },
  fp32: { supportedDevices: ["wasm", "webgpu"], ...PRECISIONS.fp32 },
};
const MANIFEST = {
  contractVersion: "1.0",
  model: {
    id: ID,
    displayName: "F2LLM-v2 160M — Multilingual Embeddings",
    provider: "CodeFuse",
    providerUrl: "https://github.com/codefuse-ai",
    license: "Apache-2.0",
    licenseUrl: "https://www.apache.org/licenses/LICENSE-2.0.txt",
    lastUpdated: "2026-09-03T11:58:16.000Z",
    sourceRepository: "codefuse-ai/F2LLM-v2-160M",
    repository: REPO,
    artifactRevision: REV,
    artifactLastUpdated: "2026-08-27T06:44:50.000Z",
    task: "feature-extraction",
    description:
      "A compact 640-dimensional multilingual text embedding model for semantic retrieval in the browser.",
    intendedUses: [
      "Multilingual semantic search and retrieval.",
      "Clustering, similarity, and classification prototypes.",
    ],
    limitations: [
      "Retrieval quality depends on language, domain, input quality, and correct query instructions.",
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
        default: "last_token",
        enum: ["last_token"],
      },
      { name: "normalize", type: "boolean", default: true },
    ],
    generation: [],
  },
  output: {
    description: "One normalized 640-dimensional embedding per input.",
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
    report: "benchmarks/f2llm-v2-160m.json",
    testedAt: "2026-09-09T22:42:00.000Z",
    fixture: {
      file: "f2llm-v2-160m.json",
      url: "https://f005.backblazeb2.com/file/hubters-webai-public/assets/benchmarks/f2llm-v2-160m.json",
      size: 576,
      sha1: "e8a6ac52dcbb23157d108932572d2c44e173e48a",
    },
    candidates: PRECISIONS,
    qualityGate:
      "Finite normalized 640d vectors, distinct rows, and positive cosine exceeds negative by at least 0.1.",
    invalidatedBy: [
      "artifact revision change",
      "runtime version change",
      "fixture or quality gate change",
      "browser/backend change",
    ],
    results: [
      {
        precision: "fp32",
        device: "wasm",
        status: "pass",
        runtime: { embeddingsPerSecond: 12.892, grade: "good" },
      },
      {
        precision: "fp32",
        device: "webgpu",
        status: "pass",
        runtime: { embeddingsPerSecond: 18.428, grade: "good" },
      },
      {
        precision: "fp16",
        device: "wasm",
        status: "fail",
        failure: "ONNX float16 output type mismatch",
      },
      {
        precision: "fp16",
        device: "webgpu",
        status: "fail",
        failure: "bounded browser timeout",
      },
      {
        precision: "q8",
        device: "wasm",
        status: "pass",
        runtime: { embeddingsPerSecond: 12.205, grade: "good" },
      },
      {
        precision: "q8",
        device: "webgpu",
        status: "pass",
        runtime: { embeddingsPerSecond: 8.969, grade: "slow" },
      },
      {
        precision: "q4",
        device: "wasm",
        status: "fail",
        failure: "bounded browser timeout",
      },
      {
        precision: "q4",
        device: "webgpu",
        status: "fail",
        failure: "bounded browser timeout",
      },
      {
        precision: "q4f16",
        device: "wasm",
        status: "fail",
        failure: "ONNX float16 output type mismatch",
      },
      {
        precision: "q4f16",
        device: "webgpu",
        status: "fail",
        failure: "ONNX float16 output type mismatch",
      },
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
  const c = { pooling: "last_token", normalize: true, ...data.modelConfig };
  if (c.pooling !== "last_token") throw Error("pooling must be last_token");
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
