import {
  env,
  pipeline,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0";

const ID = "wav2vec2-base-superb-er",
  REPO = "onnx-community/wav2vec2-base-superb-er-ONNX",
  REV = "e2191e9e8b692c19c7d96e2a1d789ab1df5b7259",
  SHARED_BYTES = 2709;
const candidate = (weightsSize, dtype, filename) => ({
  weightsSize,
  size: weightsSize + SHARED_BYTES,
  modelKeys: [filename],
  dtype,
  candidateDevices: ["wasm", "webgpu"],
});
const PRECISIONS = {
  fp32: candidate(378495202, "fp32", "model.onnx"),
  fp16: candidate(189439719, "fp16", "model_fp16.onnx"),
  q4: candidate(89861131, "q4", "model_q4.onnx"),
  q4f16: candidate(66495610, "q4f16", "model_q4f16.onnx"),
  bnb4: candidate(89861131, "bnb4", "model_bnb4.onnx"),
};
const SUPPORTED = Object.fromEntries(
  Object.entries(PRECISIONS).map(([name, value]) => [
    name,
    { ...value, supportedDevices: value.candidateDevices },
  ]),
);
const MANIFEST = {
  contractVersion: "1.0",
  model: {
    id: ID,
    displayName: "Wav2Vec2 Base SUPERB — Emotion Recognition",
    provider: "SUPERB / S3PRL",
    providerUrl: "https://github.com/s3prl/s3prl",
    license: "Apache-2.0",
    licenseUrl: "https://www.apache.org/licenses/LICENSE-2.0.txt",
    lastUpdated: "2021-11-04T16:03:36.000Z",
    sourceRepository: "superb/wav2vec2-base-superb-er",
    repository: REPO,
    artifactRevision: REV,
    artifactLastUpdated: "2026-09-03T11:54:36.000Z",
    task: "audio-classification",
    description:
      "An English speech emotion classifier that assigns neutral, happy, angry, and sad scores to a 16 kHz mono waveform in the browser.",
    intendedUses: [
      "Exploratory analysis of broad vocal affect in English speech.",
      "Benchmarking browser audio-classification inference.",
    ],
    limitations: [
      "The four labels are coarse and do not represent the full range or ambiguity of human emotion.",
      "Accuracy varies with speaker, accent, acting style, noise, recording quality, and utterance length.",
      "The model was evaluated on acted English IEMOCAP speech and may not generalize to natural conversation or other languages.",
      "Emotion predictions are probabilistic and must not be used for medical, employment, policing, surveillance, or other consequential judgments.",
    ],
  },
  runtime: {
    engine: { name: "Transformers.js", version: "4.2.0" },
    precisions: SUPPORTED,
  },
  input: {
    description: "One non-empty mono Float32Array sampled at 16 kHz.",
    alternatives: [{ name: "audio", type: "Float32Array", required: true }],
  },
  config: {
    model: [
      { name: "top_k", type: "integer", default: 4, minimum: 1, maximum: 4 },
    ],
    generation: [],
  },
  output: {
    description: "All requested emotion labels sorted by descending score.",
    fields: [
      {
        name: "predictions",
        type: "Array<{label:string,score:number}>",
        required: true,
      },
    ],
    example: { predictions: [{ label: "ang", score: 0.8 }] },
  },
  operations: [
    "checkModelSupports",
    "init",
    "download",
    "generate",
    "clearMemory",
  ],
  benchmark: {
    report: "benchmarks/wav2vec2-base-superb-er.json",
    testedAt: "2026-09-12T06:19:51.600Z",
    fixture: {
      file: "crema-d-1001-dfa-ang.wav",
      url: "https://f005.backblazeb2.com/file/hubters-webai-public/assets/benchmarks/crema-d-1001-dfa-ang.wav",
      size: 72862,
      sha1: "3d62067948a906c4df841268c31a96d3f9d14da5",
      durationSeconds: 2.2755625,
      sampleRate: 16000,
      source: "CREMA-D 1001_DFA_ANG_XX.wav",
      license: "ODbL-1.0 / DbCL-1.0",
      licenseUrl:
        "https://github.com/CheyneyComputerScience/CREMA-D/blob/master/LICENSE.txt",
    },
    candidates: PRECISIONS,
    qualityGate:
      "Finite normalized scores for four unique labels, with angry ranked first and exceeding the strongest negative by at least 0.1.",
    invalidatedBy: [
      "artifact revision change",
      "runtime version change",
      "fixture or quality-gate change",
      "browser/backend behavior change",
    ],
    results: [
      { precision: "fp32", device: "wasm", status: "pass", runtime: { loadMs: 33464.9, inferenceMs: 2355.7, realTimeFactor: 0.345072, grade: "good" }, quality: { passed: true, topLabel: "ang", topScore: 0.984452, margin: 0.970826 } },
      { precision: "fp32", device: "webgpu", status: "pass", runtime: { loadMs: 33550.9, inferenceMs: 494.4, realTimeFactor: 0.072422, grade: "excellent" }, quality: { passed: true, topLabel: "ang", topScore: 0.984448, margin: 0.970818 } },
      { precision: "fp16", device: "wasm", status: "pass", runtime: { loadMs: 21450.3, inferenceMs: 2279.1, realTimeFactor: 0.333852, grade: "good" }, quality: { passed: true, topLabel: "ang", topScore: 0.985121, margin: 0.972141 } },
      { precision: "fp16", device: "webgpu", status: "pass", runtime: { loadMs: 23096.3, inferenceMs: 590.2, realTimeFactor: 0.086455, grade: "excellent" }, quality: { passed: true, topLabel: "ang", topScore: 0.9851, margin: 0.972148 } },
      { precision: "q4", device: "wasm", status: "pass", runtime: { loadMs: 12507.8, inferenceMs: 2364.5, realTimeFactor: 0.346361, grade: "good" }, quality: { passed: true, topLabel: "ang", topScore: 0.990658, margin: 0.984401 } },
      { precision: "q4", device: "webgpu", status: "pass", runtime: { loadMs: 12514.5, inferenceMs: 110.8, realTimeFactor: 0.01623, grade: "excellent" }, quality: { passed: true, topLabel: "ang", topScore: 0.990655, margin: 0.984396 } },
      { precision: "q4f16", device: "wasm", status: "pass", runtime: { loadMs: 10639.8, inferenceMs: 2496.7, realTimeFactor: 0.365726, grade: "good" }, quality: { passed: true, topLabel: "ang", topScore: 0.990855, margin: 0.984749 } },
      { precision: "q4f16", device: "webgpu", status: "pass", runtime: { loadMs: 11787.1, inferenceMs: 87.7, realTimeFactor: 0.012847, grade: "excellent" }, quality: { passed: true, topLabel: "ang", topScore: 0.990941, margin: 0.985037 } },
      { precision: "bnb4", device: "wasm", status: "pass", runtime: { loadMs: 13293.8, inferenceMs: 2368.9, realTimeFactor: 0.347006, grade: "good" }, quality: { passed: true, topLabel: "ang", topScore: 0.990658, margin: 0.984401 } },
      { precision: "bnb4", device: "webgpu", status: "pass", runtime: { loadMs: 12688.4, inferenceMs: 109, realTimeFactor: 0.015967, grade: "excellent" }, quality: { passed: true, topLabel: "ang", topScore: 0.990655, margin: 0.984396 } },
    ],
  },
};
const state = { initializing: false, initialized: false, generating: false };
let classifier = null;
const send = (requestId, type, data) =>
  self.postMessage({ requestId, type, data });
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
function configure(c = {}) {
  env.allowRemoteModels = c.allowRemoteModels ?? true;
  env.allowLocalModels = c.allowLocalModels ?? false;
  env.useBrowserCache = c.useBrowserCache ?? true;
  if (c.remoteHost) env.remoteHost = c.remoteHost;
  if (c.remotePathTemplate) env.remotePathTemplate = c.remotePathTemplate;
  if (c.localModelPath) env.localModelPath = c.localModelPath;
}
function validateInit(data) {
  const precision = data?.precision,
    device = data?.device,
    s = SUPPORTED[precision];
  if (!s) throw Error(`Unsupported precision: ${precision}`);
  if (!s.supportedDevices.includes(device))
    throw Error(
      `Precision ${precision} is only supported on: ${s.supportedDevices.join(", ")}`,
    );
  if (device === "webgpu" && !self.navigator?.gpu)
    throw Error("WebGPU is unavailable");
  return { precision, device };
}
async function clear() {
  if (classifier?.dispose) await classifier.dispose();
  classifier = null;
  state.initialized = false;
}
async function load(data, id) {
  const { precision, device } = validateInit(data);
  await clear();
  classifier = await pipeline("audio-classification", REPO, {
    revision: REV,
    dtype: precision,
    device,
    progress_callback: (v) => send(id, "downloadProgress", v),
  });
}
async function generate(data) {
  const audio = data?.userInput?.audio;
  if (!(audio instanceof Float32Array) || !audio.length)
    throw Error("userInput.audio must be a non-empty 16 kHz mono Float32Array");
  if (!audio.every(Number.isFinite))
    throw Error("userInput.audio must contain only finite samples");
  const topK = data?.modelConfig?.top_k ?? 4;
  if (!Number.isInteger(topK) || topK < 1 || topK > 4)
    throw Error("modelConfig.top_k must be an integer from 1 through 4");
  const output = await classifier(audio, { top_k: topK });
  if (
    !Array.isArray(output) ||
    output.some(
      (x) => typeof x?.label !== "string" || !Number.isFinite(x?.score),
    )
  )
    throw Error("Model returned invalid predictions");
  return { predictions: output.map(({ label, score }) => ({ label, score })) };
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
        const s = SUPPORTED[data?.precision];
        if (!s) throw Error(`Unsupported precision: ${data?.precision}`);
        await load(
          {
            ...data,
            device: s.supportedDevices.includes("wasm")
              ? "wasm"
              : s.supportedDevices[0],
          },
          requestId,
        );
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
        if (!state.initialized || !classifier)
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
        throw Error(
          "Stream generation is not supported for audio classification",
        );
      if (type === "clearMemory") {
        if (state.generating)
          throw Error("Cannot clear memory while generation is active");
        await clear();
        send(requestId, type, { status: "success" });
        return;
      }
      throw Error(`Unknown operation: ${type}`);
    } catch (error) {
      send(requestId, "error", {
        message: error instanceof Error ? error.message : String(error),
        context: type,
      });
    }
  },
);
self.postMessage({
  type: "worker initialized",
  data: { success: true, workerVersion: "v4", modelId: ID },
});
