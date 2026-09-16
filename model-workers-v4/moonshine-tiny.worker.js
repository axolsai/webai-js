import { env, pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0";

const ID = "moonshine-tiny";
const REPO = "onnx-community/moonshine-tiny-ONNX";
const REV = "a6da1241cd305dcd64eab1edbd615f2bb9aabb95";
const SHARED_BYTES = 3898688;
const PRECISIONS = {
  fp32: candidate(109109881, "fp32", "encoder_model.onnx", "decoder_model_merged.onnx"),
  fp16: candidate(91770581, "fp16", "encoder_model_fp16.onnx", "decoder_model_merged_fp16.onnx"),
  int8: candidate(28109347, "int8", "encoder_model_int8.onnx", "decoder_model_merged_int8.onnx"),
  uint8: candidate(28109400, "uint8", "encoder_model_uint8.onnx", "decoder_model_merged_uint8.onnx"),
  q8: candidate(28180947, "q8", "encoder_model_quantized.onnx", "decoder_model_merged_quantized.onnx"),
  q4: candidate(55382958, "q4", "encoder_model_q4.onnx", "decoder_model_merged_q4.onnx"),
  q4f16: candidate(56024215, "q4f16", "encoder_model_q4f16.onnx", "decoder_model_merged_q4f16.onnx"),
  bnb4: candidate(54388824, "bnb4", "encoder_model_bnb4.onnx", "decoder_model_merged_bnb4.onnx"),
};

function candidate(weightsSize, dtype, encoder, decoder) {
  return {
    weightsSize,
    size: weightsSize + SHARED_BYTES,
    modelKeys: [encoder, decoder],
    dtype: { encoder_model: dtype, decoder_model_merged: dtype },
    candidateDevices: ["wasm", "webgpu"],
  };
}

const SUPPORTED = {
  fp32: { ...PRECISIONS.fp32, supportedDevices: ["wasm", "webgpu"] },
  int8: { ...PRECISIONS.int8, supportedDevices: ["webgpu"] },
  uint8: { ...PRECISIONS.uint8, supportedDevices: ["webgpu"] },
  q8: { ...PRECISIONS.q8, supportedDevices: ["webgpu"] },
  q4: { ...PRECISIONS.q4, supportedDevices: ["wasm", "webgpu"] },
  bnb4: { ...PRECISIONS.bnb4, supportedDevices: ["wasm", "webgpu"] },
};

const MANIFEST = {
  contractVersion: "1.0",
  model: {
    id: ID,
    displayName: "Moonshine Tiny — English Speech Recognition",
    provider: "Moonshine AI / Useful Sensors",
    providerUrl: "https://github.com/usefulsensors/moonshine",
    license: "MIT",
    licenseUrl: "https://opensource.org/license/mit/",
    lastUpdated: "2025-01-30T16:07:38.000Z",
    sourceRepository: "moonshine-ai/moonshine-tiny",
    repository: REPO,
    artifactRevision: REV,
    artifactLastUpdated: "2025-01-17T14:05:00.000Z",
    task: "automatic-speech-recognition",
    description: "A compact English speech-recognition model that transcribes 16 kHz mono audio locally in the browser.",
    intendedUses: ["Client-side English speech transcription.", "Prototyping low-latency browser voice interfaces."],
    limitations: [
      "The model supports English speech only.",
      "Accuracy varies with accent, noise, microphone quality, speaking rate, and specialized vocabulary.",
      "Long or silent recordings can produce omissions or hallucinated text and should be reviewed.",
      "Transcripts must not be treated as authoritative for safety-critical decisions.",
    ],
  },
  runtime: { engine: { name: "Transformers.js", version: "4.2.0" }, precisions: SUPPORTED },
  input: {
    description: "One non-empty mono Float32Array sampled at 16 kHz.",
    alternatives: [{ name: "audio", type: "Float32Array", required: true }],
  },
  config: {
    model: [{ name: "max_new_tokens", type: "integer", default: 256, minimum: 1, maximum: 512 }],
    generation: [],
  },
  output: {
    description: "The complete English transcript.",
    fields: [{ name: "text", type: "string", required: true }],
    example: { text: "Hello world." },
  },
  operations: ["checkModelSupports", "init", "download", "generate", "clearMemory"],
  benchmark: {
    report: "benchmarks/moonshine-tiny.json",
    testedAt: "2026-09-10T11:58:01.358Z",
    fixture: {
      file: "whisper-standard.mp3",
      url: "https://f005.backblazeb2.com/file/hubters-webai-public/assets/benchmarks/whisper-standard.mp3",
      size: 471925,
      sha1: "5faad1507b5d6bc61a39ca3184db5a7b554a7479",
      durationSeconds: 29.49225,
    },
    candidates: PRECISIONS,
    qualityGate: "Non-empty, non-repetitive English transcript with word error rate at most 0.35 against the checked-in reference.",
    invalidatedBy: ["artifact revision change", "runtime version change", "fixture or quality gate change", "browser/backend change"],
    results: [
      { precision: "fp32", device: "wasm", status: "pass", runtime: { realTimeFactor: 0.0431, grade: "excellent" }, quality: { wordErrorRate: 0.1212, grade: "excellent" } },
      { precision: "fp32", device: "webgpu", status: "pass", runtime: { realTimeFactor: 0.025, grade: "excellent" }, quality: { wordErrorRate: 0.1212, grade: "excellent" } },
      { precision: "fp16", device: "wasm", status: "fail", failure: "ONNX session graph error" },
      { precision: "fp16", device: "webgpu", status: "fail", failure: "invalid ONNX subgraph output" },
      { precision: "int8", device: "wasm", status: "fail", failure: "missing quantization scale" },
      { precision: "int8", device: "webgpu", status: "pass", runtime: { realTimeFactor: 0.1305, grade: "excellent" }, quality: { wordErrorRate: 0.1818, grade: "acceptable" } },
      { precision: "uint8", device: "wasm", status: "fail", failure: "missing quantization scale" },
      { precision: "uint8", device: "webgpu", status: "pass", runtime: { realTimeFactor: 0.1046, grade: "excellent" }, quality: { wordErrorRate: 0.1667, grade: "acceptable" } },
      { precision: "q8", device: "wasm", status: "fail", failure: "missing quantization scale" },
      { precision: "q8", device: "webgpu", status: "pass", runtime: { realTimeFactor: 0.0936, grade: "excellent" }, quality: { wordErrorRate: 0.1212, grade: "excellent" } },
      { precision: "q4", device: "wasm", status: "pass", runtime: { realTimeFactor: 0.07, grade: "excellent" }, quality: { wordErrorRate: 0.2121, grade: "acceptable" } },
      { precision: "q4", device: "webgpu", status: "pass", runtime: { realTimeFactor: 0.0237, grade: "excellent" }, quality: { wordErrorRate: 0.2121, grade: "acceptable" } },
      { precision: "q4f16", device: "wasm", status: "fail", failure: "ONNX session graph error" },
      { precision: "q4f16", device: "webgpu", status: "fail", failure: "invalid ONNX subgraph output" },
      { precision: "bnb4", device: "wasm", status: "pass", runtime: { realTimeFactor: 0.0559, grade: "excellent" }, quality: { wordErrorRate: 0.1515, grade: "acceptable" } },
      { precision: "bnb4", device: "webgpu", status: "pass", runtime: { realTimeFactor: 0.0988, grade: "excellent" }, quality: { wordErrorRate: 0.1515, grade: "acceptable" } },
    ],
  },
};

const state = { initializing: false, initialized: false, generating: false };
let transcriber = null;
const caps = {
  supportedModes: ["webai"], supportedPrecisions: Object.keys(SUPPORTED), supportedPrecisionsDevicesMap: SUPPORTED,
  doesSupportStreamGeneration: false, externalInterrupt: true, workerVersion: "v4", transformersJsVersion: "4.2.0", cacheModelId: REPO,
};
const send = (requestId, type, data) => self.postMessage({ requestId, type, data });

function configure(config = {}) {
  env.allowRemoteModels = config.allowRemoteModels ?? true;
  env.allowLocalModels = config.allowLocalModels ?? false;
  env.useBrowserCache = config.useBrowserCache ?? true;
  if (config.remoteHost) env.remoteHost = config.remoteHost;
  if (config.remotePathTemplate) env.remotePathTemplate = config.remotePathTemplate;
  if (config.localModelPath) env.localModelPath = config.localModelPath;
}

function validateInit(data) {
  const precision = data?.precision, device = data?.device, supported = SUPPORTED[precision];
  if (!supported) throw Error(`Unsupported precision: ${precision}`);
  if (!supported.supportedDevices.includes(device)) throw Error(`Precision ${precision} is only supported on: ${supported.supportedDevices.join(", ")}`);
  if (device === "webgpu" && !self.navigator?.gpu) throw Error("WebGPU is unavailable");
  return { precision, device };
}

async function clear() {
  if (transcriber?.dispose) await transcriber.dispose();
  transcriber = null;
  state.initialized = false;
}

async function load(data, requestId) {
  const { precision, device } = validateInit(data);
  await clear();
  transcriber = await pipeline("automatic-speech-recognition", REPO, {
    revision: REV, dtype: PRECISIONS[precision].dtype, device,
    progress_callback: (value) => send(requestId, "downloadProgress", value),
  });
}

async function generate(data) {
  const audio = data?.userInput?.audio;
  if (!(audio instanceof Float32Array) || audio.length === 0) throw Error("userInput.audio must be a non-empty 16 kHz mono Float32Array");
  if (!audio.every(Number.isFinite)) throw Error("userInput.audio must contain only finite samples");
  const maxNewTokens = data?.modelConfig?.max_new_tokens ?? 256;
  if (!Number.isInteger(maxNewTokens) || maxNewTokens < 1 || maxNewTokens > 512) throw Error("max_new_tokens must be an integer from 1 through 512");
  const output = await transcriber(audio, { max_new_tokens: maxNewTokens });
  if (!output || typeof output.text !== "string") throw Error("Model returned an invalid transcript");
  return { text: output.text };
}

self.addEventListener("message", async ({ data: { requestId, type, data } = {} }) => {
  try {
    if (type === "checkModelSupports") { configure(data?.workerConfig); send(requestId, type, { ...caps, manifest: MANIFEST }); return; }
    if (type === "download") { if (state.initializing || state.generating) throw Error("Worker is busy"); await load(data, requestId); await clear(); send(requestId, type, { status: "success" }); return; }
    if (type === "init") { if (state.initializing || state.generating) throw Error("Worker is busy"); state.initializing = true; try { await load(data, requestId); state.initialized = true; send(requestId, type, { status: "success" }); } finally { state.initializing = false; } return; }
    if (type === "generate") { if (!state.initialized || !transcriber) throw Error("Call init before generate"); if (state.generating) throw Error("Generation already in progress"); state.generating = true; try { send(requestId, "generated", { status: "success", result: await generate(data) }); } finally { state.generating = false; } return; }
    if (type === "generateStream") throw Error("Stream generation is not supported for speech recognition");
    if (type === "clearMemory") { if (state.generating) throw Error("Cannot clear memory while generation is active"); await clear(); send(requestId, type, { status: "success" }); return; }
    throw Error(`Unknown operation: ${type}`);
  } catch (error) { send(requestId, "error", { message: error instanceof Error ? error.message : String(error), context: type }); }
});

self.postMessage({ type: "worker initialized", data: { success: true, workerVersion: "v4", modelId: ID } });
