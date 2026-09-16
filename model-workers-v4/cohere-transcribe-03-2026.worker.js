import { env, pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0";
import { MPEGDecoderWebWorker } from "https://cdn.jsdelivr.net/npm/mpg123-decoder@1.0.0/+esm";
import wavDecoder from "https://cdn.jsdelivr.net/npm/wav-decoder@1.3.0/+esm";

const ID = "cohere-transcribe-03-2026";
const REPO = "onnx-community/cohere-transcribe-03-2026-ONNX";
const REV = "31b1c6211c9000d76b077ddd23b74c9090badeba";
const SHARED_BYTES = 1163478;
const PRECISIONS = {
  fp32: candidate(8263437701, "fp32", "encoder_model.onnx", "decoder_model_merged.onnx"),
  fp16: candidate(4132397864, "fp16", "encoder_model_fp16.onnx", "decoder_model_merged_fp16.onnx"),
  q8: candidate(3068524402, "q8", "encoder_model_quantized.onnx", "decoder_model_merged_quantized.onnx"),
  q4: candidate(2126516259, "q4", "encoder_model_q4.onnx", "decoder_model_merged_q4.onnx"),
  q4f16: candidate(1535058724, "q4f16", "encoder_model_q4f16.onnx", "decoder_model_merged_q4f16.onnx"),
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
  q4: { ...PRECISIONS.q4, supportedDevices: ["webgpu"] },
};

const MANIFEST = {
  contractVersion: "1.0",
  model: {
    id: ID,
    displayName: "Cohere Transcribe 03-2026 — Multilingual ASR",
    provider: "Cohere Labs",
    providerUrl: "https://cohere.com/research",
    license: "Apache-2.0",
    licenseUrl: "https://huggingface.co/CohereLabs/cohere-transcribe-03-2026#license",
    lastUpdated: "2026-06-10T15:46:19.000Z",
    sourceRepository: "CohereLabs/cohere-transcribe-03-2026",
    repository: REPO,
    artifactRevision: REV,
    artifactLastUpdated: "2026-03-30T17:27:09.000Z",
    task: "automatic-speech-recognition",
    description: "A two-billion-parameter multilingual Conformer speech-recognition model that transcribes 16 kHz mono audio locally in the browser.",
    intendedUses: ["Client-side transcription in one of fourteen explicitly selected languages.", "Benchmarking multilingual browser speech recognition."],
    limitations: [
      "The model requires an explicit language and does not reliably detect languages or handle code-switching.",
      "Word- or segment-level timestamps and speaker diarization are not supported; output contains transcript text only.",
      "Accuracy varies with accent, noise, microphone quality, speaking rate, and specialized vocabulary.",
      "Long or silent recordings can produce omissions or hallucinated text and should be reviewed.",
      "Transcripts must not be treated as authoritative for safety-critical decisions.",
    ],
  },
  runtime: { engine: { name: "Transformers.js", version: "4.2.0" }, precisions: SUPPORTED },
  input: {
    description: "Provide exactly one 16 kHz mono Float32Array or a browser blob URL containing WAV/MP3 audio.",
    alternatives: [{ name: "audio", type: "Float32Array", required: true }, { name: "audio_blob_url", type: "string(blob URL)", required: true }],
  },
  config: {
    model: [{ name: "language", type: "string", default: "en", enum: ["en", "fr", "de", "es", "it", "pt", "nl", "pl", "el", "ar", "ja", "zh", "vi", "ko"], description: "Required spoken language: en (English), fr (French), de (German), es (Spanish), it (Italian), pt (Portuguese), nl (Dutch), pl (Polish), el (Greek), ar (Arabic), ja (Japanese), zh (Chinese/Mandarin), vi (Vietnamese), or ko (Korean). Automatic detection and code-switching are not supported." }, { name: "max_new_tokens", type: "integer", default: 256, minimum: 1, maximum: 1024 }],
    generation: [],
  },
  output: {
    description: "The complete transcript in the explicitly selected spoken language.",
    fields: [{ name: "text", type: "string", required: true }],
    example: { text: "Hello world." },
  },
  operations: ["checkModelSupports", "init", "download", "generate", "clearMemory"],
  benchmark: {
    report: "benchmarks/cohere-transcribe-03-2026.json",
    testedAt: "2026-09-12T13:46:52.787Z",
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
      { precision: "fp32", device: "wasm", status: "fail", failure: "Cold initialization exceeded the bounded 20-minute benchmark timeout." },
      { precision: "fp32", device: "webgpu", status: "fail", failure: "Runtime module fetch failed in the isolated browser; support was not established." },
      { precision: "fp16", device: "wasm", status: "fail", failure: "Runtime module fetch failed in the isolated browser; support was not established." },
      { precision: "fp16", device: "webgpu", status: "fail", failure: "Runtime module fetch failed in the isolated browser; support was not established." },
      { precision: "q8", device: "wasm", status: "fail", failure: "Runtime module fetch failed in the isolated browser; support was not established." },
      { precision: "q8", device: "webgpu", status: "fail", failure: "Runtime module fetch failed in the isolated browser; support was not established." },
      { precision: "q4", device: "wasm", status: "fail", failure: "ONNX Runtime Web has no WASM implementation for the decoder GatherBlockQuantized node." },
      { precision: "q4", device: "webgpu", status: "pass", runtime: { loadMs: 215768.8, inferenceMs: 1832.1, realTimeFactor: 0.0621, grade: "excellent" }, quality: { wordErrorRate: 0.1667, grade: "acceptable", gibberish: false } },
      { precision: "q4f16", device: "wasm", status: "fail", failure: "Runtime module fetch failed in the isolated browser; support was not established." },
      { precision: "q4f16", device: "webgpu", status: "fail", failure: "Runtime module fetch failed in the isolated browser; support was not established." },
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

function convertToMono(channelData, sampleCount) {
  if (channelData.length === 1) return channelData[0];
  const mono = new Float32Array(sampleCount);
  for (let i = 0; i < sampleCount; i++) mono[i] = (channelData[0][i] + channelData[1][i]) * 0.5;
  return mono;
}

function resampleAudio(audio, sourceRate, targetRate = 16000) {
  if (sourceRate === targetRate) return audio;
  const ratio = sourceRate / targetRate, output = new Float32Array(Math.floor(audio.length / ratio));
  for (let i = 0; i < output.length; i++) {
    const position = i * ratio, lower = Math.floor(position), fraction = position - lower;
    output[i] = lower + 1 < audio.length ? audio[lower] * (1 - fraction) + audio[lower + 1] * fraction : audio[lower];
  }
  return output;
}

async function decodeAudio(arrayBuffer) {
  try {
    const decoded = await wavDecoder.decode(arrayBuffer);
    return resampleAudio(convertToMono(decoded.channelData, decoded.channelData[0].length), decoded.sampleRate);
  } catch {
    const decoder = new MPEGDecoderWebWorker();
    try {
      await decoder.ready;
      const decoded = await decoder.decode(arrayBuffer);
      return resampleAudio(convertToMono(decoded.channelData, decoded.samplesDecoded), decoded.sampleRate);
    } finally { await decoder.free(); }
  }
}

async function generate(data) {
  let audio = data?.userInput?.audio;
  if (!(audio instanceof Float32Array) && data?.userInput?.audio_blob_url?.startsWith("blob:")) {
    const response = await fetch(data.userInput.audio_blob_url);
    if (!response.ok) throw Error(`Failed to fetch audio blob: HTTP ${response.status}`);
    audio = await decodeAudio(await response.arrayBuffer());
  }
  if (!(audio instanceof Float32Array) || audio.length === 0) throw Error("userInput.audio must be a non-empty 16 kHz mono Float32Array");
  if (!audio.every(Number.isFinite)) throw Error("userInput.audio must contain only finite samples");
  const languages = ["en", "fr", "de", "es", "it", "pt", "nl", "pl", "el", "ar", "ja", "zh", "vi", "ko"];
  const language = data?.modelConfig?.language ?? "en";
  if (!languages.includes(language)) throw Error(`language must be one of: ${languages.join(", ")}`);
  const maxNewTokens = data?.modelConfig?.max_new_tokens ?? 256;
  if (!Number.isInteger(maxNewTokens) || maxNewTokens < 1 || maxNewTokens > 1024) throw Error("max_new_tokens must be an integer from 1 through 1024");
  const output = await transcriber(audio, { max_new_tokens: maxNewTokens, language });
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
