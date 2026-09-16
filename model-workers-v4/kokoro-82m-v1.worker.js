import { KokoroTTS, env } from "https://cdn.jsdelivr.net/npm/kokoro-js@1.2.1/dist/kokoro.web.js";

const ID = "kokoro-82m-v1";
const REPO = "onnx-community/Kokoro-82M-v1.0-ONNX";
const REV = "1939ad2a8e416c0acfeecc08a694d14ef25f2231";
const DEFAULT_VOICE = "af_heart";
const VOICE_BYTES = 522240;
const SHARED_BYTES = 44 + 3497 + 113 + VOICE_BYTES;
const VOICES = ["af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica", "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky", "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael", "am_onyx", "am_puck", "am_santa", "bf_alice", "bf_emma", "bf_isabella", "bf_lily", "bm_daniel", "bm_fable", "bm_george", "bm_lewis"];

function candidate(weightsSize, dtype, filename) {
  return { weightsSize, size: weightsSize + SHARED_BYTES, modelKeys: [filename], dtype, candidateDevices: ["wasm", "webgpu"] };
}

const PRECISIONS = {
  fp32: candidate(325532232, "fp32", "model.onnx"),
  fp16: candidate(163234740, "fp16", "model_fp16.onnx"),
  q4: candidate(305215966, "q4", "model_q4.onnx"),
  q4f16: candidate(154586422, "q4f16", "model_q4f16.onnx"),
  q8f16: candidate(86033585, "q8f16", "model_q8f16.onnx"),
  q8: candidate(92361116, "q8", "model_quantized.onnx"),
  uint8: candidate(177464632, "uint8", "model_uint8.onnx"),
  uint8f16: candidate(114209226, "uint8f16", "model_uint8f16.onnx"),
};

const SUPPORTED = {
  fp32: { ...PRECISIONS.fp32, supportedDevices: ["wasm", "webgpu"] },
  fp16: { ...PRECISIONS.fp16, supportedDevices: ["wasm", "webgpu"] },
  q4: { ...PRECISIONS.q4, supportedDevices: ["wasm", "webgpu"] },
  q4f16: { ...PRECISIONS.q4f16, supportedDevices: ["wasm", "webgpu"] },
  q8: { ...PRECISIONS.q8, supportedDevices: ["wasm"] },
  uint8: { ...PRECISIONS.uint8, supportedDevices: ["wasm"] },
};

const MANIFEST = {
  contractVersion: "1.0",
  model: {
    id: ID, displayName: "Kokoro 82M v1.0 — Text to Speech", provider: "Hexgrad", providerUrl: "https://github.com/hexgrad/kokoro",
    license: "Apache-2.0", licenseUrl: "https://www.apache.org/licenses/LICENSE-2.0",
    lastUpdated: "2025-04-10T18:12:48.000Z", sourceRepository: "hexgrad/Kokoro-82M", repository: REPO,
    artifactRevision: REV, artifactLastUpdated: "2025-02-08T12:15:27.000Z", task: "text-to-speech",
    description: "An 82-million-parameter text-to-speech model that converts short English text into 24 kHz mono speech in a browser worker.",
    intendedUses: ["Local narration and accessibility prototypes.", "Generating short English voice prompts without sending text to an inference server."],
    limitations: ["The bundled phonemizer and advertised voices support English only.", "Names, acronyms, numbers, and unusual punctuation may be pronounced incorrectly.", "Long text is truncated to the model context; split long passages into sentences.", "Synthetic speech can sound unnatural and must not be used to impersonate a person or mislead listeners."],
  },
  runtime: { engine: { name: "kokoro-js (Transformers.js adapter)", version: "1.2.1 / 3.5.1" }, precisions: SUPPORTED },
  input: { description: "A non-empty English text string.", alternatives: [{ name: "text", type: "string", required: true, maximumLength: 2000 }] },
  config: {
    model: [],
    generation: [
      { name: "voice", type: "string", default: DEFAULT_VOICE, enum: VOICES },
      { name: "speed", type: "number", default: 1, minimum: 0.5, maximum: 2 },
    ],
  },
  output: { description: "24 kHz mono PCM samples and audio metadata.", fields: [{ name: "audio", type: "Float32Array", required: true }, { name: "samplingRate", type: "number", required: true }, { name: "durationSeconds", type: "number", required: true }], example: { audio: "Float32Array", samplingRate: 24000, durationSeconds: 2.4 } },
  operations: ["checkModelSupports", "init", "download", "generate", "clearMemory"],
  benchmark: {
    report: "benchmarks/kokoro-82m-v1.json", testedAt: "2026-09-10T16:36:45.442Z",
    fixture: { file: "kokoro-82m-v1.json", url: "https://f005.backblazeb2.com/file/hubters-webai-public/assets/benchmarks/kokoro-82m-v1.json", size: 338, sha1: "9b670566697f3b67757bbbb2f15a20512540258a", license: "CC0-1.0", reference: { file: "kokoro-82m-v1-reference.wav", url: "https://f005.backblazeb2.com/file/hubters-webai-public/assets/benchmarks/kokoro-82m-v1-reference.wav", size: 158444, sha1: "fd763f81add8aa7fe892dea39117072a35a31fe0", samplingRate: 24000 } },
    candidates: PRECISIONS,
    qualityGate: "Finite, non-silent 24 kHz audio of plausible duration, with waveform similarity to the checked-in fp32/WASM reference.",
    invalidatedBy: ["artifact revision change", "runtime version change", "fixture or reference waveform change", "quality-gate change", "browser/backend behavior change"],
    results: [
      { precision: "fp32", device: "wasm", status: "pass", runtime: { loadMs: 41754.4, inferenceMs: 4349.5, secondsPerAudioSecond: 1.318, grade: "slow" }, quality: { envelopeCorrelation: 1, durationRatio: 1 } },
      { precision: "fp32", device: "webgpu", status: "pass", runtime: { loadMs: 40312.9, inferenceMs: 464.9, secondsPerAudioSecond: 0.141, grade: "excellent" }, quality: { envelopeCorrelation: 1, durationRatio: 1 } },
      { precision: "fp16", device: "wasm", status: "pass", runtime: { loadMs: 23716.8, inferenceMs: 4484.7, secondsPerAudioSecond: 1.359, grade: "slow" }, quality: { envelopeCorrelation: 0.9994, durationRatio: 1 } },
      { precision: "fp16", device: "webgpu", status: "pass", runtime: { loadMs: 22822.5, inferenceMs: 473.2, secondsPerAudioSecond: 0.143, grade: "excellent" }, quality: { envelopeCorrelation: 0.9703, durationRatio: 1 } },
      { precision: "q4", device: "wasm", status: "pass", runtime: { loadMs: 37887.5, inferenceMs: 4511.3, secondsPerAudioSecond: 1.367, grade: "slow" }, quality: { envelopeCorrelation: 0.9069, durationRatio: 1 } },
      { precision: "q4", device: "webgpu", status: "pass", runtime: { loadMs: 37563.2, inferenceMs: 470.7, secondsPerAudioSecond: 0.143, grade: "excellent" }, quality: { envelopeCorrelation: 0.895, durationRatio: 1 } },
      { precision: "q4f16", device: "wasm", status: "pass", runtime: { loadMs: 25810.3, inferenceMs: 4515.3, secondsPerAudioSecond: 1.368, grade: "slow" }, quality: { envelopeCorrelation: 0.9089, durationRatio: 1 } },
      { precision: "q4f16", device: "webgpu", status: "pass", runtime: { loadMs: 22052.7, inferenceMs: 472.9, secondsPerAudioSecond: 0.143, grade: "excellent" }, quality: { envelopeCorrelation: 0.8906, durationRatio: 1 } },
      { precision: "q8f16", device: "wasm", status: "fail", failure: "kokoro-js rejects the q8f16 dtype" },
      { precision: "q8f16", device: "webgpu", status: "fail", failure: "kokoro-js rejects the q8f16 dtype" },
      { precision: "q8", device: "wasm", status: "pass", runtime: { loadMs: 14929.6, inferenceMs: 5986, secondsPerAudioSecond: 1.76, grade: "slow" }, quality: { envelopeCorrelation: 0.688, durationRatio: 1.0303 } },
      { precision: "q8", device: "webgpu", status: "fail", failure: "generated audio failed the reference-envelope quality threshold" },
      { precision: "uint8", device: "wasm", status: "pass", runtime: { loadMs: 26865.8, inferenceMs: 4562, secondsPerAudioSecond: 1.362, grade: "slow" }, quality: { envelopeCorrelation: 0.8494, durationRatio: 1.0152 } },
      { precision: "uint8", device: "webgpu", status: "fail", failure: "generated audio contained invalid samples" },
      { precision: "uint8f16", device: "wasm", status: "fail", failure: "kokoro-js rejects the uint8f16 dtype" },
      { precision: "uint8f16", device: "webgpu", status: "fail", failure: "kokoro-js rejects the uint8f16 dtype" },
    ],
  },
};

const state = { initializing: false, initialized: false, generating: false };
let tts = null;
const send = (requestId, type, data) => self.postMessage({ requestId, type, data });
const caps = { supportedModes: ["webai"], supportedPrecisions: Object.keys(SUPPORTED), supportedPrecisionsDevicesMap: SUPPORTED, doesSupportStreamGeneration: false, externalInterrupt: true, workerVersion: "v4", transformersJsVersion: "3.5.1", cacheModelId: REPO };

function configure(config = {}) {
  if (config.wasmPaths) env.wasmPaths = config.wasmPaths;
}

function validateInit(data) {
  const precision = data?.precision, device = data?.device, supported = SUPPORTED[precision];
  if (!supported) throw Error(`Unsupported precision: ${precision}`);
  if (!supported.supportedDevices.includes(device)) throw Error(`Precision ${precision} is only supported on: ${supported.supportedDevices.join(", ")}`);
  if (device === "webgpu" && !self.navigator?.gpu) throw Error("WebGPU is unavailable");
  return { precision, device };
}

async function ensurePinnedVoice(voice) {
  if (!VOICES.includes(voice)) throw Error(`Unsupported voice: ${voice}`);
  const cache = await caches.open("kokoro-voices");
  const runtimeUrl = `https://huggingface.co/${REPO}/resolve/main/voices/${voice}.bin`;
  if (await cache.match(runtimeUrl)) return;
  const pinnedUrl = `https://huggingface.co/${REPO}/resolve/${REV}/voices/${voice}.bin`;
  const response = await fetch(pinnedUrl);
  if (!response.ok) throw Error(`Unable to download voice ${voice}: HTTP ${response.status}`);
  await cache.put(runtimeUrl, response);
}

async function clear() {
  if (tts?.model?.dispose) await tts.model.dispose();
  tts = null;
  state.initialized = false;
}

async function load(data, requestId) {
  const { precision, device } = validateInit(data);
  await clear();
  await ensurePinnedVoice(DEFAULT_VOICE);
  tts = await KokoroTTS.from_pretrained(REPO, { dtype: PRECISIONS[precision].dtype, device, progress_callback: (value) => send(requestId, "downloadProgress", value) });
}

function validateGeneration(data) {
  const text = data?.userInput?.text;
  if (typeof text !== "string" || !text.trim()) throw Error("userInput.text must be a non-empty string");
  if (text.length > 2000) throw Error("userInput.text must be at most 2000 characters");
  const voice = data?.generateConfig?.voice ?? DEFAULT_VOICE;
  const speed = data?.generateConfig?.speed ?? 1;
  if (!VOICES.includes(voice)) throw Error(`voice must be one of: ${VOICES.join(", ")}`);
  if (typeof speed !== "number" || !Number.isFinite(speed) || speed < 0.5 || speed > 2) throw Error("speed must be a finite number from 0.5 through 2");
  return { text: text.trim(), voice, speed };
}

async function generate(data) {
  const { text, voice, speed } = validateGeneration(data);
  await ensurePinnedVoice(voice);
  const result = await tts.generate(text, { voice, speed });
  const audio = result?.audio;
  if (!(audio instanceof Float32Array) || audio.length === 0 || !audio.every(Number.isFinite)) throw Error("Model returned invalid audio samples");
  return { audio, samplingRate: result.sampling_rate, durationSeconds: audio.length / result.sampling_rate };
}

self.addEventListener("message", async ({ data: { requestId, type, data } = {} }) => {
  try {
    if (type === "checkModelSupports") { configure(data?.workerConfig); send(requestId, type, { ...caps, manifest: MANIFEST }); return; }
    if (type === "download") { if (state.initializing || state.generating) throw Error("Worker is busy"); const supported = SUPPORTED[data?.precision]; if (!supported) throw Error(`Unsupported precision: ${data?.precision}`); const device = supported.supportedDevices.includes("wasm") ? "wasm" : supported.supportedDevices[0]; await load({ ...data, device }, requestId); await clear(); send(requestId, type, { status: "success" }); return; }
    if (type === "init") { if (state.initializing || state.generating) throw Error("Worker is busy"); state.initializing = true; try { await load(data, requestId); state.initialized = true; send(requestId, type, { status: "success" }); } finally { state.initializing = false; } return; }
    if (type === "generate") { if (!state.initialized || !tts) throw Error("Call init before generate"); if (state.generating) throw Error("Generation already in progress"); state.generating = true; try { send(requestId, "generated", { status: "success", result: await generate(data) }); } finally { state.generating = false; } return; }
    if (type === "generateStream") throw Error("Stream generation is not supported by this worker");
    if (type === "clearMemory") { if (state.generating) throw Error("Cannot clear memory while generation is active"); await clear(); send(requestId, type, { status: "success" }); return; }
    throw Error(`Unknown operation: ${type}`);
  } catch (error) { send(requestId, "error", { message: error instanceof Error ? error.message : String(error), context: type }); }
});

self.postMessage({ type: "worker initialized", data: { success: true, workerVersion: "v4", modelId: ID } });
