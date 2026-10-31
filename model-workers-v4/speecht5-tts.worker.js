import {
  env,
  pipeline,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.6";

const ID = "speecht5-tts";
const REPO = "Xenova/speecht5_tts";
const REV = "1723781b8ce2d02f0400c8337be04ae8ee3d6d56";
const DEFAULT_SPEAKER =
  "https://f005.backblazeb2.com/file/hubters-webai-public/assets/benchmarks/speecht5-speaker-embedding.bin";
const SHARED_BYTES = 253844;
const candidate = (weightsSize, suffix) => ({
  weightsSize,
  size: weightsSize + SHARED_BYTES,
  modelKeys: [
    `encoder_model${suffix}.onnx`,
    `decoder_model_merged${suffix}.onnx`,
  ],
  candidateDevices: ["wasm", "webgpu"],
});
const PRECISIONS = {
  fp32: candidate(642722177, ""),
  fp16: candidate(346652927, "_fp16"),
  int8: candidate(207883080, "_int8"),
  uint8: candidate(207883151, "_uint8"),
  q8: candidate(214921427, "_quantized"),
  q4: candidate(154299075, "_q4"),
  q4f16: candidate(140984418, "_q4f16"),
  bnb4: candidate(145363764, "_bnb4"),
};
const SUPPORTED = {
  fp32: { ...PRECISIONS.fp32, supportedDevices: ["wasm"] },
  fp16: { ...PRECISIONS.fp16, supportedDevices: ["wasm"] },
  q8: { ...PRECISIONS.q8, supportedDevices: ["wasm"] },
  q4: { ...PRECISIONS.q4, supportedDevices: ["wasm"] },
  q4f16: { ...PRECISIONS.q4f16, supportedDevices: ["wasm"] },
};
const RESULTS = [
  {
    precision: "fp32",
    device: "wasm",
    status: "pass",
    runtime: {
      loadMs: 18970.3,
      inferenceMs: 5525.5,
      totalMs: 24635.3,
      secondsPerAudioSecond: 1.289,
      grade: "slow",
    },
    quality: {
      envelopeDtwDistance: 0.221,
      durationRatio: 0.971,
      grade: "good",
    },
  },
  {
    precision: "fp32",
    device: "webgpu",
    status: "fail",
    runtime: {
      loadMs: 18389.6,
      inferenceMs: null,
      totalMs: 21315.6,
      grade: "failed",
    },
    failure: "webgpu-matmul-error",
  },
  {
    precision: "fp16",
    device: "wasm",
    status: "pass",
    runtime: {
      loadMs: 9905.4,
      inferenceMs: 8464.1,
      totalMs: 18573.7,
      secondsPerAudioSecond: 2.083,
      grade: "very-slow",
    },
    quality: { envelopeDtwDistance: 0.227, durationRatio: 0.92, grade: "good" },
  },
  {
    precision: "fp16",
    device: "webgpu",
    status: "fail",
    runtime: {
      loadMs: null,
      inferenceMs: null,
      totalMs: 8328,
      grade: "failed",
    },
    failure: "webgpu-session-error",
  },
  {
    precision: "int8",
    device: "wasm",
    status: "fail",
    runtime: {
      loadMs: 6809.3,
      inferenceMs: 5488.5,
      totalMs: 12434.4,
      secondsPerAudioSecond: 1.665,
      grade: "very-slow",
    },
    quality: {
      envelopeDtwDistance: 0.349,
      durationRatio: 0.746,
      grade: "failed",
    },
    failure: "quality-duration",
  },
  {
    precision: "int8",
    device: "webgpu",
    status: "fail",
    runtime: {
      loadMs: 6002.7,
      inferenceMs: 3796.8,
      totalMs: 9980.4,
      secondsPerAudioSecond: 10.786,
      grade: "very-slow",
    },
    quality: {
      envelopeDtwDistance: 0.612,
      durationRatio: 0.08,
      grade: "failed",
    },
    failure: "quality-duration-envelope",
  },
  {
    precision: "uint8",
    device: "wasm",
    status: "fail",
    runtime: {
      loadMs: 6171.6,
      inferenceMs: 5307.8,
      totalMs: 11617.3,
      secondsPerAudioSecond: 1.659,
      grade: "very-slow",
    },
    quality: {
      envelopeDtwDistance: 0.377,
      durationRatio: 0.725,
      grade: "failed",
    },
    failure: "quality-duration",
  },
  {
    precision: "uint8",
    device: "webgpu",
    status: "fail",
    runtime: {
      loadMs: 5783.5,
      inferenceMs: 3068.4,
      totalMs: 8978.8,
      secondsPerAudioSecond: 95.887,
      grade: "very-slow",
    },
    quality: {
      envelopeDtwDistance: 0.552,
      durationRatio: 0.007,
      grade: "failed",
    },
    failure: "quality-duration-envelope",
  },
  {
    precision: "q8",
    device: "wasm",
    status: "pass",
    runtime: {
      loadMs: 6389.7,
      inferenceMs: 6237.7,
      totalMs: 12756.9,
      secondsPerAudioSecond: 1.477,
      grade: "slow",
    },
    quality: {
      envelopeDtwDistance: 0.263,
      durationRatio: 0.957,
      grade: "good",
    },
  },
  {
    precision: "q8",
    device: "webgpu",
    status: "fail",
    runtime: {
      loadMs: 6364.6,
      inferenceMs: 4988.2,
      totalMs: 11504.1,
      secondsPerAudioSecond: 6.235,
      grade: "very-slow",
    },
    quality: {
      envelopeDtwDistance: 0.508,
      durationRatio: 0.181,
      grade: "failed",
    },
    failure: "quality-duration-envelope",
  },
  {
    precision: "q4",
    device: "wasm",
    status: "pass",
    runtime: {
      loadMs: 4354.5,
      inferenceMs: 11196.6,
      totalMs: 15744.5,
      secondsPerAudioSecond: 2.845,
      grade: "very-slow",
    },
    quality: {
      envelopeDtwDistance: 0.276,
      durationRatio: 0.891,
      grade: "good",
    },
  },
  {
    precision: "q4",
    device: "webgpu",
    status: "fail",
    runtime: {
      loadMs: 4541.9,
      inferenceMs: null,
      totalMs: 7301.6,
      grade: "failed",
    },
    failure: "webgpu-session-error",
  },
  {
    precision: "q4f16",
    device: "wasm",
    status: "pass",
    runtime: {
      loadMs: 4422.7,
      inferenceMs: 11680.8,
      totalMs: 16242.1,
      secondsPerAudioSecond: 2.897,
      grade: "very-slow",
    },
    quality: {
      envelopeDtwDistance: 0.283,
      durationRatio: 0.913,
      grade: "good",
    },
  },
  {
    precision: "q4f16",
    device: "webgpu",
    status: "fail",
    runtime: {
      loadMs: null,
      inferenceMs: null,
      totalMs: 4353.9,
      grade: "failed",
    },
    failure: "webgpu-session-error",
  },
  {
    precision: "bnb4",
    device: "wasm",
    status: "fail",
    runtime: {
      loadMs: 4583.4,
      inferenceMs: 8820,
      totalMs: 13527,
      secondsPerAudioSecond: 2.278,
      grade: "very-slow",
    },
    quality: {
      envelopeDtwDistance: 0.539,
      durationRatio: 0.877,
      grade: "failed",
    },
    failure: "quality-envelope",
  },
  {
    precision: "bnb4",
    device: "webgpu",
    status: "fail",
    runtime: {
      loadMs: 4488.6,
      inferenceMs: 84768.6,
      totalMs: 89398.4,
      secondsPerAudioSecond: 4.014,
      grade: "very-slow",
    },
    quality: {
      envelopeDtwDistance: 0.696,
      durationRatio: 4.783,
      grade: "failed",
    },
    failure: "quality-duration-envelope",
  },
];
const MANIFEST = {
  contractVersion: "1.0",
  model: {
    id: ID,
    displayName: "SpeechT5 — English Text to Speech",
    provider: "Microsoft",
    providerUrl: "https://github.com/microsoft/SpeechT5",
    license: "MIT",
    licenseUrl: "https://github.com/microsoft/SpeechT5/blob/main/LICENSE",
    lastUpdated: "2023-11-08T14:37:23.000Z",
    sourceRepository: "microsoft/speecht5_tts",
    sourceRevision: "30fcde30f19b87502b8435427b5f5068e401d5f6",
    repository: REPO,
    artifactRevision: REV,
    artifactLastUpdated: "2025-08-27T22:30:58.000Z",
    task: "text-to-speech",
    description:
      "Microsoft SpeechT5 fine-tuned on LibriTTS for browser-local English text-to-speech. It uses a 512-value speaker embedding and returns 16 kHz mono PCM samples.",
    intendedUses: [
      "Local English narration and accessibility prototypes.",
      "Generating short, user-reviewed voice prompts in a browser.",
      "Researching speaker-conditioned text-to-speech with supplied embeddings.",
    ],
    limitations: [
      "The source model is trained for English; other languages and code-switching are unsupported.",
      "Names, acronyms, numbers, punctuation, and uncommon words may be pronounced incorrectly.",
      "Speaker embeddings influence voice characteristics but do not guarantee faithful identity reproduction.",
      "Synthetic audio may contain omissions, repetitions, artifacts, or misleading prosody and must not be used for impersonation, deception, or safety-critical communication.",
    ],
  },
  runtime: {
    engine: { name: "Transformers.js", version: "3.7.6" },
    precisions: SUPPORTED,
  },
  input: {
    description:
      "Non-empty English text plus an optional URL containing exactly 512 little-endian float32 speaker-embedding values.",
    alternatives: [
      {
        name: "text",
        fields: [
          {
            name: "text",
            type: "string",
            required: true,
            minLength: 1,
            description:
              "English text. The model supports at most 600 tokenizer positions; long passages should be split into sentences.",
          },
          {
            name: "speakerEmbedding",
            type: "string(URL)",
            required: false,
            minLength: 1,
            format: "url",
            description:
              "Optional 2,048-byte speaker embedding; defaults to the pinned benchmark voice.",
          },
        ],
      },
    ],
  },
  config: { model: [], generation: [] },
  output: {
    description: "16 kHz mono float32 PCM audio and its duration.",
    fields: [
      { name: "audio", type: "Float32Array", required: true, minItems: 1 },
      { name: "samplingRate", type: "integer", required: true, enum: [16000] },
      { name: "durationSeconds", type: "number", required: true, minimum: 0 },
    ],
    example: {
      audio: "Float32Array",
      samplingRate: 16000,
      durationSeconds: 2.4,
    },
  },
  operations: [
    "checkModelSupports",
    "init",
    "download",
    "generate",
    "clearMemory",
  ],
  benchmark: {
    report: "benchmarks/speecht5-tts.json",
    testedAt: "2026-10-23T12:52:43.471Z",
    fixture: {
      file: "speecht5-tts.json",
      url: "https://f005.backblazeb2.com/file/hubters-webai-public/assets/benchmarks/speecht5-tts.json",
      size: 2177,
      sha1: "b11a7a07ab6586b410c367c280b46c44edcff514",
      license: "CC0-1.0",
      reference:
        "64-bin absolute-amplitude envelope generated by FP32/WASM and checked into the fixture",
      speakerEmbedding: {
        file: "speecht5-speaker-embedding.bin",
        url: DEFAULT_SPEAKER,
        size: 2048,
        sha1: "7eca50ca7a6e271a2107f7f2db31c00d0bcdfb66",
        values: 512,
        sourceRepository: "Xenova/transformers.js-docs",
        sourceRevision: "542437c29cc5273cfd29e25c77696f2b349badd7",
      },
    },
    candidates: PRECISIONS,
    qualityGate:
      "Finite, non-clipped, non-silent 16 kHz audio of plausible duration whose normalized 64-bin amplitude envelope has dynamic-time-warping distance at most 0.45 from the checked-in stochastic FP32/WASM reference.",
    environment:
      "Isolated headless Google Chrome on macOS; Transformers.js 3.7.6",
    invalidatedBy: [
      "artifact revision change",
      "runtime version change",
      "fixture, speaker embedding, reference, or quality-gate change",
      "browser/backend behavior change",
    ],
    results: RESULTS,
  },

  samples: [
  {
    "id": "standard",
    "label": "Try sample text",
    "description": "Standard sample for SpeechT5 — English Text to Speech.",
    "userInput": {
      "text": "The quick brown fox jumps over the lazy dog near the river bank.",
      "speakerEmbedding": "/test-audio/speecht5-speaker-embedding.bin"
    },
    "assets": [
      {
        "field": "speakerEmbedding",
        "url": "https://f005.backblazeb2.com/file/hubters-webai-public/assets/benchmarks/speecht5-speaker-embedding.bin",
        "file": "model-samples/speecht5-speaker-embedding.bin",
        "mediaType": "application/octet-stream",
        "size": 2048,
        "sha1": "7eca50ca7a6e271a2107f7f2db31c00d0bcdfb66",
        "encoding": "blob-url"
      }
    ],
    "modelConfig": {},
    "generateConfig": {},
    "source": "https://f005.backblazeb2.com/file/hubters-webai-public/assets/benchmarks/speecht5-tts.json",
    "license": "CC0-1.0",
    "fixture": {
      "file": "model-samples/speecht5-tts.json",
      "url": "https://f005.backblazeb2.com/file/hubters-webai-public/assets/benchmarks/speecht5-tts.json",
      "size": 2177,
      "sha1": "b11a7a07ab6586b410c367c280b46c44edcff514"
    },
    "benchmarkReport": "benchmarks/speecht5-tts.json"
  }
],
};
const caps = {
  supportedModes: ["webai"],
  supportedPrecisions: Object.keys(SUPPORTED),
  supportedPrecisionsDevicesMap: SUPPORTED,
  doesSupportStreamGeneration: false,
  externalInterrupt: true,
  workerVersion: "v4",
  transformersJsVersion: "3.7.6",
  cacheModelId: REPO,
};
const state = { initializing: false, initialized: false, generating: false };
let tts = null;
const send = (requestId, type, data) =>
  self.postMessage({ requestId, type, data });
function configure(config = {}) {
  env.allowRemoteModels = config.allowRemoteModels ?? true;
  env.allowLocalModels = config.allowLocalModels ?? false;
  env.useBrowserCache = config.useBrowserCache ?? true;
  if (config.remoteHost) env.remoteHost = config.remoteHost;
  if (config.remotePathTemplate)
    env.remotePathTemplate = config.remotePathTemplate;
  if (config.localModelPath) env.localModelPath = config.localModelPath;
}
function validateInit(data) {
  const precision = data?.precision,
    device = data?.device,
    supported = SUPPORTED[precision];
  if (!supported) throw Error(`Unsupported precision: ${precision}`);
  if (!supported.supportedDevices.includes(device))
    throw Error(
      `Precision ${precision} is only supported on: ${supported.supportedDevices.join(", ")}`,
    );
  if (device === "webgpu" && !navigator.gpu) throw Error("WebGPU unavailable");
  return { precision, device };
}
async function clear() {
  if (tts?.dispose) await tts.dispose();
  tts = null;
  state.initialized = false;
}
async function load(data, requestId) {
  const { precision, device } = validateInit(data);
  await clear();
  tts = await pipeline("text-to-speech", REPO, {
    revision: REV,
    dtype: precision,
    device,
    progress_callback: (value) =>
      send(requestId, "downloadProgress", {
        ...value,
        progress: value.status === "ready" ? 100 : value.progress,
      }),
  });
}
function validateGeneration(data) {
  const text = data?.userInput?.text,
    speakerEmbedding = data?.userInput?.speakerEmbedding ?? DEFAULT_SPEAKER;
  if (typeof text !== "string" || !text.trim())
    throw Error("userInput.text must be a non-empty string");
  if (typeof speakerEmbedding !== "string" || !speakerEmbedding.trim())
    throw Error("userInput.speakerEmbedding must be a non-empty URL");
  return { text: text.trim(), speakerEmbedding };
}
async function generate(data) {
  const { text, speakerEmbedding } = validateGeneration(data),
    output = await tts(text, { speaker_embeddings: speakerEmbedding }),
    audio = output?.audio,
    samplingRate = output?.sampling_rate;
  if (
    !(audio instanceof Float32Array) ||
    !audio.length ||
    !audio.every(Number.isFinite) ||
    samplingRate !== 16000
  )
    throw Error("Model returned invalid 16 kHz audio");
  return { audio, samplingRate, durationSeconds: audio.length / samplingRate };
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
        if (!supported)
          throw Error(`Unsupported precision: ${data?.precision}`);
        await load(
          {
            ...data,
            device: supported.supportedDevices.includes("wasm")
              ? "wasm"
              : supported.supportedDevices[0],
          },
          requestId,
        );
        await tts("Hi.", { speaker_embeddings: DEFAULT_SPEAKER });
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
        if (!state.initialized || !tts)
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
        throw Error("Stream generation is not supported by this worker");
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
