import {
  pipeline,
  env,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0";
import { MPEGDecoderWebWorker } from "https://cdn.jsdelivr.net/npm/mpg123-decoder@1.0.0/+esm";
import wavDecoder from "https://cdn.jsdelivr.net/npm/wav-decoder@1.3.0/+esm";

// ==========================================
// Configuration
// ==========================================
const MODEL_CONFIG_FIELDS = [
  { name: "language", type: "string", default: "en", description: "Spoken-language name or code; omit for automatic detection." },
  { name: "task", type: "string", default: "transcribe", enum: ["transcribe", "translate"], description: "Preserve the source language or translate speech to English." },
  { name: "return_timestamps", type: "'word' | boolean", default: "word", enum: ["word", true, false], description: "Word timestamps, segment timestamps, or text only." },
  { name: "condition_on_prev_tokens", type: "boolean", default: true, description: "Use prior decoded tokens as context across chunks." },
  { name: "no_speech_threshold", type: "number", default: 0.6, minimum: 0, maximum: 1, description: "Probability threshold used to classify a segment as silence." },
  { name: "num_beams", type: "integer", default: 1, minimum: 1, description: "Beam-search width; larger values trade speed for search quality." },
];

const GENERATION_CONFIG_FIELDS = [
  { name: "chunk_length_s", type: "number", default: 29, minimum: 1, maximum: 29, description: "Long-audio chunk duration in seconds." },
  { name: "stride_length_s", type: "number", default: 5, minimum: 0, description: "Overlap on each side; cannot exceed half the chunk duration." },
];

function defaultsFrom(fields) {
  return Object.fromEntries(fields.map(({ name, default: value }) => [name, value]));
}

const CONFIG = {
  MODEL_ID: "whisper-small",
  MODEL_REPOSITORY: "onnx-community/whisper-small_timestamped",
  MODEL_REVISION: "65caa70f294b46e1c33ff820aae6b16d048ab818",
  WORKER_VERSION: "v4",
  TRANSFORMERS_JS_VERSION: "4.2.0",
  EXTERNAL_INTERRUPT: true,
  SUPPORTED_MODES: ["webai"],
  SUPPORTED_PRECISIONS_DEVICES_MAP: {
    bnb4: { size: 291617358, weightsSize: 287215655, supportedDevices: ["wasm", "webgpu"], modelKeys: ["encoder_model_bnb4.onnx", "decoder_model_merged_bnb4.onnx"], dtype: { encoder_model: "bnb4", decoder_model_merged: "bnb4" } },
    q4: { size: 304001406, weightsSize: 299599703, supportedDevices: ["wasm", "webgpu"], modelKeys: ["encoder_model_q4.onnx", "decoder_model_merged_q4.onnx"], dtype: { encoder_model: "q4", decoder_model_merged: "q4" } },
    q8: { size: 253437951, weightsSize: 249036248, supportedDevices: ["webgpu"], modelKeys: ["encoder_model_quantized.onnx", "decoder_model_merged_quantized.onnx"], dtype: { encoder_model: "q8", decoder_model_merged: "q8" } },
    int8: { size: 253437918, weightsSize: 249036215, supportedDevices: ["webgpu"], modelKeys: ["encoder_model_int8.onnx", "decoder_model_merged_int8.onnx"], dtype: { encoder_model: "int8", decoder_model_merged: "int8" } },
    uint8: { size: 253438012, weightsSize: 249036309, supportedDevices: ["webgpu"], modelKeys: ["encoder_model_uint8.onnx", "decoder_model_merged_uint8.onnx"], dtype: { encoder_model: "uint8", decoder_model_merged: "uint8" } },
    fp32: { size: 972849700, weightsSize: 968447997, supportedDevices: ["wasm", "webgpu"], modelKeys: ["encoder_model.onnx", "decoder_model_merged.onnx"], dtype: { encoder_model: "fp32", decoder_model_merged: "fp32" } },
  },
  DEFAULT_MODEL_CONFIG: defaultsFrom(MODEL_CONFIG_FIELDS),
  DEFAULT_GENERATION_CONFIG: defaultsFrom(GENERATION_CONFIG_FIELDS),
};

const BENCHMARK_RESULTS = [
  { precision: "fp32", device: "wasm", status: "pass", runtime: { loadMs: 101379.7, inferenceMs: 20311.5, totalMs: 121691.2, grade: "slow", realTimeFactor: 0.6887, loadGrade: "slow" }, quality: { grade: "acceptable", wordErrorRate: 0.1515, gibberish: false } },
  { precision: "fp32", device: "webgpu", status: "pass", runtime: { loadMs: 100526.2, inferenceMs: 4301.2, totalMs: 104827.6, grade: "excellent", realTimeFactor: 0.1458, loadGrade: "slow" }, quality: { grade: "acceptable", wordErrorRate: 0.1515, gibberish: false } },
  { precision: "fp16", device: "wasm", status: "fail", runtime: { loadMs: null, inferenceMs: null, totalMs: 42152.5, grade: "failed" }, failure: "onnx-session-error" },
  { precision: "fp16", device: "webgpu", status: "fail", runtime: { loadMs: 53530.3, inferenceMs: 38963.8, totalMs: 92494.2, grade: "very-slow", realTimeFactor: 1.3212, loadGrade: "slow" }, quality: { grade: "failed", wordErrorRate: 3.9091, gibberish: true }, failure: "quality-gibberish" },
  { precision: "int8", device: "wasm", status: "fail", runtime: { loadMs: null, inferenceMs: null, totalMs: 34916.4, grade: "failed" }, failure: "onnx-quantization-error" },
  { precision: "int8", device: "webgpu", status: "pass", runtime: { loadMs: 145063.1, inferenceMs: 23608.3, totalMs: 168671.4, grade: "slow", realTimeFactor: 0.8005, loadGrade: "slow" }, quality: { grade: "acceptable", wordErrorRate: 0.1667, gibberish: false } },
  { precision: "uint8", device: "wasm", status: "fail", runtime: { loadMs: null, inferenceMs: null, totalMs: 98727.8, grade: "failed" }, failure: "onnx-quantization-error" },
  { precision: "uint8", device: "webgpu", status: "pass", runtime: { loadMs: 119272.5, inferenceMs: 24282.1, totalMs: 143554.6, grade: "slow", realTimeFactor: 0.8233, loadGrade: "slow" }, quality: { grade: "acceptable", wordErrorRate: 0.1515, gibberish: false } },
  { precision: "q8", device: "wasm", status: "fail", runtime: { loadMs: null, inferenceMs: null, totalMs: 40520.5, grade: "failed" }, failure: "onnx-quantization-error" },
  { precision: "q8", device: "webgpu", status: "pass", runtime: { loadMs: 38966.5, inferenceMs: 21720.7, totalMs: 60687.3, grade: "slow", realTimeFactor: 0.7365, loadGrade: "slow" }, quality: { grade: "acceptable", wordErrorRate: 0.1667, gibberish: false } },
  { precision: "q4", device: "wasm", status: "pass", runtime: { loadMs: 43001.9, inferenceMs: 42684.9, totalMs: 85686.8, grade: "very-slow", realTimeFactor: 1.4473, loadGrade: "slow" }, quality: { grade: "excellent", wordErrorRate: 0.1061, gibberish: false } },
  { precision: "q4", device: "webgpu", status: "pass", runtime: { loadMs: 42880.7, inferenceMs: 3861.4, totalMs: 46742.1, grade: "excellent", realTimeFactor: 0.1309, loadGrade: "slow" }, quality: { grade: "excellent", wordErrorRate: 0.1061, gibberish: false } },
  { precision: "q4f16", device: "wasm", status: "fail", runtime: { loadMs: null, inferenceMs: null, totalMs: 42401.1, grade: "failed" }, failure: "onnx-session-error" },
  { precision: "q4f16", device: "webgpu", status: "fail", runtime: { loadMs: 38809.7, inferenceMs: 30753.1, totalMs: 69562.9, grade: "very-slow", realTimeFactor: 1.0428, loadGrade: "slow" }, quality: { grade: "failed", wordErrorRate: 8.5303, gibberish: true }, failure: "quality-gibberish" },
  { precision: "bnb4", device: "wasm", status: "pass", runtime: { loadMs: 92612.2, inferenceMs: 35025.9, totalMs: 127638.1, grade: "very-slow", realTimeFactor: 1.1876, loadGrade: "slow" }, quality: { grade: "acceptable", wordErrorRate: 0.1818, gibberish: false } },
  { precision: "bnb4", device: "webgpu", status: "pass", runtime: { loadMs: 87098.9, inferenceMs: 34800.4, totalMs: 121899.4, grade: "very-slow", realTimeFactor: 1.18, loadGrade: "slow" }, quality: { grade: "acceptable", wordErrorRate: 0.1818, gibberish: false } },
];

const MANIFEST = {
  contractVersion: "1.0",
  model: {
    id: CONFIG.MODEL_ID,
    displayName: "Whisper Small — Timestamped",
    provider: "OpenAI",
    providerUrl: "https://openai.com/",
    license: "Apache-2.0",
    licenseUrl: "https://www.apache.org/licenses/LICENSE-2.0.txt",
    lastUpdated: "2024-02-29T10:57:38.000Z",
    sourceRepository: "openai/whisper-small",
    repository: CONFIG.MODEL_REPOSITORY,
    artifactRevision: CONFIG.MODEL_REVISION,
    artifactLastUpdated: "2025-03-05T00:13:06.000Z",
    task: "automatic-speech-recognition",
    description: "A multilingual Whisper Small speech-recognition model that runs locally in the browser and provides higher-capacity transcription, translation to English, and optional word- or segment-level timestamps.",
    intendedUses: [
      "Private, client-side transcription of short and medium-length speech recordings.",
      "Generating word- or segment-timestamped transcripts for captions and search.",
      "Translating supported spoken languages into English text.",
      "Prototyping browser speech-recognition features without a hosted inference API.",
    ],
    limitations: [
      "Small requires substantially more download, memory, and compute than Whisper Tiny or Base.",
      "Word timestamps are approximate DTW alignments and may drift around silence, noise, or overlapping speech.",
      "Accuracy varies by language, accent, recording quality, background noise, and domain vocabulary.",
      "Long recordings are chunked and may contain boundary repetition or omissions.",
      "Generated transcripts must not be treated as authoritative for safety-critical decisions.",
    ],
  },
  runtime: {
    precisions: CONFIG.SUPPORTED_PRECISIONS_DEVICES_MAP,
  },
  input: {
    description: "Provide exactly one 16 kHz mono Float32Array or a browser blob URL containing WAV/MP3 audio.",
    alternatives: [
      {
        name: "decoded audio",
        description: "Fastest path; audio is already decoded and resampled by the application.",
        fields: [{ name: "audio", type: "Float32Array", required: true, description: "Mono PCM samples at 16 kHz in the range -1 to 1." }],
      },
      {
        name: "audio blob",
        description: "Worker fetches and decodes a WAV or MP3 browser object URL.",
        fields: [{ name: "audio_blob_url", type: "string(blob URL)", required: true, description: "A blob: URL accessible from this worker." }],
      },
    ],
  },
  config: {
    model: MODEL_CONFIG_FIELDS,
    generation: GENERATION_CONFIG_FIELDS,
  },
  output: {
    description: "The worker returns model-specific transcription data. WebAI wraps it as { result, runtime: { durationMs } }.",
    fields: [
      { name: "result", type: "string", required: true, description: "Complete transcription text." },
      { name: "chunks", type: "Array<{ text: string, timestamp: [number, number | null] }>", description: "Timestamped words or segments when requested." },
    ],
    example: { result: " Hello world.", chunks: [{ text: " Hello", timestamp: [0.42, 0.88] }] },
  },
  operations: ["checkModelSupports", "init", "download", "generate", "clearMemory"],
  benchmark: {
    report: "benchmarks/whisper-small_timestamped.json",
    testedAt: "2026-09-07T16:09:32.719Z",
    fixture: {
      file: "whisper-standard.mp3",
      url: "https://f005.backblazeb2.com/file/hubters-webai-public/assets/benchmarks/whisper-standard.mp3",
      size: 471925,
      sha1: "5faad1507b5d6bc61a39ca3184db5a7b554a7479",
      durationSeconds: 29.49225,
    },
    note: "Measured values are machine/browser-specific; regenerate the report on target hardware.",
    environment: "Headless Chrome 151 on macOS; Transformers.js 4.2.0",
    invalidatedBy: [
      "model artifact revision change",
      "Transformers.js or ONNX Runtime version change",
      "benchmark fixture or quality-gate change",
      "target browser/backend behavior change",
    ],
    results: BENCHMARK_RESULTS,
  },
};

// ==========================================
// Global state
// ==========================================
let MODEL;
let modelState = {
  isInitializing: false,
  isInitialized: false,
  isGenerating: false,
};

// ==========================================
// Environment Setup
// ==========================================
function initializeEnvironment(workerConfig = {}) {
  env.allowRemoteModels = workerConfig.allowRemoteModels ?? true;
  env.allowLocalModels = workerConfig.allowLocalModels ?? false;
  env.useBrowserCache = workerConfig.useBrowserCache ?? true;

  if (workerConfig.remoteHost) {
    env.remoteHost = workerConfig.remoteHost;
  }
  if (workerConfig.remotePathTemplate) {
    env.remotePathTemplate = workerConfig.remotePathTemplate;
  }
  if (workerConfig.localModelPath) {
    env.localModelPath = workerConfig.localModelPath;
  }

  if (workerConfig.modelId) {
    MODEL.model_id = workerConfig.modelId;
    MODEL.revision = workerConfig.modelRevision;
  } else if (workerConfig.modelRevision) {
    MODEL.revision = workerConfig.modelRevision;
  }
}

// ==========================================
// Utility Functions
// ==========================================
function checkModelSupports() {
  return {
    manifest: {
      ...MANIFEST,
      model: {
        ...MANIFEST.model,
        repository: MODEL.model_id,
        artifactRevision: MODEL.revision ?? "unversioned",
      },
    },
    supportedModes: CONFIG.SUPPORTED_MODES,
    supportedPrecisions: Object.keys(CONFIG.SUPPORTED_PRECISIONS_DEVICES_MAP),
    supportedPrecisionsDevicesMap: CONFIG.SUPPORTED_PRECISIONS_DEVICES_MAP,
    doesSupportStreamGeneration: false,
    externalInterrupt: CONFIG.EXTERNAL_INTERRUPT,
    workerVersion: CONFIG.WORKER_VERSION,
    transformersJsVersion: CONFIG.TRANSFORMERS_JS_VERSION,
    cacheModelId: MODEL.model_id,
  };
}

function mergeConfigs(defaults, userConfig) {
  return { ...defaults, ...userConfig };
}

function validateTranscriptionConfigs(modelConfig, generationConfig) {
  if (!["transcribe", "translate"].includes(modelConfig.task)) {
    throw new Error("modelConfig.task must be 'transcribe' or 'translate'.");
  }
  if (!["word", true, false].includes(modelConfig.return_timestamps)) {
    throw new Error("modelConfig.return_timestamps must be 'word', true, or false.");
  }
  if (
    !Number.isFinite(modelConfig.no_speech_threshold) ||
    modelConfig.no_speech_threshold < 0 ||
    modelConfig.no_speech_threshold > 1
  ) {
    throw new Error("modelConfig.no_speech_threshold must be between 0 and 1.");
  }
  if (!Number.isInteger(modelConfig.num_beams) || modelConfig.num_beams < 1) {
    throw new Error("modelConfig.num_beams must be a positive integer.");
  }
  if (
    !Number.isFinite(generationConfig.chunk_length_s) ||
    generationConfig.chunk_length_s <= 0
  ) {
    throw new Error("generateConfig.chunk_length_s must be greater than 0.");
  }
  if (
    !Number.isFinite(generationConfig.stride_length_s) ||
    generationConfig.stride_length_s < 0 ||
    generationConfig.stride_length_s * 2 > generationConfig.chunk_length_s
  ) {
    throw new Error(
      "generateConfig.stride_length_s must be non-negative and no more than half the chunk length.",
    );
  }
}

function postResponse(requestId, type, data) {
  self.postMessage({ requestId, type, data });
}

function postProgress(requestId, progress) {
  postResponse(requestId, "downloadProgress", {
    ...progress,
    progress: progress.status === "ready" ? 100 : progress.progress,
  });
}

// ==========================================
// Audio Processing - Optimized
// ==========================================

/**
 * Convert stereo to mono with optimized direct iteration
 */
function convertToMono(channelData, sampleCount) {
  if (channelData.length === 1) {
    return channelData[0];
  }

  const monoData = new Float32Array(sampleCount);
  const left = channelData[0];
  const right = channelData[1];

  for (let i = 0; i < sampleCount; i++) {
    monoData[i] = (left[i] + right[i]) * 0.5;
  }

  return monoData;
}

/**
 * Resample audio with linear interpolation for better quality
 */
function resampleAudio(audioData, sourceSampleRate, targetSampleRate) {
  if (sourceSampleRate === targetSampleRate) {
    return audioData;
  }

  const resampleRatio = sourceSampleRate / targetSampleRate;
  const resampledLength = Math.floor(audioData.length / resampleRatio);
  const resampledData = new Float32Array(resampledLength);

  for (let i = 0; i < resampledLength; i++) {
    const srcIndex = i * resampleRatio;
    const srcIndexFloor = Math.floor(srcIndex);
    const frac = srcIndex - srcIndexFloor;

    if (srcIndexFloor + 1 < audioData.length) {
      // Linear interpolation
      resampledData[i] =
        audioData[srcIndexFloor] * (1 - frac) +
        audioData[srcIndexFloor + 1] * frac;
    } else {
      resampledData[i] = audioData[srcIndexFloor];
    }
  }

  return resampledData;
}

/**
 * Process audio data with proper resource cleanup
 */
async function processAudioData(arrayBuffer, targetSampleRate = 16000) {
  // Try WAV first, then MP3
  try {
    const audioData = await wavDecoder.decode(arrayBuffer);
    const monoData = convertToMono(
      audioData.channelData,
      audioData.channelData[0].length,
    );
    return resampleAudio(monoData, audioData.sampleRate, targetSampleRate);
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
  } catch (wavError) {
    // Try MP3 decoder
    const decoder = new MPEGDecoderWebWorker();
    try {
      await decoder.ready;
      const { channelData, samplesDecoded, sampleRate } =
        await decoder.decode(arrayBuffer);
      const monoData = convertToMono(channelData, samplesDecoded);
      return resampleAudio(monoData, sampleRate, targetSampleRate);
    } finally {
      await decoder.free(); // Always cleanup
    }
  }
}

// ==========================================
// Model Class
// ==========================================
class WebAIModel {
  constructor() {
    this.model_id = CONFIG.MODEL_REPOSITORY;
    this.revision = CONFIG.MODEL_REVISION;
    this.pipe = null;
    this.precision = null;
    this.device = null;
  }

  setDefaults({ precision, device }) {
    const precisionConfig = CONFIG.SUPPORTED_PRECISIONS_DEVICES_MAP[precision];
    if (!precisionConfig) {
      throw new Error(`Unsupported precision: ${precision}`);
    }
    if (!precisionConfig.supportedDevices.includes(device)) {
      throw new Error(
        `Device ${device} is not supported for precision ${precision}`,
      );
    }
    this.precision = precision;
    this.device = device;
  }

  async downloadModel(precision, requestId) {
    if (!precision) {
      throw new Error("Precision must be set to download the model");
    }

    const precisionConfig = CONFIG.SUPPORTED_PRECISIONS_DEVICES_MAP[precision];
    if (!precisionConfig) throw new Error(`Unsupported precision: ${precision}`);
    const downloadedPipeline = await pipeline(
      "automatic-speech-recognition",
      this.model_id,
      {
        device: precisionConfig.supportedDevices.includes("wasm") ? "wasm" : "webgpu",
        dtype: precisionConfig.dtype,
        ...(this.revision ? { revision: this.revision } : {}),
        progress_callback: (progress) => {
          postProgress(requestId, progress);
        },
      },
    );

    if (typeof downloadedPipeline.dispose === "function") {
      await downloadedPipeline.dispose();
    }

    postResponse(requestId, "download", { status: "success" });
  }

  async loadModel(requestId) {
    if (!this.precision || !this.device) {
      throw new Error(
        "Precision and device must be set via init before loading the model",
      );
    }

    if (this.pipe && typeof this.pipe.dispose === "function") {
      await this.pipe.dispose();
      this.pipe = null;
    }

    // pipeline() loads from browser cache when available and downloads otherwise.
    this.pipe = await pipeline("automatic-speech-recognition", this.model_id, {
      device: this.device,
      dtype: CONFIG.SUPPORTED_PRECISIONS_DEVICES_MAP[this.precision].dtype,
      ...(this.revision ? { revision: this.revision } : {}),
      progress_callback: (progress) => {
        postProgress(requestId, progress);
      },
    });
  }

  async generate(data, requestId) {
    const { userInput, modelConfig, generateConfig } = data ?? {};

    if (!this.pipe || !this.precision || !this.device) {
      throw new Error(
        "Model not initialized. You must call init first to set precision and device and load the model first.",
      );
    }

    if (modelState.isGenerating) {
      throw new Error(
        "A generation is already in progress. Please wait or interrupt the current generation.",
      );
    }

    modelState.isGenerating = true;

    try {
      const finalGenerationConfig = mergeConfigs(
        CONFIG.DEFAULT_GENERATION_CONFIG,
        generateConfig,
      );
      let audioData;

      if (!userInput || typeof userInput !== "object") {
        throw new Error("userInput must be an object containing audio input.");
      }

      if (userInput.audio instanceof Float32Array) {
        audioData = userInput.audio;
      } else if (!userInput.audio_blob_url?.startsWith("blob:")) {
        throw new Error(
          "Provide userInput.audio as Float32Array or a valid audio_blob_url.",
        );
      } else {
        const response = await fetch(userInput.audio_blob_url);
        if (!response.ok) {
          throw new Error(
            `Failed to fetch audio data from blob URL: ${response.statusText}`,
          );
        }
        const audioArrayBuffer = await response.arrayBuffer();
        try {
          audioData = await processAudioData(audioArrayBuffer, 16000);
        } catch (error) {
          throw new Error(`Preprocessing audio data failed: ${error.message}`);
        }
      }

      // Run transcription
      const finalModelConfig = mergeConfigs(
        CONFIG.DEFAULT_MODEL_CONFIG,
        modelConfig,
      );
      validateTranscriptionConfigs(finalModelConfig, finalGenerationConfig);
      const rawOutputs = await this.pipe(audioData, {
        ...finalModelConfig,
        chunk_length_s: finalGenerationConfig.chunk_length_s,
        stride_length_s: finalGenerationConfig.stride_length_s,
      });

      const finalOutputs = {
        result: rawOutputs.text,
        ...rawOutputs,
      };
      delete finalOutputs.text;

      postResponse(requestId, "generated", {
        status: "success",
        result: finalOutputs,
      });
    } finally {
      modelState.isGenerating = false;
    }
  }

  generateStream() {
    throw new Error(
      "Stream generation is not supported for this model, check documentation for how to implement real-time transcription.",
    );
  }

  async clearMemory() {
    if (this.pipe) {
      if (typeof this.pipe.dispose === "function") {
        await this.pipe.dispose();
      }
      this.pipe = null;
    }
  }
}

// ==========================================
// Message Handler
// ==========================================
async function handleMessage(event) {
  const { requestId, type, data } = event.data ?? {};
  console.log("Received message from main thread:", type);

  try {
    switch (type) {
      case "init":
        if (modelState.isInitializing) {
          throw new Error("Model is already initializing. Please wait.");
        }
        if (!data.precision || !data.device) {
          throw new Error(
            "Init message must contain both precision and device parameters",
          );
        }

        modelState.isInitializing = true;
        modelState.isInitialized = false;
        try {
          MODEL.setDefaults(data);
          await MODEL.loadModel(requestId);
          modelState.isInitialized = true;
          postResponse(requestId, "init", {
            status: "success",
            precision: MODEL.precision,
            device: MODEL.device,
          });
        } finally {
          modelState.isInitializing = false;
        }
        break;

      case "checkModelSupports":
        initializeEnvironment(data?.workerConfig);
        postResponse(requestId, "checkModelSupports", checkModelSupports());
        break;

      case "download":
        await MODEL.downloadModel(data.precision, requestId);
        break;

      case "generate":
        if (!modelState.isInitialized) {
          throw new Error(
            "Model not initialized. You must call init first to set precision and device.",
          );
        }
        await MODEL.generate(data, requestId);
        break;

      case "generateStream":
        MODEL.generateStream();
        break;

      case "clearMemory":
        await MODEL.clearMemory();
        modelState.isInitialized = false;
        modelState.isGenerating = false;
        postResponse(requestId, "clearMemory", { status: "success" });
        break;

      default:
        throw new Error(`Unknown message type: ${type}`);
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    postResponse(requestId, "error", {
      message: message || "Worker operation failed",
      context: type,
    });
  }
}

// ==========================================
// Initialize Application
// ==========================================
function initializeApp() {
  // Global error handlers
  self.addEventListener("error", (event) => {
    self.postMessage({
      type: "error",
      data: {
        message: event.message,
        context: "global_error",
      },
    });
    event.preventDefault();
  });

  self.addEventListener("unhandledrejection", (event) => {
    self.postMessage({
      type: "error",
      data: {
        message: event.reason?.message || "Unhandled Promise Rejection",
        context: "unhandled_promise_rejection",
      },
    });
    event.preventDefault();
  });

  MODEL = new WebAIModel();

  self.addEventListener("message", handleMessage);

  console.log(
    `[WebAI Worker] Initialized worker: ${CONFIG.MODEL_ID} (Worker Version: ${CONFIG.WORKER_VERSION}, Transformers.js: ${CONFIG.TRANSFORMERS_JS_VERSION})`
  );

  self.postMessage({
    type: "worker initialized",
    data: {
      success: true,
      message: `Web worker initialized successfully (${CONFIG.WORKER_VERSION})`,
      workerVersion: CONFIG.WORKER_VERSION,
      transformersJsVersion: CONFIG.TRANSFORMERS_JS_VERSION,
      modelId: CONFIG.MODEL_ID,
    },
  });
}

initializeApp();
