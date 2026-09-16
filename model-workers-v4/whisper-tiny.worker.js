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
  MODEL_ID: "whisper-tiny",
  MODEL_REPOSITORY: "onnx-community/whisper-tiny_timestamped",
  MODEL_REVISION: "517244293732ee2d58139af5814231b7e6830a0d",
  WORKER_VERSION: "v4",
  TRANSFORMERS_JS_VERSION: "4.2.0",
  EXTERNAL_INTERRUPT: true,
  SUPPORTED_MODES: ["webai"],
  SUPPORTED_PRECISIONS_DEVICES_MAP: {
    q4: {
      size: 100224019,
      weightsSize: 95824473,
      supportedDevices: ["wasm", "webgpu"],
      modelKeys: ["encoder_model_q4.onnx", "decoder_model_merged_q4.onnx"],
      dtype: { encoder_model: "q4", decoder_model_merged: "q4" },
    },
    q8: {
      size: 45226919,
      weightsSize: 40827373,
      supportedDevices: ["webgpu"],
      modelKeys: ["encoder_model_quantized.onnx", "decoder_model_merged_quantized.onnx"],
      dtype: { encoder_model: "q8", decoder_model_merged: "q8" },
    },
    uint8: {
      size: 45226949,
      weightsSize: 40827403,
      supportedDevices: ["webgpu"],
      modelKeys: ["encoder_model_uint8.onnx", "decoder_model_merged_uint8.onnx"],
      dtype: { encoder_model: "uint8", decoder_model_merged: "uint8" },
    },
    fp32: {
      size: 155958188,
      weightsSize: 151558642,
      modelKeys: ["encoder_model.onnx", "decoder_model_merged.onnx"],
      supportedDevices: ["wasm", "webgpu"],
      dtype: { encoder_model: "fp32", decoder_model_merged: "fp32" },
    },
  },
  DEFAULT_MODEL_CONFIG: defaultsFrom(MODEL_CONFIG_FIELDS),
  DEFAULT_GENERATION_CONFIG: defaultsFrom(GENERATION_CONFIG_FIELDS),
};

const BENCHMARK_RESULTS = [
  { precision: "fp32", device: "wasm", status: "pass", runtime: { loadMs: 24783.6, inferenceMs: 3149.2, totalMs: 27932.8, grade: "excellent", realTimeFactor: 0.1068, loadGrade: "moderate" }, quality: { grade: "excellent", wordErrorRate: 0.0758, gibberish: false } },
  { precision: "fp32", device: "webgpu", status: "pass", runtime: { loadMs: 18944.6, inferenceMs: 1891.3, totalMs: 20835.9, grade: "excellent", realTimeFactor: 0.0641, loadGrade: "moderate" }, quality: { grade: "excellent", wordErrorRate: 0.0758, gibberish: false } },
  { precision: "fp16", device: "wasm", status: "fail", runtime: { loadMs: null, inferenceMs: null, totalMs: 8233.5, grade: "failed" }, failure: "onnx-session-error" },
  { precision: "fp16", device: "webgpu", status: "fail", runtime: { loadMs: 54273.8, inferenceMs: 6430.6, totalMs: 60704.4, grade: "excellent", realTimeFactor: 0.218, loadGrade: "slow" }, quality: { grade: "failed", wordErrorRate: 2.2273, gibberish: true }, failure: "quality-gibberish" },
  { precision: "int8", device: "wasm", status: "fail", runtime: { loadMs: null, inferenceMs: null, totalMs: 7714.2, grade: "failed" }, failure: "onnx-quantization-error" },
  { precision: "int8", device: "webgpu", status: "fail", runtime: { loadMs: 7960.3, inferenceMs: 4833.6, totalMs: 12793.9, grade: "excellent", realTimeFactor: 0.1639, loadGrade: "fast" }, quality: { grade: "failed", wordErrorRate: 0.3939, gibberish: false }, failure: "quality-word-error-rate" },
  { precision: "uint8", device: "wasm", status: "fail", runtime: { loadMs: null, inferenceMs: null, totalMs: 8382.7, grade: "failed" }, failure: "onnx-quantization-error" },
  { precision: "uint8", device: "webgpu", status: "pass", runtime: { loadMs: 8529.1, inferenceMs: 4703.9, totalMs: 13233, grade: "excellent", realTimeFactor: 0.1595, loadGrade: "fast" }, quality: { grade: "acceptable", wordErrorRate: 0.303, gibberish: false } },
  { precision: "q8", device: "wasm", status: "fail", runtime: { loadMs: null, inferenceMs: null, totalMs: 7820.3, grade: "failed" }, failure: "onnx-quantization-error" },
  { precision: "q8", device: "webgpu", status: "pass", runtime: { loadMs: 8701.6, inferenceMs: 4810.2, totalMs: 13511.8, grade: "excellent", realTimeFactor: 0.1631, loadGrade: "fast" }, quality: { grade: "acceptable", wordErrorRate: 0.303, gibberish: false } },
  { precision: "q4", device: "wasm", status: "pass", runtime: { loadMs: 13428.9, inferenceMs: 4539.5, totalMs: 17968.5, grade: "excellent", realTimeFactor: 0.1539, loadGrade: "moderate" }, quality: { grade: "acceptable", wordErrorRate: 0.197, gibberish: false } },
  { precision: "q4", device: "webgpu", status: "pass", runtime: { loadMs: 11271.9, inferenceMs: 1839.2, totalMs: 13111.2, grade: "excellent", realTimeFactor: 0.0624, loadGrade: "moderate" }, quality: { grade: "acceptable", wordErrorRate: 0.197, gibberish: false } },
  { precision: "q4f16", device: "wasm", status: "fail", runtime: { loadMs: null, inferenceMs: null, totalMs: 7001, grade: "failed" }, failure: "onnx-session-error" },
  { precision: "q4f16", device: "webgpu", status: "fail", runtime: { loadMs: 17766.5, inferenceMs: 6469.5, totalMs: 24236, grade: "excellent", realTimeFactor: 0.2194, loadGrade: "moderate" }, quality: { grade: "failed", wordErrorRate: 0.6818, gibberish: false }, failure: "quality-word-error-rate" },
  { precision: "bnb4", device: "wasm", status: "fail", runtime: { loadMs: 16761.7, inferenceMs: 4824.4, totalMs: 21586.1, grade: "excellent", realTimeFactor: 0.1636, loadGrade: "moderate" }, quality: { grade: "failed", wordErrorRate: 0.2576, gibberish: true }, failure: "quality-gibberish" },
  { precision: "bnb4", device: "webgpu", status: "fail", runtime: { loadMs: 13294.8, inferenceMs: 6197.5, totalMs: 19492.3, grade: "excellent", realTimeFactor: 0.2101, loadGrade: "moderate" }, quality: { grade: "failed", wordErrorRate: 0.2576, gibberish: true }, failure: "quality-gibberish" },
];

const MANIFEST = {
  contractVersion: "1.0",
  model: {
    id: CONFIG.MODEL_ID,
    displayName: "Whisper Tiny — Timestamped",
    provider: "OpenAI",
    providerUrl: "https://openai.com/",
    license: "Apache-2.0",
    licenseUrl: "https://www.apache.org/licenses/LICENSE-2.0.txt",
    lastUpdated: "2024-02-29T10:57:33.000Z",
    sourceRepository: "openai/whisper-tiny",
    repository: CONFIG.MODEL_REPOSITORY,
    artifactRevision: CONFIG.MODEL_REVISION,
    artifactLastUpdated: "2025-03-05T00:13:43.000Z",
    task: "automatic-speech-recognition",
    description: "A compact multilingual Whisper speech-recognition model that runs locally in the browser and supports transcription, translation to English, and optional word- or segment-level timestamps.",
    intendedUses: [
      "Private, client-side transcription of short and medium-length speech recordings.",
      "Generating word- or segment-timestamped transcripts for captions and search.",
      "Translating supported spoken languages into English text.",
      "Prototyping browser speech-recognition features without a hosted inference API.",
    ],
    limitations: [
      "Tiny prioritizes download size and speed over the accuracy of larger Whisper variants.",
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
    report: "benchmarks/whisper-tiny_timestamped.json",
    testedAt: "2026-09-07T15:06:36.366Z",
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
