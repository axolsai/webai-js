import {
  pipeline,
  env,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@latest";
import { MPEGDecoderWebWorker } from "https://cdn.jsdelivr.net/npm/mpg123-decoder@1.0.0/+esm";
import wavDecoder from "https://cdn.jsdelivr.net/npm/wav-decoder@1.3.0/+esm";

// ==========================================
// Configuration
// ==========================================
const CONFIG = {
  MODEL_ID: "whisper-large-v3-turbo",
  EXTERNAL_INTERRUPT: true,
  SUPPORTED_MODES: ["webai"],
  SUPPORTED_PRECISIONS_DEVICES_MAP: {
    q4: {
      size: 759211851,
      modelKeys: ["encoder_model_q4.onnx", "decoder_model_merged_q4.onnx"],
      supportedDevices: ["webgpu"],
      speed: { webgpu: 6.5 },
    },
    q4f16: {
      size: 563601377,
      modelKeys: [
        "encoder_model_q4f16.onnx",
        "decoder_model_merged_q4f16.onnx",
      ],
      supportedDevices: ["webgpu"],
      speed: { webgpu: 7 },
    },
    fp16: {
      size: 1618692224,
      modelKeys: ["encoder_model_fp16.onnx", "decoder_model_merged_fp16.onnx"],
      supportedDevices: ["webgpu"],
      speed: { webgpu: 8 },
    },
  },
  DEFAULT_MODEL_CONFIG: {
    language: "en",
    task: "transcribe",
    return_timestamps: true,
    condition_on_prev_tokens: true,
    no_speech_threshold: 0.6,
    num_beans: 1,
  },
  DEFAULT_GENERATION_CONFIG: {
    chunk_length_s: 30,
    target_sample_rate: 16000,
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
function initializeEnvironment() {
  env.allowRemoteModels = true;
  env.remoteHost = "https://assets.axolsai.com";
  env.remotePathTemplate = "/models/{model}/model-repo/";
}

// ==========================================
// Utility Functions
// ==========================================
function checkModelSupports() {
  return {
    supportedModes: CONFIG.SUPPORTED_MODES,
    supportedPrecisions: Object.keys(CONFIG.SUPPORTED_PRECISIONS_DEVICES_MAP),
    supportedPrecisionsDevicesMap: CONFIG.SUPPORTED_PRECISIONS_DEVICES_MAP,
    doesSupportStreamGeneration: false,
    externalInterrupt: CONFIG.EXTERNAL_INTERRUPT,
  };
}

function mergeConfigs(defaults, userConfig) {
  return { ...defaults, ...userConfig };
}

export async function checkIsModelDownloaded(modelKeys, modelId) {
  try {
    const cache = await caches.open("transformers-cache");
    const keys = await cache.keys();

    if (keys.length === 0) return false;

    // Build Set for O(1) lookups
    const keySet = new Set(keys.map((k) => k.url));

    return modelKeys.every((modelKey) =>
      Array.from(keySet).some(
        (url) => url.includes(modelKey) && url.includes(modelId),
      ),
    );
  } catch (error) {
    console.error("Error checking model download:", error);
    return false;
  }
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
    this.model_id = CONFIG.MODEL_ID;
    this.pipe = null;
    this.precision = null;
    this.device = null;
  }

  setDefaults({ precision, device }) {
    this.precision = precision;
    this.device = device;
  }

  async downloadModel(precision) {
    if (!precision) {
      throw new Error("Precision must be set to download the model");
    }

    await pipeline("automatic-speech-recognition", this.model_id, {
      dtype: {
        encoder_model: precision,
        decoder_model_merged: precision,
      },
      progress_callback: (progress) => {
        self.postMessage({
          type: "downloadProgress",
          data: progress,
        });
      },
    });

    self.postMessage({
      type: "download",
      data: { status: "success" },
    });
  }

  async loadModel() {
    if (!this.precision || !this.device) {
      throw new Error(
        "Precision and device must be set via init before loading the model",
      );
    }

    const isDownloaded = await checkIsModelDownloaded(
      CONFIG.SUPPORTED_PRECISIONS_DEVICES_MAP[this.precision].modelKeys,
      this.model_id,
    );

    if (!isDownloaded) {
      throw new Error(
        "Model not downloaded. Call .init() method to download first.",
      );
    }

    this.pipe = await pipeline("automatic-speech-recognition", this.model_id, {
      device: this.device,
      dtype: {
        encoder_model: this.precision,
        decoder_model_merged: this.precision,
      },
      progress_callback: (progress) => {
        self.postMessage({
          type: "downloadProgress",
          data: progress,
        });
      },
    });
  }

  async generate(data) {
    const { userInput, modelConfig, generateConfig } = data;

    if (!this.pipe || !this.precision || !this.device) {
      throw new Error(
        "Model not initialized. You must call init first to set precision and device and load the model first. If issue persists, please reopen the browser or redownload the model.",
      );
    }

    if (modelState.isGenerating) {
      throw new Error(
        "A generation is already in progress. Please wait or interrupt the current generation. If issue persists, please reopen the browser or redownload the model. (This might happen when your device automatically kills the process in the background for memory optimization, so the generation status is not updated, especially if you have seen GPU map error before.)",
      );
    }

    modelState.isGenerating = true;

    try {
      // Validate blob URL
      if (
        !userInput.audio_blob_url ||
        !userInput.audio_blob_url.startsWith("blob:")
      ) {
        throw new Error(
          "Missing or invalid audio_blob_url. Please provide a valid blob URL.",
        );
      }

      // Fetch audio from blob URL
      const response = await fetch(userInput.audio_blob_url);
      if (!response.ok) {
        throw new Error(
          `Failed to fetch audio data from blob URL: ${response.statusText}`,
        );
      }
      const audioArrayBuffer = await response.arrayBuffer();

      // Process audio with optimized functions
      const finalGenerationConfig = mergeConfigs(
        CONFIG.DEFAULT_GENERATION_CONFIG,
        generateConfig,
      );
      const targetSampleRate =
        finalGenerationConfig.target_sample_rate || 16000;

      let audioData;
      try {
        audioData = await processAudioData(audioArrayBuffer, targetSampleRate);
      } catch (error) {
        throw new Error(`Preprocessing audio data failed: ${error.message}`);
      }

      // Run transcription
      const finalModelConfig = mergeConfigs(
        CONFIG.DEFAULT_MODEL_CONFIG,
        modelConfig,
      );
      const rawOutputs = await this.pipe(audioData, {
        ...finalModelConfig,
        chunk_length_s: finalGenerationConfig.chunk_length_s,
      });

      const finalOutputs = {
        result: rawOutputs.text,
        ...rawOutputs,
      };
      delete finalOutputs.text;

      self.postMessage({
        type: "generated",
        data: {
          status: "success",
          result: finalOutputs,
        },
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
}

// ==========================================
// Message Handler
// ==========================================
async function handleMessage(event) {
  const { type, data } = event.data;
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
        try {
          MODEL.setDefaults(data);
          await MODEL.loadModel();
          modelState.isInitialized = true;
          self.postMessage({
            type: "init",
            data: {
              status: "success",
              precision: MODEL.precision,
              device: MODEL.device,
            },
          });
        } finally {
          modelState.isInitializing = false;
        }
        break;

      case "checkModelSupports":
        self.postMessage({
          type: "checkModelSupports",
          data: checkModelSupports(),
        });
        break;

      case "download":
        await MODEL.downloadModel(data.precision);
        break;

      case "generate":
        if (!modelState.isInitialized) {
          throw new Error(
            "Model not initialized. You must call init first to set precision and device.",
          );
        }
        await MODEL.generate(data);
        break;

      case "generateStream":
        if (!modelState.isInitialized) {
          throw new Error(
            "Model not initialized. You must call init first to set precision and device.",
          );
        }
        MODEL.generateStream();
        break;

      case "interrupt":
        // External interrupt - placeholder for future implementation
        break;

      case "clearMemory":
        // Dispose pipeline if available
        if (MODEL.pipe && typeof MODEL.pipe.dispose === "function") {
          await MODEL.pipe.dispose();
        }
        MODEL.pipe = null;

        // Force garbage collection if available (development only)
        if (typeof gc !== "undefined") {
          gc();
        }

        modelState.isInitialized = false;
        modelState.isGenerating = false;

        self.postMessage({
          type: "clearMemory",
          data: { status: "success" },
        });
        break;

      default:
        throw new Error(`Unknown message type: ${type}`);
    }
  } catch (error) {
    self.postMessage({
      type: "error",
      data: {
        message: error.message || "Worker operation failed",
        context: type,
      },
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

  initializeEnvironment();
  MODEL = new WebAIModel();

  self.addEventListener("message", handleMessage);

  self.postMessage({
    type: "worker initialized",
    data: {
      success: true,
      message: "Web worker initialized successfully",
    },
  });
}

initializeApp();
