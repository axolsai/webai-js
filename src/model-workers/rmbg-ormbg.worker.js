import {
  env,
  pipeline,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@latest";

// ==========================================
// Configuration
// ==========================================
const CONFIG = {
  MODEL_ID: "rmbg-ormbg",
  EXTERNAL_INTERRUPT: true,
  SUPPORTED_MODES: ["webai"],
  SUPPORTED_PRECISIONS_DEVICES_MAP: {
    uint8: {
      size: 44315205,
      modelKeys: ["model_uint8.onnx"],
      supportedDevices: ["wasm"],
      speed: {
        wasm: 4,
      },
    },
    q4f16: {
      size: 88117949,
      modelKeys: ["model_q4f16.onnx"],
      supportedDevices: ["wasm"],
      speed: {
        wasm: 4,
      },
    },
    q8: {
      size: 44315205,
      modelKeys: ["model_quantized.onnx"],
      supportedDevices: ["wasm"],
      speed: {
        wasm: 4,
      },
    },
    fp16: {
      size: 88117930,
      modelKeys: ["model_fp16.onnx"],
      supportedDevices: ["wasm"],
      speed: {
        wasm: 4,
      },
    },
    fp32: {
      size: 176116019,
      modelKeys: ["model.onnx"],
      supportedDevices: ["wasm"],
      speed: {
        wasm: 4,
      },
    },
  },
  DEFAULT_MODEL_CONFIG: {},
  DEFAULT_GENERATION_CONFIG: {},
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

function checkModelSupports() {
  return {
    supportedModes: CONFIG.SUPPORTED_MODES,
    supportedPrecisions: Object.keys(CONFIG.SUPPORTED_PRECISIONS_DEVICES_MAP),
    supportedPrecisionsDevicesMap: CONFIG.SUPPORTED_PRECISIONS_DEVICES_MAP,
    doesSupportStreamGeneration: false,
    externalInterrupt: CONFIG.EXTERNAL_INTERRUPT,
  };
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

  // Helper to validate dimension (ensure divisible by 32 for optimal performance)
  _validateDimension(dim) {
    if (!dim || isNaN(dim) || dim < 32) {
      throw new Error(
        `Invalid dimension: ${dim}. Must be a number greater than or equal to 32.`,
      );
    }
    return Math.round(dim / 32) * 32;
  }

  // Helper method to convert tensor data to blob URL and mask
  async _convertTensorToBlob(tensorOutput) {
    const { data, width, height, channels } = tensorOutput;

    // Create canvases for image and mask
    const canvas = new OffscreenCanvas(width, height);
    const ctx = canvas.getContext("2d");
    const maskCanvas = new OffscreenCanvas(width, height);
    const maskCtx = maskCanvas.getContext("2d");

    // Create ImageData objects
    const imageData = ctx.createImageData(width, height);
    const maskImageData = maskCtx.createImageData(width, height);
    const pixelData = imageData.data;
    const maskPixelData = maskImageData.data;

    // Convert tensor data to RGBA format
    const pixelCount = width * height;
    for (let i = 0; i < pixelCount; i++) {
      const tensorIndex = i * channels;
      const pixelIndex = i * 4;

      if (channels === 4) {
        // RGBA format
        pixelData[pixelIndex] = data[tensorIndex]; // R
        pixelData[pixelIndex + 1] = data[tensorIndex + 1]; // G
        pixelData[pixelIndex + 2] = data[tensorIndex + 2]; // B
        pixelData[pixelIndex + 3] = data[tensorIndex + 3]; // A

        // Create mask from alpha channel
        const alpha = data[tensorIndex + 3];
        maskPixelData[pixelIndex] = alpha;
        maskPixelData[pixelIndex + 1] = alpha;
        maskPixelData[pixelIndex + 2] = alpha;
        maskPixelData[pixelIndex + 3] = 255;
      } else if (channels === 3) {
        // RGB format - add full opacity
        pixelData[pixelIndex] = data[tensorIndex];
        pixelData[pixelIndex + 1] = data[tensorIndex + 1];
        pixelData[pixelIndex + 2] = data[tensorIndex + 2];
        pixelData[pixelIndex + 3] = 255;

        // Create white mask for RGB
        maskPixelData[pixelIndex] = 255;
        maskPixelData[pixelIndex + 1] = 255;
        maskPixelData[pixelIndex + 2] = 255;
        maskPixelData[pixelIndex + 3] = 255;
      } else if (channels === 1) {
        // Grayscale - treat as alpha mask
        const value = data[tensorIndex];
        pixelData[pixelIndex] = 0;
        pixelData[pixelIndex + 1] = 0;
        pixelData[pixelIndex + 2] = 0;
        pixelData[pixelIndex + 3] = value;

        // Create grayscale mask
        maskPixelData[pixelIndex] = value;
        maskPixelData[pixelIndex + 1] = value;
        maskPixelData[pixelIndex + 2] = value;
        maskPixelData[pixelIndex + 3] = 255;
      }
    }

    // Put image data on canvases
    ctx.putImageData(imageData, 0, 0);
    maskCtx.putImageData(maskImageData, 0, 0);

    const outputMimeType = "image/png";

    // Convert canvases to blobs
    const [maskedImageBlob, maskBlob] = await Promise.all([
      canvas.convertToBlob({
        type: outputMimeType,
        quality: 1.0,
      }),
      maskCanvas.convertToBlob({
        type: outputMimeType,
        quality: 1.0,
      }),
    ]);

    return {
      resultBlobUrl: URL.createObjectURL(maskedImageBlob),
      maskBlobUrl: URL.createObjectURL(maskBlob),
      size: maskedImageBlob.size,
      mimeType: outputMimeType,
    };
  }

  async downloadModel(precision) {
    if (!precision) {
      throw new Error("Precision must be set to download the model");
    }

    await pipeline("background-removal", this.model_id, {
      dtype: precision,
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

    this.pipe = await pipeline("background-removal", this.model_id, {
      dtype: this.precision,
      device: this.device,
      progress_callback: (progress) => {
        self.postMessage({
          type: "downloadProgress",
          data: progress,
        });
      },
    });
  }

  async generate(data) {
    const { userInput } = data;

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
      if (
        !userInput.image_blob_url ||
        !userInput.image_blob_url.startsWith("blob:")
      ) {
        throw new Error(
          "Missing or invalid image_blob_url. Please provide a valid blob URL.",
        );
      }

      const output = await this.pipe(userInput.image_blob_url);

      // Convert tensor data to blob URL and get metadata
      const { resultBlobUrl, maskBlobUrl, size, mimeType } =
        await this._convertTensorToBlob(output[0]);

      self.postMessage({
        type: "generated",
        data: {
          status: "success",
          result: {
            result: resultBlobUrl,
            result_mask: maskBlobUrl,
            size: size,
            mimeType: mimeType,
          },
        },
      });
    } finally {
      modelState.isGenerating = false;
    }
  }

  generateStream() {
    throw new Error("Stream generation is not supported for this model type");
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
          MODEL.setDefaults({
            precision: data.precision,
            device: data.device,
          });
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

        await MODEL.generate({
          userInput: data.userInput || data,
          generateConfig: data.generateConfig || {},
          modelConfig: data.modelConfig || {},
        });
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
        MODEL.pipe = null;
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
