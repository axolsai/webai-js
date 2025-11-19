import {
  env,
  AutoModel,
  AutoProcessor,
  RawImage,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@latest";

// ==========================================
// Configuration
// ==========================================
const CONFIG = {
  MODEL_ID: "rmbg-modnet",
  EXTERNAL_INTERRUPT: true,
  SUPPORTED_MODES: ["webai"],
  SUPPORTED_PRECISIONS_DEVICES_MAP: {
    q8: {
      size: 6632188,
      modelKeys: ["model_quantized.onnx"],
      supportedDevices: ["webgpu", "wasm"],
      speed: {
        webgpu: 9,
        wasm: 9,
      },
    },
    fp16: {
      size: 12984781,
      modelKeys: ["model_fp16.onnx"],
      supportedDevices: ["wasm"],
      speed: {
        wasm: 9,
      },
    },
    fp32: {
      size: 25888640,
      modelKeys: ["model.onnx"],
      supportedDevices: ["webgpu", "wasm"],
      speed: {
        webgpu: 10,
        wasm: 9,
      },
    },
  },
  DEFAULT_MODEL_CONFIG: {
    processingResolution: { width: 1024, height: 1024 },
  },
  DEFAULT_GENERATION_CONFIG: {
    confidence_threshold: 0.5,
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

export async function checkIsModelDownloaded(modelKeys, modelId) {
  try {
    const cache = await caches.open("transformers-cache");
    const keys = await cache.keys();

    if (keys.length === 0) return false;

    // Build Set for O(1) lookups - optimized from bge-small pattern
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

function mergeConfigs(defaults, userConfig) {
  return { ...defaults, ...(userConfig || {}) };
}

// ==========================================
// Model Class
// ==========================================
class WebAIModel {
  constructor() {
    this.model_id = CONFIG.MODEL_ID;
    this.processor = null;
    this.model = null;
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

  async downloadModel(precision) {
    if (!precision) {
      throw new Error("Precision must be set to download the model");
    }

    await AutoModel.from_pretrained(this.model_id, {
      dtype: precision,
      progress_callback: (progress) => {
        self.postMessage({
          type: "downloadProgress",
          data: progress,
        });
      },
    });

    await AutoProcessor.from_pretrained(this.model_id);

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

    this.model = await AutoModel.from_pretrained(this.model_id, {
      dtype: this.precision,
      device: this.device,
      model_type: "custom",
      progress_callback: (progress) => {
        self.postMessage({
          type: "downloadProgress",
          data: progress,
        });
      },
    });

    // Create processor with standard config plus custom processing resolution
    this.processor = await AutoProcessor.from_pretrained(this.model_id, {
      config: {
        do_normalize: true,
        do_pad: false,
        do_rescale: true,
        do_resize: true,
        image_mean: [0.5, 0.5, 0.5],
        feature_extractor_type: "ImageFeatureExtractor",
        image_std: [1, 1, 1],
        resample: 2,
        rescale_factor: 0.00392156862745098,
        size: CONFIG.DEFAULT_MODEL_CONFIG.processingResolution,
      },
    });
  }

  async generate(data) {
    const { userInput, modelConfig, generateConfig } = data;

    if (!this.processor || !this.model || !this.precision || !this.device) {
      throw new Error(
        "Model not initialized. You must call init first to set precision and device and load the model first.",
      );
    }

    if (modelState.isGenerating) {
      throw new Error(
        "A generation is already in progress. Please wait or interrupt the current generation.",
      );
    }

    const finalGenerateConfig = mergeConfigs(
      CONFIG.DEFAULT_GENERATION_CONFIG,
      generateConfig,
    );

    // Always use PNG format for transparency support
    const mimeType = "png";
    const outputMimeType = `image/${mimeType}`;

    modelState.isGenerating = true;

    try {
      // Validate input
      if (
        !userInput.image_blob_url ||
        !userInput.image_blob_url.startsWith("blob:")
      ) {
        throw new Error(
          "Missing or invalid image_blob_url. Please provide a valid blob URL.",
        );
      }

      // Fetch and process the original image
      const response = await fetch(userInput.image_blob_url);
      if (!response.ok) {
        throw new Error(
          `Failed to fetch image blob. Status: ${response.status} ${response.statusText}`,
        );
      }
      const imageBlob = await response.blob();
      const image = await RawImage.fromBlob(imageBlob);

      // Apply merged model config if provided
      const finalModelConfig = mergeConfigs(
        CONFIG.DEFAULT_MODEL_CONFIG,
        modelConfig,
      );

      if (finalModelConfig.processingResolution) {
        const width = this._validateDimension(
          finalModelConfig.processingResolution.width,
        );
        const height = this._validateDimension(
          finalModelConfig.processingResolution.height,
        );
        finalModelConfig.processingResolution = { width, height };
      }

      this.processor.config.size = finalModelConfig.processingResolution;

      // Process the image through the model
      const { pixel_values } = await this.processor(image);
      const { output } = await this.model({ input: pixel_values });

      // Get the raw output tensor from the model
      let outputTensor = output[0];

      if (finalGenerateConfig.confidence_threshold !== 0.5) {
        const threshold = Math.max(
          0,
          Math.min(1, finalGenerateConfig.confidence_threshold),
        );
        outputTensor = outputTensor.map((value) =>
          value >= threshold ? 1.0 : 0.0,
        );
      }

      // Convert model output to mask - multiply by 255 to get values in 0-255 range
      let mask = await RawImage.fromTensor(
        outputTensor.mul(255).to("uint8"),
      ).resize(image.width, image.height);

      image.putAlpha(mask);

      // Create output blobs
      const maskedImageBlob = await image.toBlob(outputMimeType);
      const maskedImageBlobUrl = URL.createObjectURL(maskedImageBlob);

      const maskBlob = await mask.toBlob(outputMimeType);
      const maskBlobUrl = URL.createObjectURL(maskBlob);

      // Return results
      self.postMessage({
        type: "generated",
        data: {
          status: "success",
          result: {
            result: maskedImageBlobUrl,
            result_mask: maskBlobUrl,
            size: maskedImageBlob.size,
            mimeType: outputMimeType,
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
        MODEL.model = null;
        MODEL.processor = null;
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
