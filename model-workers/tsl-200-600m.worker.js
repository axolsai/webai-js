// Import dependencies
import {
  pipeline,
  env,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@latest";

// ==========================================
// Configuration
// ==========================================
const CONFIG = {
  MODEL_ID: "tsl-200-600m",
  EXTERNAL_INTERRUPT: true,
  SUPPORTED_MODES: ["webai"],
  SUPPORTED_PRECISIONS_DEVICES_MAP: {
    q8: {
      size: 419120483,
      modelKeys: ["model_quantized.onnx"],
      supportedDevices: ["wasm"],
      speed: { wasm: 6 },
    },
  },
  DEFAULT_MODEL_CONFIG: {
    src_lang: null,
    tgt_lang: null,
  },
  DEFAULT_GENERATION_CONFIG: {
    skip_special_tokens: true,
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

// Deep merge function for nested configurations
function deepMerge(target, source) {
  const result = { ...target };

  for (const key in source) {
    if (source.hasOwnProperty(key)) {
      if (
        source[key] !== null &&
        typeof source[key] === "object" &&
        !Array.isArray(source[key])
      ) {
        if (
          result[key] &&
          typeof result[key] === "object" &&
          !Array.isArray(result[key])
        ) {
          result[key] = deepMerge(result[key], source[key]);
        } else {
          result[key] = { ...source[key] };
        }
      } else {
        result[key] = source[key];
      }
    }
  }

  return result;
}

function mergeConfigs(defaults, userConfig) {
  return deepMerge(defaults, userConfig || {});
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

    await pipeline("translation", this.model_id, {
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
      data: {
        status: "success",
      },
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

    this.pipe = await pipeline("translation", this.model_id, {
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
    const { userInput, modelConfig } = data;

    if (!this.pipe || !this.precision || !this.device) {
      throw new Error(
        "Model not initialized. You must call init first to set precision and device and load the model first.",
      );
    }

    // Check if already generating to prevent concurrent operations
    if (modelState.isGenerating) {
      throw new Error(
        "A generation is already in progress. Please wait or interrupt the current generation.",
      );
    }

    modelState.isGenerating = true;

    try {
      const finalModelConfig = mergeConfigs(
        CONFIG.DEFAULT_MODEL_CONFIG,
        modelConfig,
      );

      const translations = [];
      for (const text of userInput.texts) {
        const result = await this.pipe(text, {
          ...finalModelConfig,
        });
        // NLLB returns array of translations, we take the first one
        translations.push(result[0]["translation_text"]);
      }

      self.postMessage({
        type: "generated",
        data: {
          status: "success",
          result: {
            result: translations,
          },
        },
      });
    } finally {
      modelState.isGenerating = false;
    }
  }

  async generateStream() {
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
        // Prevent multiple simultaneous initialization
        if (modelState.isInitializing) {
          throw new Error("Model is already initializing. Please wait.");
        }

        // Check required params
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
        // Check if model is initialized
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
        // Check if model is initialized
        if (!modelState.isInitialized) {
          throw new Error(
            "Model not initialized. You must call init first to set precision and device.",
          );
        }

        await MODEL.generateStream(data);
        break;

      case "interrupt":
        // External interrupt
        modelState.isGenerating = false;
        self.postMessage({
          type: "interrupted",
          data: { status: "success" },
        });
        break;

      case "clearMemory":
        MODEL.pipe = null;
        modelState.isInitialized = false;
        modelState.isGenerating = false;
        self.postMessage({
          type: "clearMemory",
          data: {
            status: "success",
          },
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

  // Notify that worker is ready
  self.postMessage({
    type: "worker initialized",
    data: {
      success: true,
      message: "Web worker initialized successfully",
    },
  });
}

// Start the application
initializeApp();
