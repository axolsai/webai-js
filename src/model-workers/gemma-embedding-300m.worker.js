import {
  AutoModel,
  AutoTokenizer,
  env,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@latest";
// ==========================================
// Configuration
// ==========================================
const CONFIG = {
  MODEL_ID: "gemma-embedding-300m",
  EXTERNAL_INTERRUPT: true,
  SUPPORTED_MODES: ["webai"],
  SUPPORTED_PRECISIONS_DEVICES_MAP: {
    q4: {
      size: 196725760,
      price: 0.06,
      modelKeys: ["model_q4.onnx"],
      supportedDevices: ["webgpu", "wasm"],
      speed: {
        webgpu: 10,
        wasm: 8,
      },
    },
    q4f16: {
      size: 175410176,
      price: 0.06,
      modelKeys: ["model_q4f16.onnx"],
      supportedDevices: ["wasm"],
      speed: {
        wasm: 8,
      },
    },
    q8: {
      size: 308890624,
      price: 0.06,
      modelKeys: ["model_quantized.onnx"],
      supportedDevices: ["webgpu", "wasm"],
      speed: {
        webgpu: 10,
        wasm: 8,
      },
    },

    fp16: {
      size: 617434112,
      price: 0.09,
      modelKeys: ["model_fp16.onnx"],
      supportedDevices: ["wasm"],
      speed: {
        wasm: 8,
      },
    },
    fp32: {
      size: 1234521088,
      price: 0.12,
      modelKeys: ["model.onnx"],
      supportedDevices: ["webgpu", "wasm"],
      speed: {
        webgpu: 10,
        wasm: 9.5,
      },
    },
  },
  DEFAULT_MODEL_CONFIG: { pooling: "mean", normalize: true },
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
// Centralized Error Handling
// ==========================================
async function safeWorkerOperation(operation, context = "general") {
  try {
    return await operation();
  } catch (error) {
    self.postMessage({
      type: "error",
      data: {
        message: error.message || "Worker operation failed",
        context: context,
      },
    });
    throw error;
  }
}

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
  return safeWorkerOperation(async () => {
    const cache = await caches.open("transformers-cache");
    const keys = (await cache.keys()).map((key) => key.url);
    if (keys.length === 0) return false;

    return modelKeys.every((modelKey) =>
      keys.some((key) => key.includes(modelKey) && key.includes(modelId)),
    );
  }, "check_model_downloaded");
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

function mergeModelConfigs(userConfig) {
  return { ...CONFIG.DEFAULT_MODEL_CONFIG, ...userConfig };
}

function mergeGenerationConfigs(userConfig) {
  return { ...CONFIG.DEFAULT_GENERATION_CONFIG, ...userConfig };
}

// ==========================================
// Model Class
// ==========================================
class WebAIModel {
  constructor() {
    this.model_id = CONFIG.MODEL_ID;
    this.model = null;
    this.tokenizer = null;
    this.precision = null;
    this.device = null;
  }

  setDefaults({ precision, device }) {
    this.precision = precision;
    this.device = device;
  }

  async downloadModel(precision) {
    return safeWorkerOperation(async () => {
      if (!precision) {
        throw new Error("Precision must be set to download the model");
      }

      await AutoModel.from_pretrained(this.model_id, {
        dtype: precision,
        progress_callback: (progress) => {
          self.postMessage({
            type: "downloadProgress",
            data: {
              ...progress,
            },
          });
        },
      });
      await AutoTokenizer.from_pretrained(this.model_id);

      self.postMessage({
        type: "download",
        data: {
          status: "success",
        },
      });
    }, "downloadModel");
  }

  async loadModel() {
    return safeWorkerOperation(async () => {
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
        device: this.device,
        dtype: this.precision,
      });
      this.tokenizer = await AutoTokenizer.from_pretrained(this.model_id);
    }, "loadModel");
  }

  async generate(data) {
    return safeWorkerOperation(async () => {
      const { userInput, modelConfig, generateConfig } = data;
      if (!this.model || !this.precision || !this.device) {
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

      const finalGenerateConfig = mergeGenerationConfigs(generateConfig);
      void finalGenerateConfig;
      const finalModelConfig = mergeModelConfigs(modelConfig);

      modelState.isGenerating = true;
      try {
        if (
          !userInput.texts ||
          !Array.isArray(userInput.texts) ||
          userInput.texts.length === 0 ||
          !userInput.texts.every((text) => typeof text === "string")
        ) {
          throw new Error(
            "Missing or invalid texts input. Please provide a valid array of strings.",
          );
        }
        const s_time = Date.now();
        const inputs = await this.tokenizer(userInput.texts, {
          padding: true,
        });
        const { sentence_embedding } = await this.model(inputs, {
          ...finalModelConfig,
        });

        const e_time = Date.now();

        const embeddingsArray = sentence_embedding.tolist();

        self.postMessage({
          type: "generated",
          data: {
            status: "success",
            result: {
              result: embeddingsArray,
              runtime: e_time - s_time,
            },
          },
        });
      } finally {
        // Always reset the generating state, even if an error occurred
        modelState.isGenerating = false;
      }
    }, "generate");
  }
  async generateStream() {
    return safeWorkerOperation(async () => {
      throw new Error(
        "Stream generation is not supported for this model, check documentation for how to implement real-time transcription.",
      );
    }, "generateStream");
  }
}

// ==========================================
// Main Application Logic
// ==========================================
// Initialize error handlers
function setupErrorHandling() {
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
    const error = event.reason;
    self.postMessage({
      type: "error",
      data: {
        message: error?.message || "Unhandled Promise Rejection",
        context: "unhandled_promise_rejection",
      },
    });
    event.preventDefault();
  });
}

// Initialize application
function initializeApp() {
  setupErrorHandling();
  initializeEnvironment();

  MODEL = new WebAIModel();

  // Set up message handler with centralized error handling
  self.addEventListener("message", async (event) => {
    await safeWorkerOperation(
      async () => {
        const { type, data } = event.data;
        console.log("Received message from main thread.", type, event.data);

        switch (type) {
          case "init":
            // Prevent multiple simultaneous initialization
            if (modelState.isInitializing) {
              self.postMessage({
                type: "error",
                data: {
                  message: "Model is already initializing. Please wait.",
                  context: "init_already_initializing",
                },
              });
              return;
            }

            // Check required params
            if (!data.precision || !data.device) {
              throw new Error(
                "Init message must contain both precision and device parameters",
              );
            }

            modelState.isInitializing = true;
            try {
              // Pass model config as part of init
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
                  modelConfig: MODEL.modelConfig, // Return applied config
                },
              });
            } finally {
              modelState.isInitializing = false;
            }
            break;

          case "checkModelSupports":
            const resData = checkModelSupports();
            self.postMessage({
              type: "checkModelSupports",
              data: resData,
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

            // Pass both userInput and generationConfig
            await MODEL.generate({
              userInput: data.userInput || data, // For backward compatibility
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
            break;

          case "clearMemory":
            MODEL.model = null;
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
      },
      `message_handler_${event.data?.type || "unknown"}`,
    );
  });

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
