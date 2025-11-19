import {
  AutoModelForSequenceClassification,
  AutoTokenizer,
  env,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@latest";

// ==========================================
// Configuration
// ==========================================
const CONFIG = {
  MODEL_ID: "bge-reranker-base",
  EXTERNAL_INTERRUPT: true,
  SUPPORTED_MODES: ["webai"],
  SUPPORTED_PRECISIONS_DEVICES_MAP: {
    q8: {
      size: 279301077,
      modelKeys: ["model_quantized.onnx"],
      supportedDevices: ["webgpu", "wasm"],
      speed: {
        webgpu: 3,
        wasm: 3,
      },
    },
    fp16: {
      size: 556462150,
      modelKeys: ["model_fp16.onnx"],
      supportedDevices: ["wasm", "webgpu"],
      speed: {
        webgpu: 10,
        wasm: 3,
      },
    },
    fp32: {
      size: 1112459588,
      modelKeys: ["model.onnx"],
      supportedDevices: ["webgpu", "wasm"],
      speed: {
        webgpu: 10,
        wasm: 3,
      },
    },
  },
  DEFAULT_MODEL_CONFIG: {
    function_to_apply: "sigmoid", // Function to apply to logits (sigmoid or none)
    top_k: null, // Limit to top K results, null means return all
    padding: true, // Add padding to match sequence lengths
    truncation: true, // Truncate sequences that exceed max_length
    max_length: 512, // Maximum sequence length
    add_special_tokens: true, // Add [CLS], [SEP] tokens automatically
    use_cache: true, // Use KV cache for faster inference
  },
  DEFAULT_GENERATION_CONFIG: {
    normalize_scores: true, // Normalize scores between 0-1
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

function mergeConfigs(defaults, userConfig) {
  return { ...defaults, ...(userConfig || {}) };
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
    if (!precision) {
      throw new Error("Precision must be set to download the model");
    }

    await AutoTokenizer.from_pretrained(this.model_id);
    await AutoModelForSequenceClassification.from_pretrained(this.model_id, {
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

    this.tokenizer = await AutoTokenizer.from_pretrained(this.model_id);
    this.model = await AutoModelForSequenceClassification.from_pretrained(
      this.model_id,
      {
        dtype: this.precision,
        device: this.device,
      },
    );
  }

  async generate(data) {
    const { userInput, modelConfig, generateConfig } = data;

    if (!this.model || !this.tokenizer || !this.precision || !this.device) {
      throw new Error(
        "Model or tokenizer not initialized. You must call init first to set precision and device and load the model first.",
      );
    }

    if (modelState.isGenerating) {
      throw new Error(
        "A generation is already in progress. Please wait or interrupt the current generation.",
      );
    }

    modelState.isGenerating = true;

    try {
      // Validate input
      if (
        !userInput.query ||
        !Array.isArray(userInput.documents) ||
        userInput.documents.length === 0
      ) {
        throw new Error(
          "Missing or invalid userInput. Required format: {query: string, documents: string[]}",
        );
      }

      const finalGenerateConfig = mergeConfigs(
        CONFIG.DEFAULT_GENERATION_CONFIG,
        generateConfig,
      );
      const finalModelConfig = mergeConfigs(
        CONFIG.DEFAULT_MODEL_CONFIG,
        modelConfig,
      );

      const { query, documents } = userInput;

      // Create arrays for queries and documents
      const queries = new Array(documents.length).fill(query);

      // Extract tokenizer options from modelConfig
      const tokenizerOptions = {
        text_pair: documents, // Pass documents as text_pair
        padding: finalModelConfig.padding,
        truncation: finalModelConfig.truncation,
        max_length: finalModelConfig.max_length,
        add_special_tokens: finalModelConfig.add_special_tokens,
      };

      // Tokenize all pairs at once
      const inputs = this.tokenizer(queries, tokenizerOptions);

      // Run inference with all pairs at once
      const outputs = await this.model(inputs, {
        use_cache: finalModelConfig.use_cache,
      });

      // Extract scores from model output
      const logits = outputs.logits.data;

      // Process scores
      const results = documents.map((document, i) => {
        let score;
        if (finalModelConfig.function_to_apply === "sigmoid") {
          score = 1 / (1 + Math.exp(-logits[i]));
        } else {
          score = logits[i];
        }

        return {
          document,
          score,
          originalIndex: i,
        };
      });

      // Sort results by score (descending)
      const sortedResults = [...results].sort((a, b) => b.score - a.score);

      // Apply top_k if specified
      const topK = finalModelConfig.top_k;
      const finalResults = topK ? sortedResults.slice(0, topK) : sortedResults;

      // Apply normalized scores if requested
      if (finalGenerateConfig.normalize_scores) {
        const maxScore = Math.max(...finalResults.map((r) => r.score));
        const minScore = Math.min(...finalResults.map((r) => r.score));
        const scoreRange = maxScore - minScore;

        if (scoreRange > 0) {
          finalResults.forEach((r) => {
            r.normalized_score = (r.score - minScore) / scoreRange;
          });
        } else {
          finalResults.forEach((r, index) => {
            r.normalized_score = 1.0 - index * 0.01;
          });
        }
      }

      self.postMessage({
        type: "generated",
        data: {
          status: "success",
          result: {
            result: finalResults,
          },
        },
      });
    } finally {
      modelState.isGenerating = false;
    }
  }

  generateStream() {
    throw new Error("Stream generation is not supported for reranker models.");
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
        MODEL.tokenizer = null;
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
