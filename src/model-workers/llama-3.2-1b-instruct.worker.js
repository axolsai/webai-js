// Import dependencies
import {
  env,
  AutoTokenizer,
  AutoModelForCausalLM,
  TextStreamer,
  StoppingCriteria,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@latest";

import { createStructuredOutput } from "https://cdn.jsdelivr.net/npm/@hubters-webai/utils@0.7.9/+esm";

// ==========================================
// Configuration
// ==========================================
const CONFIG = {
  MODEL_ID: "llama-3.2-1b-instruct",
  SUPPORTED_MODES: ["webai"],
  SUPPORTED_PRECISIONS_DEVICES_MAP: {
    q4f16: {
      size: 1237750815,
      modelKeys: ["model_q4f16.onnx"],
      supportedDevices: ["webgpu"],
      speed: { webgpu: 7 },
    },
  },
  DEFAULT_MODEL_CONFIG: {
    max_new_tokens: 512,
    temperature: 0.7,
    early_stopping: false,
    stop_strings: undefined,
    do_sample: true,
    num_beams: 1,
    top_k: 50,
    top_p: 1.0,
    repetition_penalty: 1.0,
    no_repeat_ngram_size: 0,
  },
  DEFAULT_GENERATION_CONFIG: {
    skip_special_tokens: true,
    structured_output: false,
    zod_schema_json: null,
    structured_output_config: {
      max_retries: 3,
      retry_delay: 300,
      timeout: 30000,
      strict_validation: true,
      enable_adaptive_prompting: true,
      max_prompt_adaptations: 3,
      adaptation_strategies: [],
      context_window: 5,
      adaptation_threshold: 2,
    },
  },
};

// ==========================================
// Global state
// ==========================================
let MODEL;
let stopping_criteria;
let structuredOutputFunction = null;
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
// Utility Classes
// ==========================================

class CallbackTextStreamer extends TextStreamer {
  constructor(tokenizer, streamCallback) {
    super(tokenizer, {
      skip_prompt: true,
      skip_special_tokens: true,
    });
    this.callback = streamCallback;
  }

  on_finalized_text(text) {
    if (text) {
      this.callback(text);
    }
  }
}

class InterruptableStoppingCriteria extends StoppingCriteria {
  constructor() {
    super();
    this.interrupted = false;
  }

  interrupt() {
    this.interrupted = true;
  }

  reset() {
    this.interrupted = false;
  }

  _call(input_ids) {
    return new Array(input_ids.length).fill(this.interrupted);
  }
}

// ==========================================
// Structured Output Utilities
// ==========================================

function initializeStructuredOutput(zodSchemaJSON, config = {}) {
  try {
    if (!MODEL.tokenizer || !MODEL.model) {
      throw new Error(
        "Model and tokenizer must be loaded before initializing structured output",
      );
    }

    structuredOutputFunction = createStructuredOutput(
      MODEL.model,
      MODEL.tokenizer,
      zodSchemaJSON,
      extractAssistantMessage,
      config,
    );

    return true;
  } catch (error) {
    console.error("Failed to initialize structured output:", error);
    throw error;
  }
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
    doesSupportStreamGeneration: true,
    doesSupportStructuredOutput: true,
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

function extractAssistantMessage(inputString) {
  const pattern = "assistant\n";
  const index = inputString.lastIndexOf(pattern);
  if (index !== -1) {
    return inputString.substring(index + pattern.length);
  }
  return inputString;
}

function decodeOutput(tokenizer, outputs, skipSpecialTokens) {
  const decodedText = tokenizer.decode(outputs, {
    skip_special_tokens: skipSpecialTokens,
  });

  return skipSpecialTokens ? extractAssistantMessage(decodedText) : decodedText;
}

// ==========================================
// Model Class
// ==========================================
class WebAIModel {
  constructor() {
    this.model_id = CONFIG.MODEL_ID;
    this.tokenizer = null;
    this.model = null;
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
    await AutoModelForCausalLM.from_pretrained(this.model_id, {
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
    this.model = await AutoModelForCausalLM.from_pretrained(this.model_id, {
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
    const { userInput, modelConfig, generateConfig } = data;

    if (!this.tokenizer || !this.model || !this.precision || !this.device) {
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
      const finalModelConfig = mergeConfigs(
        CONFIG.DEFAULT_MODEL_CONFIG,
        modelConfig,
      );
      const finalGenerationConfig = mergeConfigs(
        CONFIG.DEFAULT_GENERATION_CONFIG,
        generateConfig,
      );

      // Handle structured output
      if (
        finalGenerationConfig.structured_output &&
        finalGenerationConfig.zod_schema_json
      ) {
        if (
          !finalGenerationConfig.zod_schema_json ||
          typeof finalGenerationConfig.zod_schema_json !== "object"
        ) {
          throw new Error("Invalid JSON schema provided for structured output");
        }

        initializeStructuredOutput(
          finalGenerationConfig.zod_schema_json,
          finalGenerationConfig.structured_output_config || {},
        );

        const result = await structuredOutputFunction(
          userInput.messages,
          finalModelConfig,
        );

        const outputText =
          typeof result.data === "string"
            ? result.data
            : JSON.stringify(result.data, null, 2);

        self.postMessage({
          type: "generated",
          data: {
            status: "success",
            result: {
              result: outputText,
              structured_output: true,
              metadata: result.metadata,
            },
          },
        });
      } else {
        // Regular generation
        const inputs = await this.tokenizer.apply_chat_template(
          userInput.messages,
          {
            add_generation_prompt: true,
            return_dict: true,
          },
        );

        const outputs = await this.model.generate({
          ...inputs,
          ...finalModelConfig,
          stopping_criteria,
        });

        const outputText = decodeOutput(
          this.tokenizer,
          outputs,
          finalGenerationConfig.skip_special_tokens,
        );

        self.postMessage({
          type: "generated",
          data: {
            status: "success",
            result: {
              result: outputText,
              messages: [
                ...userInput.messages,
                { role: "assistant", content: outputText },
              ],
              structured_output: false,
            },
          },
        });
      }
    } finally {
      modelState.isGenerating = false;
    }
  }

  async generateStream(data) {
    const { userInput, modelConfig, generateConfig } = data;

    if (!this.tokenizer || !this.model || !this.precision || !this.device) {
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
      const finalModelConfig = mergeConfigs(
        CONFIG.DEFAULT_MODEL_CONFIG,
        modelConfig,
      );
      const finalGenerationConfig = mergeConfigs(
        CONFIG.DEFAULT_GENERATION_CONFIG,
        generateConfig,
      );

      // Structured output with streaming falls back to non-streaming
      if (
        finalGenerationConfig.structured_output &&
        finalGenerationConfig.zod_schema_json
      ) {
        console.warn(
          "Structured output with streaming is not fully supported. Falling back to non-streaming generation.",
        );
        await this.generate(data);
        return;
      }

      // Apply chat template
      const inputs = await this.tokenizer.apply_chat_template(
        userInput.messages,
        {
          add_generation_prompt: true,
          return_dict: true,
        },
      );

      // Setup streaming callback
      const streamCallback = (output) => {
        self.postMessage({
          type: "stream",
          data: {
            result: output,
            structured_output: false,
          },
        });
      };

      // Create streamer
      const streamer = new CallbackTextStreamer(this.tokenizer, streamCallback);

      // Generate with streaming
      const outputs = await this.model.generate({
        ...inputs,
        ...finalModelConfig,
        streamer,
        stopping_criteria,
      });

      const outputText = decodeOutput(
        this.tokenizer,
        outputs,
        finalGenerationConfig.skip_special_tokens,
      );

      // Send final complete result
      self.postMessage({
        type: "generated",
        data: {
          status: "success",
          result: {
            result: outputText,
            messages: [
              ...userInput.messages,
              { role: "assistant", content: outputText },
            ],
            structured_output: false,
          },
        },
      });
    } finally {
      modelState.isGenerating = false;
    }
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
        stopping_criteria.reset();
        await MODEL.generate(data);
        break;

      case "generateStream":
        if (!modelState.isInitialized) {
          throw new Error(
            "Model not initialized. You must call init first to set precision and device.",
          );
        }
        stopping_criteria.reset();
        await MODEL.generateStream(data);
        break;

      case "interrupt":
        stopping_criteria.interrupt();
        modelState.isGenerating = false;
        self.postMessage({
          type: "interrupted",
          data: { status: "success" },
        });
        break;

      case "clearMemory":
        MODEL.model = null;
        MODEL.tokenizer = null;
        structuredOutputFunction = null;
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
  stopping_criteria = new InterruptableStoppingCriteria();

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
