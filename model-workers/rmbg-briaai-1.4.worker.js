import {
  env,
  pipeline,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@latest";

// ==========================================
// Configuration
// ==========================================
const CONFIG = {
  MODEL_ID: "rmbg-briaai-1.4",
  EXTERNAL_INTERRUPT: true,
  SUPPORTED_MODES: ["webai"],
  SUPPORTED_PRECISIONS_DEVICES_MAP: {
    fp16: {
      size: 88217533,
      modelKeys: ["model_fp16.onnx"],
      supportedDevices: ["webgpu", "wasm"],
      speed: {
        webgpu: 10,
        wasm: 5,
      },
    },
    fp32: {
      size: 176153355,
      modelKeys: ["model.onnx"],
      supportedDevices: ["webgpu", "wasm"],
      speed: {
        webgpu: 10,
        wasm: 5,
      },
    },
    q8: {
      size: 44403226,
      modelKeys: ["model_quantized.onnx"],
      supportedDevices: ["webgpu", "wasm"],
      speed: {
        webgpu: 5,
        wasm: 5,
      },
    },
  },
  DEFAULT_MODEL_CONFIG: {
    processingResolution: { width: 1024, height: 1024 },
  },
  DEFAULT_GENERATION_CONFIG: {
    return_mask: true, // Whether to return the mask
    return_image: true, // Whether to return the masked image
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

function mergeConfigs(defaults, userConfig) {
  return { ...defaults, ...(userConfig || {}) };
}

// ==========================================
// Model Class
// ==========================================
class WebAIModel {
  constructor() {
    this.model_id = CONFIG.MODEL_ID;
    this.pipe = null;
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

  // Helper method to convert tensor data to blob URL and mask - optimized version
  async _convertTensorToBlob(tensorOutput, returnImage, returnMask) {
    const { data, width, height, channels } = tensorOutput;
    const result = {};

    // Create canvases only as needed
    let canvas, ctx, maskCanvas, maskCtx;

    if (returnImage) {
      canvas = new OffscreenCanvas(width, height);
      ctx = canvas.getContext("2d");
    }

    if (returnMask) {
      maskCanvas = new OffscreenCanvas(width, height);
      maskCtx = maskCanvas.getContext("2d");
    }

    // Create ImageData objects only as needed
    let imageData, maskImageData;

    if (returnImage) {
      imageData = ctx.createImageData(width, height);
    }

    if (returnMask) {
      maskImageData = maskCtx.createImageData(width, height);
    }

    // Convert tensor data to RGBA format
    for (let i = 0; i < width * height; i++) {
      const tensorIndex = i * channels;
      const pixelIndex = i * 4;

      if (channels === 4) {
        // RGBA format
        if (returnImage) {
          imageData.data[pixelIndex] = data[tensorIndex]; // R
          imageData.data[pixelIndex + 1] = data[tensorIndex + 1]; // G
          imageData.data[pixelIndex + 2] = data[tensorIndex + 2]; // B
          imageData.data[pixelIndex + 3] = data[tensorIndex + 3]; // A
        }

        // Create mask from alpha channel
        if (returnMask) {
          const alpha = data[tensorIndex + 3];
          maskImageData.data[pixelIndex] = alpha; // R
          maskImageData.data[pixelIndex + 1] = alpha; // G
          maskImageData.data[pixelIndex + 2] = alpha; // B
          maskImageData.data[pixelIndex + 3] = 255; // A (full opacity)
        }
      } else if (channels === 3) {
        // RGB format - add full opacity
        if (returnImage) {
          imageData.data[pixelIndex] = data[tensorIndex]; // R
          imageData.data[pixelIndex + 1] = data[tensorIndex + 1]; // G
          imageData.data[pixelIndex + 2] = data[tensorIndex + 2]; // B
          imageData.data[pixelIndex + 3] = 255; // A (full opacity)
        }

        // Create white mask for RGB
        if (returnMask) {
          maskImageData.data[pixelIndex] = 255; // R
          maskImageData.data[pixelIndex + 1] = 255; // G
          maskImageData.data[pixelIndex + 2] = 255; // B
          maskImageData.data[pixelIndex + 3] = 255; // A
        }
      } else if (channels === 1) {
        // Grayscale - treat as alpha mask
        const value = data[tensorIndex];

        if (returnImage) {
          imageData.data[pixelIndex] = 0; // R
          imageData.data[pixelIndex + 1] = 0; // G
          imageData.data[pixelIndex + 2] = 0; // B
          imageData.data[pixelIndex + 3] = value; // A
        }

        // Create grayscale mask
        if (returnMask) {
          maskImageData.data[pixelIndex] = value; // R
          maskImageData.data[pixelIndex + 1] = value; // G
          maskImageData.data[pixelIndex + 2] = value; // B
          maskImageData.data[pixelIndex + 3] = 255; // A
        }
      }
    }

    const outputMimeType = "image/png";

    // Process only what's needed
    const promises = [];

    if (returnImage) {
      ctx.putImageData(imageData, 0, 0);
      promises.push(
        canvas
          .convertToBlob({
            type: outputMimeType,
            quality: 1.0,
          })
          .then((blob) => {
            result.resultBlobUrl = URL.createObjectURL(blob);
            result.size = blob.size;
            result.mimeType = outputMimeType;
          }),
      );
    }

    if (returnMask) {
      maskCtx.putImageData(maskImageData, 0, 0);
      promises.push(
        maskCanvas
          .convertToBlob({
            type: outputMimeType,
            quality: 1.0,
          })
          .then((blob) => {
            result.maskBlobUrl = URL.createObjectURL(blob);
          }),
      );
    }

    await Promise.all(promises);
    return result;
  }

  async downloadModel(precision) {
    return safeWorkerOperation(async () => {
      if (!precision) {
        throw new Error("Precision must be set to download the model");
      }

      await pipeline("background-removal", this.model_id, {
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

      this.pipe = await pipeline("background-removal", this.model_id, {
        dtype: this.precision,
        device: this.device,
        progress_callback: (progress) => {
          self.postMessage({
            type: "downloadProgress",
            data: {
              ...progress,
            },
          });
        },
      });
    }, "loadModel");
  }

  async generate(data) {
    return safeWorkerOperation(async () => {
      const { userInput, modelConfig, generateConfig } = data;

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

      const finalGenerateConfig = mergeConfigs(
        CONFIG.DEFAULT_GENERATION_CONFIG,
        generateConfig,
      );

      modelState.isGenerating = true;
      const startTime = performance.now();

      try {
        if (
          !userInput.image_blob_url ||
          !userInput.image_blob_url.startsWith("blob:")
        ) {
          throw new Error(
            "Missing or invalid image_blob_url. Please provide a valid blob URL.",
          );
        }

        // Apply merged model config if provided
        const finalModelConfig = mergeConfigs(
          CONFIG.DEFAULT_MODEL_CONFIG,
          modelConfig,
        );

        // Validate processing resolution if provided
        if (finalModelConfig.processingResolution) {
          const width = this._validateDimension(
            finalModelConfig.processingResolution.width,
          );
          const height = this._validateDimension(
            finalModelConfig.processingResolution.height,
          );
          finalModelConfig.processingResolution = { width, height };
        }

        // Note: BiRefNet pipeline doesn't have the same processor config setup as MODNet
        // The processing resolution would need to be handled differently depending on the pipeline implementation
        // For now, we'll pass it through the pipeline call if supported
        const pipelineOptions = {};
        if (finalModelConfig.processingResolution) {
          pipelineOptions.processing_resolution =
            finalModelConfig.processingResolution;
        }

        const output = await this.pipe(
          userInput.image_blob_url,
          pipelineOptions,
        );

        const processingTime = performance.now() - startTime;

        // Convert tensor data to blob URL and get metadata - only process what's needed
        const { resultBlobUrl, maskBlobUrl, size, mimeType } =
          await this._convertTensorToBlob(
            output[0],
            finalGenerateConfig.return_image,
            finalGenerateConfig.return_mask,
          );

        // User controls what to return
        const result = {
          processing_time: processingTime,
        };

        // Return masked image if requested
        if (finalGenerateConfig.return_image && resultBlobUrl) {
          result.result = resultBlobUrl;
          result.size = size;
          result.mimeType = mimeType;
        }

        // Return mask if requested
        if (finalGenerateConfig.return_mask && maskBlobUrl) {
          result.result_mask = maskBlobUrl;
        }

        self.postMessage({
          type: "generated",
          data: {
            status: "success",
            result,
          },
        });
      } finally {
        modelState.isGenerating = false;
      }
    }, "generate");
  }

  async generateStream() {
    return safeWorkerOperation(async () => {
      throw new Error("Stream generation is not supported for this model type");
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
