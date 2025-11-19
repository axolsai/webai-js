import {
  env,
  pipeline,
  RawImage,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@latest";

// ==========================================
// Configuration
// ==========================================
const CONFIG = {
  MODEL_ID: "detr-resnet-50",
  EXTERNAL_INTERRUPT: true,
  SUPPORTED_MODES: ["webai"],
  SUPPORTED_PRECISIONS_DEVICES_MAP: {
    q8: {
      size: 43102531,
      modelKeys: ["model_quantized.onnx"],
      supportedDevices: ["webgpu", "wasm"],
      speed: {
        webgpu: 3,
        wasm: 4,
      },
    },
    fp16: {
      size: 83812437,
      modelKeys: ["model_fp16.onnx"],
      supportedDevices: ["wasm", "webgpu"],
      speed: {
        webgpu: 10,
        wasm: 4,
      },
    },
    fp32: {
      size: 166789212,
      modelKeys: ["model.onnx"],
      supportedDevices: ["webgpu", "wasm"],
      speed: {
        webgpu: 10,
        wasm: 4,
      },
    },
  },
  DEFAULT_MODEL_CONFIG: {
    threshold: 0.9,
    score_threshold: 0.5,
    iou_threshold: 0.5,
    max_detections: 100,
    target_classes: null,
    exclude_classes: [],
  },
  DEFAULT_GENERATION_CONFIG: {
    return_labeled_image: false,
    draw_config: {
      textColor: "white",
      lineWidthScale: 0.002,
      fontSizeScale: 0.025,
      fontFamily: "Arial",
      showConfidence: true,
      confidenceDecimals: 1,
    },
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
// Image Drawing Utilities
// ==========================================

/**
 * Generate a consistent color for a given label using a simple hash
 */
function getLabelColor(label) {
  const colors = [
    "#E74C3C",
    "#16A085",
    "#2980B9",
    "#27AE60",
    "#F39C12",
    "#8E44AD",
    "#2ECC71",
    "#E67E22",
    "#9B59B6",
    "#3498DB",
    "#D35400",
    "#1ABC9C",
    "#C0392B",
    "#2C3E50",
    "#7F8C8D",
    "#17A2B8",
    "#28A745",
    "#DC3545",
    "#6F42C1",
    "#20C997",
  ];

  let hash = 0;
  for (let i = 0; i < label.length; i++) {
    const char = label.charCodeAt(i);
    hash = (hash << 5) - hash + char;
    hash = hash & hash;
  }

  const colorIndex = Math.abs(hash) % colors.length;
  return colors[colorIndex];
}

/**
 * Draw bounding boxes and labels on an image
 */
async function drawLabeledImage(imageBlob, detections, options = {}) {
  const {
    textColor = "white",
    lineWidthScale = 0.002,
    fontSizeScale = 0.025,
    fontFamily = "Arial",
    showConfidence = true,
    confidenceDecimals = 1,
  } = options;

  // Create image bitmap from blob (works in Web Workers)
  const imageBitmap = await createImageBitmap(imageBlob);

  // Create canvas with image dimensions
  const canvas = new OffscreenCanvas(imageBitmap.width, imageBitmap.height);
  const ctx = canvas.getContext("2d");

  // Draw the original image
  ctx.drawImage(imageBitmap, 0, 0);

  // Calculate scaled dimensions based on image size
  const scaledLineWidth = Math.max(
    1,
    Math.round(imageBitmap.width * lineWidthScale),
  );
  const scaledFontSize = Math.max(
    12,
    Math.round(imageBitmap.height * fontSizeScale),
  );

  // Set drawing styles
  ctx.lineWidth = scaledLineWidth;
  ctx.font = `${scaledFontSize}px ${fontFamily}`;

  console.log("Image dimensions:", imageBitmap.width, "x", imageBitmap.height);
  console.log(
    "Scaled line width:",
    scaledLineWidth,
    "Scaled font size:",
    scaledFontSize,
  );
  console.log("Drawing", detections.length, "detections");

  // Draw each detection
  detections.forEach((detection, index) => {
    const { box, label, score } = detection;

    console.log(`Detection ${index}:`, { box, label, score });

    // Check if coordinates are normalized (0-1) or pixel coordinates
    const isNormalized = box.xmax <= 1 && box.ymax <= 1;

    let x, y, width, height;

    if (isNormalized) {
      // Convert from normalized coordinates (0-1) to pixel coordinates
      x = box.xmin * imageBitmap.width;
      y = box.ymin * imageBitmap.height;
      width = (box.xmax - box.xmin) * imageBitmap.width;
      height = (box.ymax - box.ymin) * imageBitmap.height;
    } else {
      // Coordinates are already in pixels - use directly
      x = box.xmin;
      y = box.ymin;
      width = box.xmax - box.xmin;
      height = box.ymax - box.ymin;
    }

    // Clamp coordinates to image bounds
    x = Math.max(0, Math.min(x, imageBitmap.width));
    y = Math.max(0, Math.min(y, imageBitmap.height));
    width = Math.max(0, Math.min(width, imageBitmap.width - x));
    height = Math.max(0, Math.min(height, imageBitmap.height - y));

    // Get label name and corresponding color
    const labelName = detection.label || `Class ${label}`;
    const labelColor = getLabelColor(labelName);
    const backgroundColor = labelColor + "CC"; // Add transparency (80% opacity)

    // Create text
    let text = labelName;
    if (showConfidence && score !== undefined) {
      const confidence = (score * 100).toFixed(confidenceDecimals);
      text = `${labelName} (${confidence}%)`;
    }

    // Set colors for this detection
    ctx.strokeStyle = labelColor;

    // Draw bounding box
    ctx.strokeRect(x, y, width, height);

    // Measure text for background
    const textMetrics = ctx.measureText(text);
    const textWidth = textMetrics.width;
    const textHeight = scaledFontSize;
    const padding = Math.max(2, Math.round(scaledFontSize * 0.25));

    // Calculate label position (above the box, or inside if too close to top)
    let labelX = x;
    let labelBgY = y - textHeight - padding * 2;
    let textY = y - padding;

    // If label would be cut off at top, put it inside the box
    if (labelBgY < 0) {
      labelBgY = y;
      textY = y + textHeight + padding;
    }

    // Draw label background
    ctx.fillStyle = backgroundColor;
    ctx.fillRect(
      labelX,
      labelBgY,
      textWidth + padding * 2,
      textHeight + padding * 2,
    );

    // Draw label text
    ctx.fillStyle = textColor;
    ctx.fillText(text, labelX + padding, textY);
  });

  // Convert canvas to blob and return URL
  const blob = await canvas.convertToBlob({ type: "image/png" });
  return URL.createObjectURL(blob);
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

    await pipeline("object-detection", this.model_id, {
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

    this.pipe = await pipeline("object-detection", this.model_id, {
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
      // Validate blob URL
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

      // Run object detection
      const finalModelConfig = mergeConfigs(
        CONFIG.DEFAULT_MODEL_CONFIG,
        modelConfig,
      );
      const output = await this.pipe(image, {
        ...finalModelConfig,
      });

      // Generate labeled image if requested
      const finalGenerateConfig = mergeConfigs(
        CONFIG.DEFAULT_GENERATION_CONFIG,
        generateConfig,
      );

      let labeledImageUrl = null;
      if (finalGenerateConfig.return_labeled_image) {
        try {
          labeledImageUrl = await drawLabeledImage(
            imageBlob,
            output,
            finalGenerateConfig.draw_config,
          );
        } catch (error) {
          console.warn("Failed to generate labeled image:", error);
          labeledImageUrl = "error: " + error.message;
        }
      }

      // Return results
      self.postMessage({
        type: "generated",
        data: {
          status: "success",
          result: {
            result: output,
            result_labeled_image: labeledImageUrl,
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
