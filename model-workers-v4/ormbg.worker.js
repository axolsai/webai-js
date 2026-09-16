import {
  env,
  pipeline,
  RawImage,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0";

const MODEL_CONFIG_FIELDS = [
  { name: "processingResolution", type: "{ width: integer, height: integer }", default: { width: 512, height: 512 }, description: "Model input dimensions; each value must be from 32 through 2048 and divisible by 32." },
];

const GENERATION_CONFIG_FIELDS = [
  { name: "threshold", type: "number | null", default: null, minimum: 0, maximum: 1, description: "Keep null for a soft alpha matte, or set 0–1 to return a binary foreground mask." },
];

const CONFIG = {
  MODEL_ID: "ormbg",
  MODEL_REPOSITORY: "onnx-community/ormbg-ONNX",
  MODEL_REVISION: "034e2d884afbab897e10e78fc5bb566b29533fd6",
  WORKER_VERSION: "v4",
  TRANSFORMERS_JS_VERSION: "4.2.0",
  EXTERNAL_INTERRUPT: true,
  SUPPORTED_MODES: ["webai"],
  SUPPORTED_PRECISIONS_DEVICES_MAP: {
    fp32: { size: 176116329, weightsSize: 176116019, supportedDevices: ["wasm", "webgpu"], modelKeys: ["model.onnx"], dtype: "fp32" },
    fp16: { size: 88118240, weightsSize: 88117930, supportedDevices: ["wasm", "webgpu"], modelKeys: ["model_fp16.onnx"], dtype: "fp16" },
    int8: { size: 44315446, weightsSize: 44315136, supportedDevices: ["wasm", "webgpu"], modelKeys: ["model_int8.onnx"], dtype: "int8" },
    q8: { size: 44315515, weightsSize: 44315205, supportedDevices: ["wasm", "webgpu"], modelKeys: ["model_quantized.onnx"], dtype: "q8" },
    uint8: { size: 44315515, weightsSize: 44315205, supportedDevices: ["wasm", "webgpu"], modelKeys: ["model_uint8.onnx"], dtype: "uint8" },
    q4: { size: 176116348, weightsSize: 176116038, supportedDevices: ["wasm", "webgpu"], modelKeys: ["model_q4.onnx"], dtype: "q4" },
    q4f16: { size: 88118259, weightsSize: 88117949, supportedDevices: ["wasm", "webgpu"], modelKeys: ["model_q4f16.onnx"], dtype: "q4f16" },
    bnb4: { size: 176116348, weightsSize: 176116038, supportedDevices: ["wasm", "webgpu"], modelKeys: ["model_bnb4.onnx"], dtype: "bnb4" },
  },
  DEFAULT_MODEL_CONFIG: { processingResolution: { width: 512, height: 512 } },
  DEFAULT_GENERATION_CONFIG: { threshold: null },
};

const BENCHMARK_RESULTS = [
  {
    "precision": "fp32",
    "device": "wasm",
    "status": "pass",
    "runtime": {
      "loadMs": 38114.8,
      "inferenceMs": 7085.2,
      "totalMs": 45230.9,
      "grade": "very-slow",
      "megapixelsPerSecond": 0.0987,
      "loadGrade": "slow"
    },
    "quality": {
      "passed": true,
      "grade": "excellent",
      "meanAbsoluteError": 0,
      "intersectionOverUnion": 1,
      "foregroundRatio": 0.256265,
      "dynamicRange": 1
    }
  },
  {
    "precision": "fp32",
    "device": "webgpu",
    "status": "pass",
    "runtime": {
      "loadMs": 24680.2,
      "inferenceMs": 264.1,
      "totalMs": 24975.5,
      "grade": "slow",
      "megapixelsPerSecond": 2.6482,
      "loadGrade": "slow"
    },
    "quality": {
      "passed": true,
      "grade": "excellent",
      "meanAbsoluteError": 0,
      "intersectionOverUnion": 1,
      "foregroundRatio": 0.256265,
      "dynamicRange": 0.996078
    }
  },
  {
    "precision": "fp16",
    "device": "wasm",
    "status": "pass",
    "runtime": {
      "loadMs": 19426.7,
      "inferenceMs": 7291.1,
      "totalMs": 26744.6,
      "grade": "very-slow",
      "megapixelsPerSecond": 0.0959,
      "loadGrade": "slow"
    },
    "quality": {
      "passed": true,
      "grade": "excellent",
      "meanAbsoluteError": 0.000002,
      "intersectionOverUnion": 1,
      "foregroundRatio": 0.256265,
      "dynamicRange": 1
    }
  },
  {
    "precision": "fp16",
    "device": "webgpu",
    "status": "pass",
    "runtime": {
      "loadMs": 18024,
      "inferenceMs": 224.2,
      "totalMs": 18278.2,
      "grade": "good",
      "megapixelsPerSecond": 3.1195,
      "loadGrade": "slow"
    },
    "quality": {
      "passed": true,
      "grade": "excellent",
      "meanAbsoluteError": 0.000887,
      "intersectionOverUnion": 0.999939,
      "foregroundRatio": 0.256267,
      "dynamicRange": 1
    }
  },
  {
    "precision": "int8",
    "device": "wasm",
    "status": "pass",
    "runtime": {
      "loadMs": 21876.1,
      "inferenceMs": 23701.3,
      "totalMs": 45606.6,
      "grade": "very-slow",
      "megapixelsPerSecond": 0.0295,
      "loadGrade": "slow"
    },
    "quality": {
      "passed": true,
      "grade": "excellent",
      "meanAbsoluteError": 0.000274,
      "intersectionOverUnion": 0.999286,
      "foregroundRatio": 0.256191,
      "dynamicRange": 1
    }
  },
  {
    "precision": "int8",
    "device": "webgpu",
    "status": "pass",
    "runtime": {
      "loadMs": 10103.7,
      "inferenceMs": 24517.2,
      "totalMs": 34648.2,
      "grade": "very-slow",
      "megapixelsPerSecond": 0.0285,
      "loadGrade": "slow"
    },
    "quality": {
      "passed": true,
      "grade": "excellent",
      "meanAbsoluteError": 0.00028,
      "intersectionOverUnion": 0.999297,
      "foregroundRatio": 0.256191,
      "dynamicRange": 1
    }
  },
  {
    "precision": "q8",
    "device": "wasm",
    "status": "pass",
    "runtime": {
      "loadMs": 8124.6,
      "inferenceMs": 7564.2,
      "totalMs": 15716.3,
      "grade": "very-slow",
      "megapixelsPerSecond": 0.0925,
      "loadGrade": "slow"
    },
    "quality": {
      "passed": true,
      "grade": "excellent",
      "meanAbsoluteError": 0.000208,
      "intersectionOverUnion": 0.999459,
      "foregroundRatio": 0.256336,
      "dynamicRange": 1
    }
  },
  {
    "precision": "q8",
    "device": "webgpu",
    "status": "pass",
    "runtime": {
      "loadMs": 9839.2,
      "inferenceMs": 7838.9,
      "totalMs": 17704.8,
      "grade": "very-slow",
      "megapixelsPerSecond": 0.0892,
      "loadGrade": "slow"
    },
    "quality": {
      "passed": true,
      "grade": "excellent",
      "meanAbsoluteError": 0.000212,
      "intersectionOverUnion": 0.999442,
      "foregroundRatio": 0.256337,
      "dynamicRange": 1
    }
  },
  {
    "precision": "uint8",
    "device": "wasm",
    "status": "pass",
    "runtime": {
      "loadMs": 8970.6,
      "inferenceMs": 7483.1,
      "totalMs": 16482.2,
      "grade": "very-slow",
      "megapixelsPerSecond": 0.0935,
      "loadGrade": "slow"
    },
    "quality": {
      "passed": true,
      "grade": "excellent",
      "meanAbsoluteError": 0.000208,
      "intersectionOverUnion": 0.999459,
      "foregroundRatio": 0.256336,
      "dynamicRange": 1
    }
  },
  {
    "precision": "uint8",
    "device": "webgpu",
    "status": "pass",
    "runtime": {
      "loadMs": 9247.6,
      "inferenceMs": 7817.3,
      "totalMs": 17091.5,
      "grade": "very-slow",
      "megapixelsPerSecond": 0.0895,
      "loadGrade": "slow"
    },
    "quality": {
      "passed": true,
      "grade": "excellent",
      "meanAbsoluteError": 0.000212,
      "intersectionOverUnion": 0.999442,
      "foregroundRatio": 0.256337,
      "dynamicRange": 1
    }
  },
  {
    "precision": "q4",
    "device": "wasm",
    "status": "pass",
    "runtime": {
      "loadMs": 24528.2,
      "inferenceMs": 7138.1,
      "totalMs": 31696.1,
      "grade": "very-slow",
      "megapixelsPerSecond": 0.098,
      "loadGrade": "slow"
    },
    "quality": {
      "passed": true,
      "grade": "excellent",
      "meanAbsoluteError": 0,
      "intersectionOverUnion": 1,
      "foregroundRatio": 0.256265,
      "dynamicRange": 1
    }
  },
  {
    "precision": "q4",
    "device": "webgpu",
    "status": "pass",
    "runtime": {
      "loadMs": 23564.2,
      "inferenceMs": 255.4,
      "totalMs": 23852.2,
      "grade": "slow",
      "megapixelsPerSecond": 2.7384,
      "loadGrade": "slow"
    },
    "quality": {
      "passed": true,
      "grade": "excellent",
      "meanAbsoluteError": 0,
      "intersectionOverUnion": 1,
      "foregroundRatio": 0.256265,
      "dynamicRange": 0.996078
    }
  },
  {
    "precision": "q4f16",
    "device": "wasm",
    "status": "pass",
    "runtime": {
      "loadMs": 15641.7,
      "inferenceMs": 7235,
      "totalMs": 22907.6,
      "grade": "very-slow",
      "megapixelsPerSecond": 0.0967,
      "loadGrade": "slow"
    },
    "quality": {
      "passed": true,
      "grade": "excellent",
      "meanAbsoluteError": 0.000002,
      "intersectionOverUnion": 1,
      "foregroundRatio": 0.256265,
      "dynamicRange": 1
    }
  },
  {
    "precision": "q4f16",
    "device": "webgpu",
    "status": "pass",
    "runtime": {
      "loadMs": 22414.4,
      "inferenceMs": 229.4,
      "totalMs": 22673.8,
      "grade": "good",
      "megapixelsPerSecond": 3.0488,
      "loadGrade": "slow"
    },
    "quality": {
      "passed": true,
      "grade": "excellent",
      "meanAbsoluteError": 0.000887,
      "intersectionOverUnion": 0.999939,
      "foregroundRatio": 0.256267,
      "dynamicRange": 1
    }
  },
  {
    "precision": "bnb4",
    "device": "wasm",
    "status": "pass",
    "runtime": {
      "loadMs": 24034.9,
      "inferenceMs": 7169.3,
      "totalMs": 31233.2,
      "grade": "very-slow",
      "megapixelsPerSecond": 0.0976,
      "loadGrade": "slow"
    },
    "quality": {
      "passed": true,
      "grade": "excellent",
      "meanAbsoluteError": 0,
      "intersectionOverUnion": 1,
      "foregroundRatio": 0.256265,
      "dynamicRange": 1
    }
  },
  {
    "precision": "bnb4",
    "device": "webgpu",
    "status": "pass",
    "runtime": {
      "loadMs": 24596.2,
      "inferenceMs": 257.3,
      "totalMs": 24883.6,
      "grade": "slow",
      "megapixelsPerSecond": 2.7182,
      "loadGrade": "slow"
    },
    "quality": {
      "passed": true,
      "grade": "excellent",
      "meanAbsoluteError": 0,
      "intersectionOverUnion": 1,
      "foregroundRatio": 0.256265,
      "dynamicRange": 0.996078
    }
  }
];

const MANIFEST = {
  contractVersion: "1.0",
  model: {
    id: CONFIG.MODEL_ID,
    displayName: "ORMBG — Human Background Removal",
    provider: "Stefan Schirrmacher",
    providerUrl: "https://github.com/schirrmacher",
    license: "Apache-2.0",
    licenseUrl: "https://www.apache.org/licenses/LICENSE-2.0.txt",
    lastUpdated: "2024-09-29T14:37:18.000Z",
    sourceRepository: "schirrmacher/ormbg",
    repository: CONFIG.MODEL_REPOSITORY,
    artifactRevision: CONFIG.MODEL_REVISION,
    artifactLastUpdated: "2026-04-14T13:06:54.000Z",
    task: "background-removal",
    description: "An open human-focused foreground segmentation model converted for Transformers.js that runs locally in the browser and returns a per-pixel foreground matte.",
    intendedUses: [
      "Removing or replacing the background of a single-person portrait.",
      "Creating a soft alpha matte for compositing portraits in browser applications.",
      "Prototyping client-side portrait segmentation without a hosted inference API.",
    ],
    limitations: [
      "ORMBG is specialized for images containing humans and is not a general object-segmentation model.",
      "The upstream author reports that the model can struggle on real images because synthetic training data does not fully represent real lighting, occlusion, scale, and backgrounds.",
      "Hair, transparent materials, motion blur, occlusion, multiple people, unusual poses, and low foreground/background contrast can produce incomplete or haloed masks.",
      "The output is an estimated matte rather than verified ground truth and must not be used for identity, biometric, medical, or safety-critical decisions.",
      "Higher processing resolutions require more memory and computation and can fail on constrained devices.",
    ],
  },
  runtime: { precisions: CONFIG.SUPPORTED_PRECISIONS_DEVICES_MAP },
  input: {
    description: "Provide exactly one browser blob URL containing a decodable image.",
    alternatives: [{
      name: "image blob",
      description: "The worker fetches and decodes the image while preserving its original dimensions for the returned PNGs.",
      fields: [{ name: "image_blob_url", type: "string(blob URL)", required: true, description: "A blob: URL accessible from this worker." }],
    }],
  },
  config: { model: MODEL_CONFIG_FIELDS, generation: GENERATION_CONFIG_FIELDS },
  output: {
    description: "WebAI wraps the native result below as { result, runtime: { durationMs } }. Blob URLs remain valid until the next generation or clearMemory call.",
    fields: [
      { name: "result", type: "string(blob URL)", required: true, description: "Transparent PNG containing the original RGB image with the predicted matte as alpha." },
      { name: "result_mask", type: "string(blob URL)", required: true, description: "Grayscale PNG alpha mask; black is background and white is foreground." },
      { name: "mimeType", type: "'image/png'", required: true, description: "MIME type of both output blobs." },
      { name: "size", type: "integer", required: true, description: "Transparent PNG size in bytes." },
      { name: "maskSize", type: "integer", required: true, description: "Mask PNG size in bytes." },
      { name: "width", type: "integer", required: true, description: "Original image width in pixels." },
      { name: "height", type: "integer", required: true, description: "Original image height in pixels." },
    ],
    example: { result: "blob:https://app.example/…", result_mask: "blob:https://app.example/…", mimeType: "image/png", size: 120000, maskSize: 24000, width: 1024, height: 683 },
  },
  operations: ["checkModelSupports", "init", "download", "generate", "clearMemory"],
  benchmark: {
    report: "benchmarks/ormbg.json",
    testedAt: "2026-09-08T17:07:37.483Z",
    fixture: {
      file: "modnet-portrait-pexels-5965592.jpg",
      url: "https://f005.backblazeb2.com/file/hubters-webai-public/assets/benchmarks/modnet-portrait-pexels-5965592.jpg",
      source: "https://www.pexels.com/photo/woman-wearing-pink-turtleneck-sweater-5965592/",
      license: "Pexels License",
      size: 78124,
      sha1: "38212739a4a4f0be25d84863cf991d7fb6c6d818",
      width: 1024,
      height: 683,
    },
    reference: { file: "ormbg-portrait-pexels-5965592-mask.png", size: 59133, sha1: "83c88c66fbf4b8c32fdc0a9e17631aa30a08c6d5", width: 1024, height: 683, generator: { precision: "fp32", device: "wasm", processingResolution: { width: 512, height: 512 } } },
    note: "Measured values are machine/browser-specific; regenerate the report on target hardware.",
    environment: "Headless Chrome 151 on macOS; Transformers.js 4.2.0",
    invalidatedBy: [
      "model artifact revision change",
      "Transformers.js or ONNX Runtime version change",
      "benchmark fixture, reference matte, or quality-gate change",
      "target browser/backend behavior change",
    ],
    results: BENCHMARK_RESULTS,
  },
};

let MODEL;
const modelState = { isInitializing: false, isInitialized: false, isGenerating: false };

function initializeEnvironment(workerConfig = {}) {
  env.allowRemoteModels = workerConfig.allowRemoteModels ?? true;
  env.allowLocalModels = workerConfig.allowLocalModels ?? false;
  env.useBrowserCache = workerConfig.useBrowserCache ?? true;
  if (workerConfig.remoteHost) env.remoteHost = workerConfig.remoteHost;
  if (workerConfig.remotePathTemplate) env.remotePathTemplate = workerConfig.remotePathTemplate;
  if (workerConfig.localModelPath) env.localModelPath = workerConfig.localModelPath;
  if (workerConfig.modelId) {
    MODEL.model_id = workerConfig.modelId;
    MODEL.revision = workerConfig.modelRevision;
  } else if (workerConfig.modelRevision) {
    MODEL.revision = workerConfig.modelRevision;
  }
}

function checkModelSupports() {
  return {
    manifest: { ...MANIFEST, model: { ...MANIFEST.model, repository: MODEL.model_id, artifactRevision: MODEL.revision ?? "unversioned" } },
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

function postResponse(requestId, type, data) { self.postMessage({ requestId, type, data }); }
function postProgress(requestId, value) {
  postResponse(requestId, "downloadProgress", { ...value, progress: value.status === "ready" ? 100 : value.progress });
}

function validatePrecisionDevice(precision, device, precisionMap = CONFIG.SUPPORTED_PRECISIONS_DEVICES_MAP) {
  const entry = precisionMap[precision];
  if (!entry) throw new Error(`Unsupported precision: ${precision}`);
  if (!entry.supportedDevices.includes(device)) throw new Error(`Device ${device} is not supported for precision ${precision}`);
  return entry;
}

function validateConfigs(modelConfig, generateConfig) {
  const resolution = modelConfig.processingResolution;
  if (!resolution || !Number.isInteger(resolution.width) || !Number.isInteger(resolution.height)) {
    throw new Error("modelConfig.processingResolution must contain integer width and height values.");
  }
  for (const [name, value] of Object.entries(resolution)) {
    if (value < 32 || value > 2048 || value % 32 !== 0) throw new Error(`processingResolution.${name} must be from 32 through 2048 and divisible by 32.`);
  }
  if (generateConfig.threshold !== null && (!Number.isFinite(generateConfig.threshold) || generateConfig.threshold < 0 || generateConfig.threshold > 1)) {
    throw new Error("generateConfig.threshold must be null or a number from 0 through 1.");
  }
}

class WebAIModel {
  constructor() {
    this.model_id = CONFIG.MODEL_REPOSITORY;
    this.revision = CONFIG.MODEL_REVISION;
    this.pipe = null;
    this.precision = null;
    this.device = null;
    this.outputUrls = [];
  }

  setDefaults({ precision, device }) {
    validatePrecisionDevice(precision, device);
    this.precision = precision;
    this.device = device;
  }

  async loadModel(requestId) {
    const entry = validatePrecisionDevice(this.precision, this.device);
    await this.disposeModel();
    try {
      this.pipe = await pipeline("background-removal", this.model_id, {
        dtype: entry.dtype,
        device: this.device,
        ...(this.revision ? { revision: this.revision } : {}),
        progress_callback: (value) => postProgress(requestId, value),
      });
    } catch (error) {
      await this.disposeModel();
      throw error;
    }
  }

  async downloadModel(precision, requestId) {
    const entry = CONFIG.SUPPORTED_PRECISIONS_DEVICES_MAP[precision];
    if (!entry) throw new Error(`Unsupported precision: ${precision}`);
    const downloadedModel = await pipeline("background-removal", this.model_id, {
      dtype: entry.dtype,
      device: entry.supportedDevices.includes("wasm") ? "wasm" : "webgpu",
      ...(this.revision ? { revision: this.revision } : {}),
      progress_callback: (value) => postProgress(requestId, value),
    });
    try {
      // Pipeline initialization downloads both weights and preprocessing assets.
    } finally {
      await downloadedModel.dispose?.();
    }
  }

  revokeOutputs() {
    for (const url of this.outputUrls) URL.revokeObjectURL(url);
    this.outputUrls = [];
  }

  async generate(data, requestId) {
    if (!this.pipe) throw new Error("Call init before generate.");
    const { userInput, modelConfig = {}, generateConfig = {} } = data ?? {};
    if (!userInput || Object.keys(userInput).filter((key) => userInput[key] != null).length !== 1 || typeof userInput.image_blob_url !== "string" || !userInput.image_blob_url.startsWith("blob:")) {
      throw new Error("userInput must contain exactly one valid image_blob_url.");
    }
    const finalModelConfig = { ...CONFIG.DEFAULT_MODEL_CONFIG, ...modelConfig };
    const finalGenerateConfig = { ...CONFIG.DEFAULT_GENERATION_CONFIG, ...generateConfig };
    validateConfigs(finalModelConfig, finalGenerateConfig);

    const response = await fetch(userInput.image_blob_url);
    if (!response.ok) throw new Error(`Failed to fetch image blob: ${response.status} ${response.statusText}`);
    const image = await RawImage.fromBlob(await response.blob());
    this.pipe.processor.config.size = finalModelConfig.processingResolution;
    let cutout = await this.pipe(image);
    if (Array.isArray(cutout)) cutout = cutout[0];
    if (!(cutout instanceof RawImage) || cutout.channels !== 4) throw new Error("ORMBG returned an invalid RGBA cutout.");
    cutout = await cutout.resize(image.width, image.height);
    const bitmap = await createImageBitmap(await cutout.toBlob("image/png"));
    const transparentCanvas = new OffscreenCanvas(image.width, image.height);
    const transparentContext = transparentCanvas.getContext("2d");
    transparentContext.drawImage(bitmap, 0, 0);
    bitmap.close();
    const cutoutPixels = transparentContext.getImageData(0, 0, image.width, image.height);
    const maskCanvas = new OffscreenCanvas(image.width, image.height);
    const maskContext = maskCanvas.getContext("2d");
    const maskPixels = maskContext.createImageData(image.width, image.height);
    for (let index = 0; index < cutoutPixels.data.length; index += 4) {
      let alpha = cutoutPixels.data[index + 3];
      if (finalGenerateConfig.threshold !== null) alpha = alpha >= finalGenerateConfig.threshold * 255 ? 255 : 0;
      cutoutPixels.data[index + 3] = alpha;
      maskPixels.data[index] = alpha;
      maskPixels.data[index + 1] = alpha;
      maskPixels.data[index + 2] = alpha;
      maskPixels.data[index + 3] = 255;
    }
    transparentContext.putImageData(cutoutPixels, 0, 0);
    maskContext.putImageData(maskPixels, 0, 0);
    const [transparentBlob, maskBlob] = await Promise.all([transparentCanvas.convertToBlob({ type: "image/png" }), maskCanvas.convertToBlob({ type: "image/png" })]);
    this.revokeOutputs();
    const result = URL.createObjectURL(transparentBlob);
    const resultMask = URL.createObjectURL(maskBlob);
    this.outputUrls.push(result, resultMask);
    postResponse(requestId, "generated", { status: "success", result: { result, result_mask: resultMask, mimeType: "image/png", size: transparentBlob.size, maskSize: maskBlob.size, width: image.width, height: image.height } });
  }

  async disposeModel() {
    const pipe = this.pipe;
    this.pipe = null;
    if (pipe) await pipe.dispose?.();
  }

  async clearMemory() {
    this.revokeOutputs();
    await this.disposeModel();
  }
}

async function handleMessage({ data: message = {} }) {
  const { requestId, type, data } = message;
  try {
    switch (type) {
      case "checkModelSupports":
        initializeEnvironment(data?.workerConfig);
        postResponse(requestId, type, checkModelSupports());
        break;
      case "init":
        if (modelState.isInitializing || modelState.isGenerating) throw new Error("Worker is busy.");
        modelState.isInitializing = true;
        modelState.isInitialized = false;
        try {
          MODEL.setDefaults(data ?? {});
          await MODEL.loadModel(requestId);
          modelState.isInitialized = true;
          postResponse(requestId, type, { status: "success", precision: MODEL.precision, device: MODEL.device });
        } finally { modelState.isInitializing = false; }
        break;
      case "download":
        await MODEL.downloadModel(data?.precision, requestId);
        postResponse(requestId, type, { status: "success" });
        break;
      case "generate":
        if (!modelState.isInitialized) throw new Error("Call init before generate.");
        if (modelState.isGenerating) throw new Error("Generation already in progress.");
        modelState.isGenerating = true;
        try { await MODEL.generate(data, requestId); }
        finally { modelState.isGenerating = false; }
        break;
      case "clearMemory":
        await MODEL.clearMemory();
        modelState.isInitialized = false;
        modelState.isGenerating = false;
        postResponse(requestId, type, { status: "success" });
        break;
      default:
        throw new Error(`Unknown operation: ${type}`);
    }
  } catch (error) {
    postResponse(requestId, "error", { message: error instanceof Error ? error.message : String(error), context: type });
  }
}

MODEL = new WebAIModel();
self.addEventListener("message", handleMessage);
self.postMessage({
  type: "worker initialized",
  data: { success: true, workerVersion: CONFIG.WORKER_VERSION, transformersJsVersion: CONFIG.TRANSFORMERS_JS_VERSION, modelId: CONFIG.MODEL_ID },
});
