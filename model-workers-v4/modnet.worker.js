import {
  env,
  AutoModel,
  AutoProcessor,
  RawImage,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0";

const MODEL_CONFIG_FIELDS = [
  { name: "processingResolution", type: "{ width: integer, height: integer }", default: { width: 512, height: 512 }, description: "Model input dimensions; each value must be from 32 through 2048 and divisible by 32." },
];

const GENERATION_CONFIG_FIELDS = [
  { name: "threshold", type: "number | null", default: null, minimum: 0, maximum: 1, description: "Keep null for a soft alpha matte, or set 0–1 to return a binary foreground mask." },
];

const CONFIG = {
  MODEL_ID: "modnet",
  MODEL_REPOSITORY: "Xenova/modnet",
  MODEL_REVISION: "fa2fa546052fba4c08921230a26cc69a333fca12",
  WORKER_VERSION: "v4",
  TRANSFORMERS_JS_VERSION: "4.2.0",
  EXTERNAL_INTERRUPT: true,
  SUPPORTED_MODES: ["webai"],
  SUPPORTED_PRECISIONS_DEVICES_MAP: {
    fp32: { size: 25889088, weightsSize: 25888640, supportedDevices: ["wasm", "webgpu"], modelKeys: ["model.onnx"], dtype: "fp32" },
    fp16: { size: 12985229, weightsSize: 12984781, supportedDevices: ["wasm", "webgpu"], modelKeys: ["model_fp16.onnx"], dtype: "fp16" },
    q8: { size: 6632636, weightsSize: 6632188, supportedDevices: ["wasm", "webgpu"], modelKeys: ["model_quantized.onnx"], dtype: "q8" },
    uint8: { size: 6627496, weightsSize: 6627048, supportedDevices: ["wasm", "webgpu"], modelKeys: ["model_uint8.onnx"], dtype: "uint8" },
    q4: { size: 23132531, weightsSize: 23132083, supportedDevices: ["wasm"], modelKeys: ["model_q4.onnx"], dtype: "q4" },
    q4f16: { size: 11802379, weightsSize: 11801931, supportedDevices: ["wasm"], modelKeys: ["model_q4f16.onnx"], dtype: "q4f16" },
    bnb4: { size: 23081347, weightsSize: 23080899, supportedDevices: ["wasm"], modelKeys: ["model_bnb4.onnx"], dtype: "bnb4" },
  },
  DEFAULT_MODEL_CONFIG: { processingResolution: { width: 512, height: 512 } },
  DEFAULT_GENERATION_CONFIG: { threshold: null },
};

const BENCHMARK_RESULTS = [
  { precision: "fp32", device: "wasm", status: "pass", runtime: { loadMs: 6575.6, inferenceMs: 1185.5, totalMs: 7831.2, grade: "very-slow", megapixelsPerSecond: 0.59, loadGrade: "slow" }, quality: { grade: "excellent", meanAbsoluteError: 0, intersectionOverUnion: 1 } },
  { precision: "fp32", device: "webgpu", status: "pass", runtime: { loadMs: 6890.2, inferenceMs: 851.4, totalMs: 7805.2, grade: "slow", megapixelsPerSecond: 0.8215, loadGrade: "slow" }, quality: { grade: "excellent", meanAbsoluteError: 0.000009, intersectionOverUnion: 1 } },
  { precision: "fp16", device: "wasm", status: "pass", runtime: { loadMs: 5623.6, inferenceMs: 1180.4, totalMs: 6867.9, grade: "very-slow", megapixelsPerSecond: 0.5925, loadGrade: "slow" }, quality: { grade: "excellent", meanAbsoluteError: 0.000027, intersectionOverUnion: 0.999938 } },
  { precision: "fp16", device: "webgpu", status: "pass", runtime: { loadMs: 6496.3, inferenceMs: 843.5, totalMs: 7403.1, grade: "slow", megapixelsPerSecond: 0.8292, loadGrade: "slow" }, quality: { grade: "excellent", meanAbsoluteError: 0.000807, intersectionOverUnion: 0.999905 } },
  { precision: "q8", device: "wasm", status: "pass", runtime: { loadMs: 4693.3, inferenceMs: 1231.5, totalMs: 5991.9, grade: "very-slow", megapixelsPerSecond: 0.5679, loadGrade: "moderate" }, quality: { grade: "excellent", meanAbsoluteError: 0.004815, intersectionOverUnion: 0.99112 } },
  { precision: "q8", device: "webgpu", status: "pass", runtime: { loadMs: 4709.8, inferenceMs: 1519.9, totalMs: 6297, grade: "very-slow", megapixelsPerSecond: 0.4602, loadGrade: "moderate" }, quality: { grade: "excellent", meanAbsoluteError: 0.004812, intersectionOverUnion: 0.991242 } },
  { precision: "uint8", device: "wasm", status: "pass", runtime: { loadMs: 4949.2, inferenceMs: 1235.4, totalMs: 6248.4, grade: "very-slow", megapixelsPerSecond: 0.5661, loadGrade: "moderate" }, quality: { grade: "excellent", meanAbsoluteError: 0.005127, intersectionOverUnion: 0.990214 } },
  { precision: "uint8", device: "webgpu", status: "pass", runtime: { loadMs: 4961.6, inferenceMs: 1376.9, totalMs: 6407, grade: "very-slow", megapixelsPerSecond: 0.5079, loadGrade: "moderate" }, quality: { grade: "excellent", meanAbsoluteError: 0.005135, intersectionOverUnion: 0.990148 } },
  { precision: "q4", device: "wasm", status: "pass", runtime: { loadMs: 6319.5, inferenceMs: 1140.6, totalMs: 7527.3, grade: "very-slow", megapixelsPerSecond: 0.6132, loadGrade: "slow" }, quality: { grade: "excellent", meanAbsoluteError: 0.00003, intersectionOverUnion: 0.99995 } },
  { precision: "q4", device: "webgpu", status: "fail", runtime: { loadMs: 7518.9, inferenceMs: 228.1, totalMs: 7813.2, grade: "good", megapixelsPerSecond: 3.0662, loadGrade: "slow" }, quality: { grade: "failed", meanAbsoluteError: 0.135041, intersectionOverUnion: 0.555223 }, failure: "quality-mask-mismatch" },
  { precision: "q4f16", device: "wasm", status: "pass", runtime: { loadMs: 5192.3, inferenceMs: 1178.3, totalMs: 6435.4, grade: "very-slow", megapixelsPerSecond: 0.5936, loadGrade: "slow" }, quality: { grade: "excellent", meanAbsoluteError: 0.000046, intersectionOverUnion: 0.999916 } },
  { precision: "q4f16", device: "webgpu", status: "fail", runtime: { loadMs: 4900.2, inferenceMs: 237.6, totalMs: 5201.4, grade: "good", megapixelsPerSecond: 2.9436, loadGrade: "moderate" }, quality: { grade: "failed", meanAbsoluteError: 0.135484, intersectionOverUnion: 0.55381 }, failure: "quality-mask-mismatch" },
  { precision: "bnb4", device: "wasm", status: "pass", runtime: { loadMs: 7617.4, inferenceMs: 1145.6, totalMs: 8833.4, grade: "very-slow", megapixelsPerSecond: 0.6105, loadGrade: "slow" }, quality: { grade: "excellent", meanAbsoluteError: 0.000041, intersectionOverUnion: 0.999899 } },
  { precision: "bnb4", device: "webgpu", status: "fail", runtime: { loadMs: 6497.7, inferenceMs: 114, totalMs: 6678.1, grade: "good", megapixelsPerSecond: 6.135, loadGrade: "slow" }, quality: { grade: "failed", meanAbsoluteError: 0.135233, intersectionOverUnion: 0.554619 }, failure: "quality-mask-mismatch" },
];

const MANIFEST = {
  contractVersion: "1.0",
  model: {
    id: CONFIG.MODEL_ID,
    displayName: "MODNet — Portrait Matting",
    provider: "ZHKKKe",
    providerUrl: "https://github.com/ZHKKKe",
    license: "Apache-2.0",
    licenseUrl: "https://github.com/ZHKKKe/MODNet/blob/master/LICENSE",
    lastUpdated: "2024-05-06T14:28:28.000Z",
    sourceRepository: "ZHKKKe/MODNet",
    repository: CONFIG.MODEL_REPOSITORY,
    artifactRevision: CONFIG.MODEL_REVISION,
    artifactLastUpdated: "2025-10-26T21:25:48.000Z",
    task: "background-removal",
    description: "A trimap-free portrait-matting model that runs locally in the browser and estimates a per-pixel foreground alpha matte for images containing people.",
    intendedUses: [
      "Removing or replacing the background of a single-person portrait.",
      "Creating a soft alpha matte for compositing portraits in browser applications.",
      "Prototyping client-side portrait segmentation without a hosted inference API.",
    ],
    limitations: [
      "MODNet is specialized for human portraits and is not a general object-segmentation model.",
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
    report: "benchmarks/modnet.json",
    testedAt: "2026-09-08T11:48:18.538Z",
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
    reference: { file: "modnet-portrait-pexels-5965592-mask.png", size: 67541, sha1: "01ace4f95454ce44cc53c2c797a87a583c86cee8", width: 1024, height: 683, generator: { precision: "fp32", device: "wasm", processingResolution: { width: 512, height: 512 } } },
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
    this.model = null;
    this.processor = null;
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
      this.model = await AutoModel.from_pretrained(this.model_id, {
        dtype: entry.dtype,
        device: this.device,
        ...(this.revision ? { revision: this.revision } : {}),
        progress_callback: (value) => postProgress(requestId, value),
      });
      this.processor = await AutoProcessor.from_pretrained(this.model_id, {
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
    const downloadedModel = await AutoModel.from_pretrained(this.model_id, {
      dtype: entry.dtype,
      device: entry.supportedDevices.includes("wasm") ? "wasm" : "webgpu",
      ...(this.revision ? { revision: this.revision } : {}),
      progress_callback: (value) => postProgress(requestId, value),
    });
    try {
      await AutoProcessor.from_pretrained(this.model_id, {
        ...(this.revision ? { revision: this.revision } : {}),
        progress_callback: (value) => postProgress(requestId, value),
      });
    } finally {
      await downloadedModel.dispose?.();
    }
  }

  revokeOutputs() {
    for (const url of this.outputUrls) URL.revokeObjectURL(url);
    this.outputUrls = [];
  }

  async generate(data, requestId) {
    if (!this.model || !this.processor) throw new Error("Call init before generate.");
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
    this.processor.config.size = finalModelConfig.processingResolution;
    const { pixel_values } = await this.processor(image);
    const { output } = await this.model({ input: pixel_values });
    let matte = output[0];
    if (finalGenerateConfig.threshold !== null) {
      matte = matte.map((value) => value >= finalGenerateConfig.threshold ? 1 : 0);
    }
    const mask = await RawImage.fromTensor(matte.mul(255).to("uint8")).resize(image.width, image.height);
    image.putAlpha(mask);
    const [transparentBlob, maskBlob] = await Promise.all([image.toBlob("image/png"), mask.toBlob("image/png")]);
    this.revokeOutputs();
    const result = URL.createObjectURL(transparentBlob);
    const resultMask = URL.createObjectURL(maskBlob);
    this.outputUrls.push(result, resultMask);
    postResponse(requestId, "generated", { status: "success", result: { result, result_mask: resultMask, mimeType: "image/png", size: transparentBlob.size, maskSize: maskBlob.size, width: image.width, height: image.height } });
  }

  async disposeModel() {
    const model = this.model;
    this.model = null;
    this.processor = null;
    if (model) await model.dispose?.();
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
