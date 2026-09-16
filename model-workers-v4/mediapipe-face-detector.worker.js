const CONFIG = {
  MODEL_ID: "mediapipe-face-detector",
  MODEL_URL: "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite?generation=1682480001393569",
  MODEL_VERSION: "blaze-face-short-range-float16-generation-1682480001393569",
  MODEL_BYTES: 229746,
  RUNTIME_VERSION: "1.0.1",
  BUNDLE_URL: "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/vision_bundle.js",
  WASM_URL: "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm",
  RUNTIME_BYTES: 12235800,
  PRECISIONS: {
    fp16: { size: 12465546, weightsSize: 229746, runtimeBytes: 12235800, supportedDevices: ["wasm"], modelKeys: ["blaze_face_short_range.tflite"], modelQuantization: "float16", runtimePrecision: "float32" },
  },
};

const MANIFEST = {
  contractVersion: "1.0",
  model: {
    id: CONFIG.MODEL_ID,
    displayName: "MediaPipe BlazeFace — Short-Range Face Detector",
    provider: "Google MediaPipe",
    providerUrl: "https://ai.google.dev/edge/mediapipe",
    license: "Apache-2.0",
    licenseUrl: "https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE",
    lastUpdated: "2023-04-26T03:33:21.000Z",
    sourceRepository: "google-ai-edge/mediapipe",
    repository: "mediapipe-models/face_detector/blaze_face_short_range",
    artifactRevision: CONFIG.MODEL_VERSION,
    artifactLastUpdated: "2023-04-26T03:33:21.000Z",
    task: "face-detection",
    description: "Detects nearby faces in one image locally in the browser with the official MediaPipe Face Detector task and returns boxes, confidence scores, and six facial keypoints.",
    intendedUses: ["Locating faces in selfies and short-range camera images.", "Driving non-identifying overlays or downstream face-processing UI."],
    limitations: ["Designed for short-range faces, typically within roughly two metres of the camera.", "Small, occluded, rotated, poorly lit, or profile faces can be missed.", "It detects faces but does not identify people or infer identity, emotion, health, or demographics.", "Do not use detection results for biometric identification or consequential decisions."],
  },
  runtime: { engine: { name: "MediaPipe Tasks Vision", version: CONFIG.RUNTIME_VERSION, backend: "WASM CPU", workerBridge: "classic-worker-inside-module-worker" }, precisions: CONFIG.PRECISIONS },
  input: { description: "Provide exactly one browser blob URL containing a decodable image.", alternatives: [{ name: "image blob", fields: [{ name: "image_blob_url", type: "string(blob URL)", required: true, description: "Image fetched and decoded inside the worker." }] }] },
  config: { model: [{ name: "minDetectionConfidence", type: "number", default: 0.5, minimum: 0, maximum: 1, description: "Minimum confidence retained by MediaPipe." }], generation: [] },
  output: { description: "MediaPipe detections in source-image pixel coordinates.", fields: [{ name: "detections", type: "Array<{ boundingBox, categories, keypoints }>", required: true }, { name: "image", type: "{ width: integer, height: integer }", required: true }, { name: "device", type: "string", required: true }], example: { detections: [{ boundingBox: { originX: 100, originY: 80, width: 220, height: 220, angle: 0 }, categories: [{ index: 0, score: 0.98, categoryName: "face", displayName: "" }], keypoints: [{ x: 0.42, y: 0.38, label: "right eye", score: 0 }] }], image: { width: 640, height: 480 }, device: "wasm" } },
  operations: ["checkModelSupports", "init", "download", "generate", "clearMemory"],
  benchmark: {
    report: "benchmarks/mediapipe-face-detector.json",
    testedAt: "2026-09-09T06:05:00.000Z",
    fixture: { file: "mediapipe-face-test.jpeg", url: "https://raw.githubusercontent.com/google-ai-edge/mediapipe-samples/7f3cf17410db91f8e084a46f7517df15bce3479a/examples/face_detector/ios/FaceDetectorTests/testImg.jpeg", size: 251136, sha1: "531674bd36742b56558d7879cc3c8461f7ddda31", width: 1280, height: 853, license: "Apache-2.0",
    licenseUrl: "https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE", sourceRevision: "google-ai-edge/mediapipe-samples@7f3cf17410db91f8e084a46f7517df15bce3479a" },
    environment: "Codex in-app Chromium 151 on macOS; MediaPipe Tasks Vision 1.0.1 WASM CPU",
    invalidatedBy: ["model artifact generation change", "MediaPipe Tasks Vision version change", "fixture or quality-gate change", "browser/backend behavior change"],
    results: [{ precision: "fp16", device: "wasm", status: "pass", runtime: { loadMs: 127.2, inferenceMs: 43.1, totalMs: 170.3, grade: "excellent" }, quality: { grade: "pass", expectedDetectionCount: 2, actualDetectionCount: 2, minimumScore: 0.925397515296936 } }],
  },
};

function mediaPipeBridge() {
  let detector;
  const reply = (id, type, data) => self.postMessage({ id, type, data });
  self.onmessage = async ({ data: message }) => {
    const { id, type, data } = message;
    try {
      if (type === "init") {
        detector?.close();
        if (!self.Vision) importScripts(data.bundleUrl);
        const fileset = await Vision.FilesetResolver.forVisionTasks(data.wasmUrl);
        detector = await Vision.FaceDetector.createFromOptions(fileset, { baseOptions: { modelAssetBuffer: new Uint8Array(data.model), delegate: "CPU" }, runningMode: "IMAGE", minDetectionConfidence: data.minDetectionConfidence });
        reply(id, "result", { status: "success" });
      } else if (type === "generate") {
        if (!detector) throw new Error("Call init before generate.");
        const pixels = new Uint8ClampedArray(data.pixels);
        const result = detector.detect(new ImageData(pixels, data.width, data.height));
        reply(id, "result", {
          detections: result.detections.map((item) => ({
            boundingBox: { originX: item.boundingBox.originX, originY: item.boundingBox.originY, width: item.boundingBox.width, height: item.boundingBox.height, angle: item.boundingBox.angle ?? 0 },
            categories: item.categories.map((category) => ({ index: category.index, score: category.score, categoryName: category.categoryName || "face", displayName: category.displayName || "" })),
            keypoints: item.keypoints.map((point) => ({ x: point.x, y: point.y, label: point.label ?? "", score: point.score ?? 0 })),
          })),
        });
      } else if (type === "clear") {
        detector?.close(); detector = null;
        reply(id, "result", { status: "success" });
      } else throw new Error(`Unknown bridge operation: ${type}`);
    } catch (error) { reply(id, "error", { message: error instanceof Error ? error.message : String(error) }); }
  };
}

const bridgeUrl = URL.createObjectURL(new Blob([`(${mediaPipeBridge.toString()})();`], { type: "text/javascript" }));
const bridge = new Worker(bridgeUrl);
URL.revokeObjectURL(bridgeUrl);
const pending = new Map();
let initialized = false;
let initializing = false;
let generating = false;
let modelBytes;
let selectedPrecision;
let selectedDevice;

bridge.addEventListener("message", ({ data }) => {
  const operation = pending.get(data.id);
  if (!operation) return;
  pending.delete(data.id);
  if (data.type === "error") operation.reject(new Error(data.data.message));
  else operation.resolve(data.data);
});

function callBridge(type, data, transfer = []) {
  const id = crypto.randomUUID();
  return new Promise((resolve, reject) => { pending.set(id, { resolve, reject }); bridge.postMessage({ id, type, data }, transfer); });
}
function respond(requestId, type, data) { self.postMessage({ requestId, type, data }); }

async function fetchModel(progress) {
  if (modelBytes) return modelBytes;
  const cache = await caches.open("webai-mediapipe-model-cache-v1");
  const cached = await cache.match(CONFIG.MODEL_URL);
  if (cached) { modelBytes = new Uint8Array(await cached.arrayBuffer()); return modelBytes; }
  const response = await fetch(CONFIG.MODEL_URL);
  if (!response.ok) throw new Error(`Model download failed: ${response.status} ${response.statusText}`);
  const cacheCopy = response.clone();
  const total = Number(response.headers.get("content-length")) || CONFIG.MODEL_BYTES;
  const reader = response.body.getReader();
  const chunks = [];
  let loaded = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value); loaded += value.byteLength;
    progress?.({ status: "progress", loaded, total, progress: total ? loaded / total * 100 : 0 });
  }
  await cache.put(CONFIG.MODEL_URL, cacheCopy);
  modelBytes = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) { modelBytes.set(chunk, offset); offset += chunk.byteLength; }
  return modelBytes;
}

async function decodeImage(blobUrl) {
  const response = await fetch(blobUrl);
  if (!response.ok) throw new Error(`Image fetch failed: ${response.status}`);
  const bitmap = await createImageBitmap(await response.blob());
  try {
    const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
    const context = canvas.getContext("2d", { willReadFrequently: true });
    context.drawImage(bitmap, 0, 0);
    const image = context.getImageData(0, 0, bitmap.width, bitmap.height);
    return { pixels: image.data, width: bitmap.width, height: bitmap.height };
  } finally { bitmap.close(); }
}

self.addEventListener("message", async ({ data: message = {} }) => {
  const { requestId, type, data } = message;
  try {
    if (type === "checkModelSupports") {
      respond(requestId, type, { manifest: MANIFEST, supportedModes: ["webai"], supportedPrecisions: Object.keys(CONFIG.PRECISIONS), supportedPrecisionsDevicesMap: CONFIG.PRECISIONS, doesSupportStreamGeneration: false, externalInterrupt: true, workerVersion: "v4", transformersJsVersion: null, runtimeName: "MediaPipe Tasks Vision", runtimeVersion: CONFIG.RUNTIME_VERSION, cacheModelId: MANIFEST.model.repository });
    } else if (type === "init") {
      if (initializing || generating) throw new Error("Worker is busy.");
      const precision = data?.precision ?? "fp16";
      const device = data?.device ?? "wasm";
      const entry = CONFIG.PRECISIONS[precision];
      if (!entry) throw new Error(`Unsupported precision: ${precision}`);
      if (!entry.supportedDevices.includes(device)) throw new Error(`Unsupported device: ${device}`);
      const minDetectionConfidence = data?.modelConfig?.minDetectionConfidence ?? 0.5;
      if (typeof minDetectionConfidence !== "number" || minDetectionConfidence < 0 || minDetectionConfidence > 1) throw new Error("modelConfig.minDetectionConfidence must be from 0 through 1.");
      initializing = true; initialized = false;
      try {
        const bytes = await fetchModel((value) => respond(requestId, "downloadProgress", value));
        const copy = bytes.slice();
        await callBridge("init", { bundleUrl: CONFIG.BUNDLE_URL, wasmUrl: CONFIG.WASM_URL, model: copy.buffer, minDetectionConfidence }, [copy.buffer]);
        selectedPrecision = precision; selectedDevice = device; initialized = true;
        respond(requestId, type, { status: "success", precision, device });
      } finally { initializing = false; }
    } else if (type === "download") {
      const precision = data?.precision ?? "fp16";
      if (!CONFIG.PRECISIONS[precision]) throw new Error(`Unsupported precision: ${precision}`);
      await fetchModel((value) => respond(requestId, "downloadProgress", value));
      respond(requestId, type, { status: "success" });
    } else if (type === "generate") {
      if (!initialized) throw new Error("Call init before generate.");
      if (generating) throw new Error("Generation already in progress.");
      const userInput = data?.userInput;
      if (!userInput || Object.keys(userInput).filter((key) => userInput[key] != null).length !== 1 || typeof userInput.image_blob_url !== "string" || !userInput.image_blob_url.startsWith("blob:")) throw new Error("userInput must contain exactly one valid image_blob_url.");
      generating = true;
      try {
        const image = await decodeImage(userInput.image_blob_url);
        const result = await callBridge("generate", { pixels: image.pixels.buffer, width: image.width, height: image.height }, [image.pixels.buffer]);
        respond(requestId, "generated", { status: "success", result: { ...result, image: { width: image.width, height: image.height }, device: selectedDevice, precision: selectedPrecision } });
      } finally { generating = false; }
    } else if (type === "clearMemory") {
      await callBridge("clear", {}); modelBytes = null; initialized = false;
      respond(requestId, type, { status: "success" });
    } else throw new Error(`Unknown operation: ${type}`);
  } catch (error) { respond(requestId, "error", { message: error instanceof Error ? error.message : String(error), context: type }); }
});

self.postMessage({ type: "worker initialized", data: { success: true, workerVersion: "v4", modelId: CONFIG.MODEL_ID, runtimeName: "MediaPipe Tasks Vision", runtimeVersion: CONFIG.RUNTIME_VERSION } });
