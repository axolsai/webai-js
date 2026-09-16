const CONFIG = {
  MODEL_ID: "mediapipe-hand-landmarker",
  MODEL_URL: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task?generation=1682480005356399",
  MODEL_VERSION: "hand-landmarker-float16-generation-1682480005356399",
  MODEL_BYTES: 7819105,
  RUNTIME_VERSION: "1.0.1",
  BUNDLE_URL: "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/vision_bundle.js",
  WASM_URL: "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm",
  RUNTIME_BYTES: 12235800,
  PRECISIONS: {
    fp16: { size: 20054905, weightsSize: 7819105, runtimeBytes: 12235800, supportedDevices: ["wasm"], modelKeys: ["hand_landmarker.task"], modelQuantization: "float16", runtimePrecision: "float32" },
  },
};

const MANIFEST = {
  contractVersion: "1.0",
  model: {
    id: CONFIG.MODEL_ID,
    displayName: "MediaPipe Hand Landmarker",
    provider: "Google MediaPipe",
    providerUrl: "https://ai.google.dev/edge/mediapipe",
    license: "Apache-2.0",
    licenseUrl: "https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE",
    lastUpdated: "2023-04-26T03:33:21.000Z",
    sourceRepository: "google-ai-edge/mediapipe",
    repository: "mediapipe-models/hand_landmarker/hand_landmarker",
    artifactRevision: CONFIG.MODEL_VERSION,
    artifactLastUpdated: "2023-04-26T03:33:21.000Z",
    task: "hand-landmark-detection",
    description: "Detects hands in one image locally in the browser and returns 21 normalized image landmarks, 21 world-coordinate landmarks, and handedness classifications for each detected hand.",
    intendedUses: ["Gesture interfaces and hand-overlay prototypes.", "Extracting hand geometry for non-consequential interaction or analysis."],
    limitations: ["Hands that are small, cropped, occluded, blurred, poorly lit, or in unusual poses can be missed or receive inaccurate landmarks.", "Left/right handedness can be unreliable for mirrored or ambiguous images.", "World landmarks are model estimates, not calibrated physical measurements.", "Do not use landmarks or handedness for identity, medical, safety, or other consequential decisions."],
  },
  runtime: { engine: { name: "MediaPipe Tasks Vision", version: CONFIG.RUNTIME_VERSION, backend: "WASM CPU", workerBridge: "classic-worker-inside-module-worker" }, precisions: CONFIG.PRECISIONS },
  input: { description: "Provide exactly one browser blob URL containing a decodable image.", alternatives: [{ name: "image blob", fields: [{ name: "image_blob_url", type: "string(blob URL)", required: true, description: "Image fetched and decoded inside the worker." }] }] },
  config: { model: [{ name: "numHands", type: "integer", default: 2, minimum: 1, maximum: 4, description: "Maximum hands returned." }, { name: "minHandDetectionConfidence", type: "number", default: 0.5, minimum: 0, maximum: 1 }, { name: "minHandPresenceConfidence", type: "number", default: 0.5, minimum: 0, maximum: 1 }], generation: [] },
  output: { description: "Per-hand handedness and 21-point image/world landmark arrays.", fields: [{ name: "handednesses", type: "Array<Array<Category>>", required: true }, { name: "landmarks", type: "Array<Array<{x,y,z,visibility}>>", required: true }, { name: "worldLandmarks", type: "Array<Array<{x,y,z,visibility}>>", required: true }, { name: "image", type: "{ width: integer, height: integer }", required: true }], example: { handednesses: [[{ categoryName: "Right", score: 0.98, index: 1, displayName: "Right" }]], landmarks: [[{ x: 0.5, y: 0.7, z: 0, visibility: 0 }]], worldLandmarks: [[{ x: 0.01, y: 0.02, z: -0.03, visibility: 0 }]], image: { width: 640, height: 480 }, device: "wasm" } },
  operations: ["checkModelSupports", "init", "download", "generate", "clearMemory"],
  benchmark: {
    report: "benchmarks/mediapipe-hand-landmarker.json",
    testedAt: "2026-09-09T06:08:00.000Z",
    fixture: { file: "mediapipe-hand-test.jpg", url: "https://raw.githubusercontent.com/google-ai-edge/mediapipe-samples/7f3cf17410db91f8e084a46f7517df15bce3479a/examples/hand_landmarker/android/app/src/androidTest/assets/test_image.jpg", size: 33051, sha1: "d44bc4fbfbc8587039ab32df880162dca7fe842e", width: 640, height: 427, license: "Apache-2.0",
    licenseUrl: "https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE", sourceRevision: "google-ai-edge/mediapipe-samples@7f3cf17410db91f8e084a46f7517df15bce3479a" },
    environment: "Codex in-app Chromium 151 on macOS; MediaPipe Tasks Vision 1.0.1 WASM CPU",
    invalidatedBy: ["model artifact generation change", "MediaPipe Tasks Vision version change", "fixture or quality-gate change", "browser/backend behavior change"],
    results: [{ precision: "fp16", device: "wasm", status: "pass", runtime: { loadMs: 2261.4, inferenceMs: 82.6, totalMs: 2344, grade: "excellent" }, quality: { grade: "pass", expectedHandCount: 1, actualHandCount: 1, imageLandmarkCount: 21, worldLandmarkCount: 21, handedness: "Right", handednessScore: 0.703342080116272 } }],
  },
};

function mediaPipeBridge() {
  let landmarker;
  const reply = (id, type, data) => self.postMessage({ id, type, data });
  self.onmessage = async ({ data: message }) => {
    const { id, type, data } = message;
    try {
      if (type === "init") {
        landmarker?.close();
        if (!self.Vision) importScripts(data.bundleUrl);
        const fileset = await Vision.FilesetResolver.forVisionTasks(data.wasmUrl);
        landmarker = await Vision.HandLandmarker.createFromOptions(fileset, { baseOptions: { modelAssetBuffer: new Uint8Array(data.model), delegate: "CPU" }, runningMode: "IMAGE", numHands: data.numHands, minHandDetectionConfidence: data.minHandDetectionConfidence, minHandPresenceConfidence: data.minHandPresenceConfidence });
        reply(id, "result", { status: "success" });
      } else if (type === "generate") {
        if (!landmarker) throw new Error("Call init before generate.");
        const pixels = new Uint8ClampedArray(data.pixels);
        const result = landmarker.detect(new ImageData(pixels, data.width, data.height));
        const mapPoint = (point) => ({ x: point.x, y: point.y, z: point.z, visibility: point.visibility ?? 0 });
        const mapCategory = (category) => ({ index: category.index, score: category.score, categoryName: category.categoryName, displayName: category.displayName });
        reply(id, "result", { handednesses: result.handednesses.map((group) => group.map(mapCategory)), landmarks: result.landmarks.map((group) => group.map(mapPoint)), worldLandmarks: result.worldLandmarks.map((group) => group.map(mapPoint)) });
      } else if (type === "clear") {
        landmarker?.close(); landmarker = null;
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
      const numHands = data?.modelConfig?.numHands ?? 2;
      const minHandDetectionConfidence = data?.modelConfig?.minHandDetectionConfidence ?? 0.5;
      const minHandPresenceConfidence = data?.modelConfig?.minHandPresenceConfidence ?? 0.5;
      if (!Number.isInteger(numHands) || numHands < 1 || numHands > 4) throw new Error("modelConfig.numHands must be an integer from 1 through 4.");
      for (const [name, value] of Object.entries({ minHandDetectionConfidence, minHandPresenceConfidence })) if (typeof value !== "number" || value < 0 || value > 1) throw new Error(`modelConfig.${name} must be from 0 through 1.`);
      initializing = true; initialized = false;
      try {
        const bytes = await fetchModel((value) => respond(requestId, "downloadProgress", value));
        const copy = bytes.slice();
        await callBridge("init", { bundleUrl: CONFIG.BUNDLE_URL, wasmUrl: CONFIG.WASM_URL, model: copy.buffer, numHands, minHandDetectionConfidence, minHandPresenceConfidence }, [copy.buffer]);
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
