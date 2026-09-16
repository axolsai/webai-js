const CONFIG = {
  MODEL_ID: "lightweight-openpose-litert",
  MODEL_REVISION: "7a1d619e67f829fc84c4b7a7e271384b5508762e",
  MODEL_URLS: {
    fp32: "https://huggingface.co/litert-community/lightweight-openpose/resolve/7a1d619e67f829fc84c4b7a7e271384b5508762e/pose_256.tflite",
    fp16: "https://huggingface.co/litert-community/lightweight-openpose/resolve/7a1d619e67f829fc84c4b7a7e271384b5508762e/pose_256_fp16.tflite",
  },
  LITERT_MODULE_URL: "https://cdn.jsdelivr.net/npm/@litertjs/core@2.5.3/+esm",
  LITERT_WASM_URL: "https://cdn.jsdelivr.net/npm/@litertjs/core@2.5.3/wasm/",
  RUNTIME_VERSION: "2.5.3",
  PRECISIONS: {
    fp32: { size: 26316941, weightsSize: 16405352, runtimeBytes: 9911589, sharedModelBytes: 0, supportedDevices: ["wasm", "webgpu"], modelKeys: ["pose_256.tflite"], dtype: "fp32" },
    fp16: { size: 18250629, weightsSize: 8339040, runtimeBytes: 9911589, sharedModelBytes: 0, supportedDevices: ["wasm", "webgpu"], modelKeys: ["pose_256_fp16.tflite"], dtype: "fp16" },
  },
};

const KEYPOINT_NAMES = ["nose", "neck", "right shoulder", "right elbow", "right wrist", "left shoulder", "left elbow", "left wrist", "right hip", "right knee", "right ankle", "left hip", "left knee", "left ankle", "right eye", "left eye", "right ear", "left ear"];

const MANIFEST = {
  contractVersion: "1.0",
  model: {
    id: CONFIG.MODEL_ID,
    displayName: "Lightweight OpenPose — LiteRT Pose Estimation",
    provider: "Daniil Osokin",
    providerUrl: "https://github.com/Daniil-Osokin",
    license: "Apache-2.0",
    licenseUrl: "https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch/blob/master/LICENSE",
    lastUpdated: "2024-04-30T09:54:40.000Z",
    sourceRepository: "Daniil-Osokin/lightweight-human-pose-estimation.pytorch",
    sourceRevision: "d23c284b09acf27a163e1febd511e7482cac25ed",
    repository: "litert-community/lightweight-openpose",
    artifactRevision: CONFIG.MODEL_REVISION,
    artifactLastUpdated: "2026-09-01T01:45:50.000Z",
    task: "keypoint-detection",
    description: "A Lightweight OpenPose heatmap model that estimates 18 anonymous human body keypoints locally in the browser using LiteRT.js.",
    intendedUses: ["Single-person pose visualization.", "Gesture and exercise prototypes that do not require identity recognition."],
    limitations: ["The simple argmax decoder returns one location per keypoint channel and is intended for one prominent person, not full multi-person association.", "Occlusion, cropped bodies, unusual poses, blur, and small people reduce accuracy.", "Coordinates are estimates and must not be used for medical, biometric, safety, or surveillance decisions."],
  },
  runtime: { engine: { name: "LiteRT.js", version: CONFIG.RUNTIME_VERSION, workerBridge: "classic-worker-inside-module-worker" }, precisions: CONFIG.PRECISIONS },
  input: { description: "Provide exactly one browser blob URL containing a decodable image.", alternatives: [{ name: "image blob", fields: [{ name: "image_blob_url", type: "string(blob URL)", required: true, description: "Image resized to 256×256 RGB and normalized to (pixel - 128) / 256." }] }] },
  config: { model: [], generation: [{ name: "minimumConfidence", type: "number", default: 0.1, minimum: 0, maximum: 1, description: "Marks decoded keypoints below this heatmap confidence as not visible." }] },
  output: { description: "Eighteen normalized body-keypoint coordinates decoded from 32×32 heatmaps.", fields: [{ name: "keypoints", type: "Array<{ name: string, x: number, y: number, confidence: number, visible: boolean }>", required: true }, { name: "fullyAccelerated", type: "boolean", required: true }], example: { keypoints: [{ name: "nose", x: 0.5, y: 0.2, confidence: 0.9, visible: true }], fullyAccelerated: true } },
  operations: ["checkModelSupports", "init", "download", "generate", "clearMemory"],
  benchmark: {
    report: "benchmarks/lightweight-openpose-litert.json",
    testedAt: "2026-09-08T17:48:00.000Z",
    fixture: { file: "modnet-portrait-pexels-5965592.jpg", url: "https://f005.backblazeb2.com/file/hubters-webai-public/assets/benchmarks/modnet-portrait-pexels-5965592.jpg", size: 78124, sha1: "38212739a4a4f0be25d84863cf991d7fb6c6d818", width: 1024, height: 683, license: "Pexels License" },
    environment: "Headless Chrome 151 on macOS; LiteRT.js 2.5.3",
    invalidatedBy: ["model artifact revision change", "LiteRT.js version change", "fixture or quality-gate change", "browser/backend behavior change"],
    results: [
      { precision: "fp32", device: "wasm", status: "pass", runtime: { loadMs: 7723.7, inferenceMs: 166.2, totalMs: 7889.9, grade: "good" }, quality: { grade: "reference", visibleKeypoints: 4 }, fullyAccelerated: false },
      { precision: "fp32", device: "webgpu", status: "pass", runtime: { loadMs: 9065.5, inferenceMs: 536, totalMs: 9601.5, grade: "slow" }, quality: { grade: "excellent", matchingGridLocations: 18, maximumConfidenceDifference: 0.000008 }, fullyAccelerated: true },
      { precision: "fp16", device: "wasm", status: "pass", runtime: { loadMs: 6411.7, inferenceMs: 714.8, totalMs: 7126.5, grade: "slow" }, quality: { grade: "acceptable", matchingGridLocations: 16, maximumGridDistance: 0.0625 }, fullyAccelerated: false },
      { precision: "fp16", device: "webgpu", status: "pass", runtime: { loadMs: 6405.9, inferenceMs: 50.9, totalMs: 6456.8, grade: "good" }, quality: { grade: "acceptable", matchingGridLocations: 16, maximumGridDistance: 0.0625 }, fullyAccelerated: true }
    ],
  },
};

function liteRtBridge() {
  let api;
  let model;
  let fullyAccelerated = false;

  const reply = (id, type, data) => self.postMessage({ id, type, data });
  async function fetchBytes(url, id) {
    const cache = await caches.open("webai-litert-model-cache-v1");
    const cached = await cache.match(url);
    if (cached) return new Uint8Array(await cached.arrayBuffer());
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Download failed: ${response.status} ${response.statusText}`);
    const cacheCopy = response.clone();
    const total = Number(response.headers.get("content-length")) || 0;
    const reader = response.body.getReader();
    const chunks = [];
    let loaded = 0;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value); loaded += value.byteLength;
      reply(id, "progress", { status: "progress", loaded, total, progress: total ? loaded / total * 100 : 0 });
    }
    await cache.put(url, cacheCopy);
    const bytes = new Uint8Array(loaded);
    let offset = 0;
    for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.byteLength; }
    return bytes;
  }

  self.onmessage = async ({ data: message }) => {
    const { id, type, data } = message;
    try {
      if (type === "init") {
        if (model) { model.delete(); model = null; }
        if (!api) {
          const nativeFetch = self.fetch.bind(self);
          self.fetch = (input, init) => nativeFetch(typeof input === "string" && !/^[a-z]+:/i.test(input) ? new URL(input, data.wasmUrl).href : input, init);
          const nativeOpen = XMLHttpRequest.prototype.open;
          XMLHttpRequest.prototype.open = function(method, url, ...rest) {
            return nativeOpen.call(this, method, typeof url === "string" && !/^[a-z]+:/i.test(url) ? new URL(url, data.wasmUrl).href : url, ...rest);
          };
          api = await import(data.moduleUrl);
          await api.loadLiteRt(data.wasmUrl, { jspi: true });
        }
        const bytes = await fetchBytes(data.modelUrl, id);
        model = await api.loadAndCompile(bytes, { accelerator: data.device });
        const input = model.getInputDetails()[0];
        const output = model.getOutputDetails()[0];
        const inputShape = Array.from(input.shape);
        const outputShape = Array.from(output.shape);
        if (JSON.stringify(inputShape) !== JSON.stringify([1, 256, 256, 3])) throw new Error(`Unexpected input shape: ${JSON.stringify(inputShape)}`);
        if (JSON.stringify(outputShape) !== JSON.stringify([1, 32, 32, 19])) throw new Error(`Unexpected output shape: ${JSON.stringify(outputShape)}`);
        fullyAccelerated = model.isFullyAccelerated;
        reply(id, "result", { fullyAccelerated, input: inputShape, output: outputShape });
      } else if (type === "download") {
        await fetchBytes(data.modelUrl, id);
        reply(id, "result", { status: "success" });
      } else if (type === "generate") {
        if (!model) throw new Error("Call init before generate.");
        const input = new api.Tensor(data.input, [1, 256, 256, 3]);
        let outputs;
        try {
          outputs = await model.run(input);
          const scores = await outputs[0].data();
          reply(id, "result", { scores: Array.from(scores), fullyAccelerated });
        } finally {
          input.delete();
          if (Array.isArray(outputs)) for (const output of outputs) output.delete();
        }
      } else if (type === "clear") {
        if (model) { model.delete(); model = null; }
        reply(id, "result", { status: "success" });
      } else throw new Error(`Unknown bridge operation: ${type}`);
    } catch (error) { reply(id, "error", { message: error instanceof Error ? error.message : String(error) }); }
  };
}

const bridgeUrl = URL.createObjectURL(new Blob([`(${liteRtBridge.toString()})();`], { type: "text/javascript" }));
const bridge = new Worker(bridgeUrl);
const pending = new Map();
let initialized = false;
let initializing = false;
let generating = false;
let selectedPrecision;
let selectedDevice;

bridge.addEventListener("message", ({ data }) => {
  const operation = pending.get(data.id);
  if (!operation) return;
  if (data.type === "progress") { operation.progress?.(data.data); return; }
  pending.delete(data.id);
  if (data.type === "error") operation.reject(new Error(data.data.message));
  else operation.resolve(data.data);
});

function callBridge(type, data, progress) {
  const id = crypto.randomUUID();
  return new Promise((resolve, reject) => { pending.set(id, { resolve, reject, progress }); bridge.postMessage({ id, type, data }, data?.input ? [data.input.buffer] : []); });
}
function respond(requestId, type, data) { self.postMessage({ requestId, type, data }); }

async function preprocess(blobUrl) {
  const response = await fetch(blobUrl);
  if (!response.ok) throw new Error(`Image fetch failed: ${response.status}`);
  const bitmap = await createImageBitmap(await response.blob());
  const canvas = new OffscreenCanvas(256, 256);
  const context = canvas.getContext("2d", { willReadFrequently: true });
  context.drawImage(bitmap, 0, 0, bitmap.width, bitmap.height, 0, 0, 256, 256);
  bitmap.close();
  const rgba = context.getImageData(0, 0, 256, 256).data;
  const input = new Float32Array(256 * 256 * 3);
  for (let pixel = 0; pixel < 256 * 256; pixel++) for (let channel = 0; channel < 3; channel++) input[pixel * 3 + channel] = (rgba[pixel * 4 + channel] - 128) / 256;
  return input;
}

self.addEventListener("message", async ({ data: message = {} }) => {
  const { requestId, type, data } = message;
  try {
    if (type === "checkModelSupports") {
      respond(requestId, type, { manifest: MANIFEST, supportedModes: ["webai"], supportedPrecisions: Object.keys(CONFIG.PRECISIONS), supportedPrecisionsDevicesMap: CONFIG.PRECISIONS, doesSupportStreamGeneration: false, externalInterrupt: true, workerVersion: "v4", transformersJsVersion: null, runtimeName: "LiteRT.js", runtimeVersion: CONFIG.RUNTIME_VERSION, cacheModelId: MANIFEST.model.repository });
    } else if (type === "init") {
      if (initializing || generating) throw new Error("Worker is busy.");
      const entry = CONFIG.PRECISIONS[data?.precision];
      if (!entry) throw new Error(`Unsupported precision: ${data?.precision}`);
      if (!entry.supportedDevices.includes(data?.device)) throw new Error(`Unsupported device: ${data?.device}`);
      initializing = true; initialized = false;
      try {
        await callBridge("init", { moduleUrl: CONFIG.LITERT_MODULE_URL, wasmUrl: CONFIG.LITERT_WASM_URL, modelUrl: CONFIG.MODEL_URLS[data.precision], device: data.device }, (value) => respond(requestId, "downloadProgress", value));
        selectedPrecision = data.precision; selectedDevice = data.device; initialized = true;
        respond(requestId, type, { status: "success", precision: selectedPrecision, device: selectedDevice });
      } finally { initializing = false; }
    } else if (type === "download") {
      if (!CONFIG.PRECISIONS[data?.precision]) throw new Error(`Unsupported precision: ${data?.precision}`);
      await callBridge("download", { modelUrl: CONFIG.MODEL_URLS[data.precision] }, (value) => respond(requestId, "downloadProgress", value));
      respond(requestId, type, { status: "success" });
    } else if (type === "generate") {
      if (!initialized) throw new Error("Call init before generate.");
      if (generating) throw new Error("Generation already in progress.");
      const userInput = data?.userInput;
      if (!userInput || Object.keys(userInput).filter((key) => userInput[key] != null).length !== 1 || typeof userInput.image_blob_url !== "string" || !userInput.image_blob_url.startsWith("blob:")) throw new Error("userInput must contain exactly one valid image_blob_url.");
      const minimumConfidence = data?.generateConfig?.minimumConfidence ?? 0.1;
      if (!Number.isFinite(minimumConfidence) || minimumConfidence < 0 || minimumConfidence > 1) throw new Error("generateConfig.minimumConfidence must be from 0 through 1.");
      generating = true;
      try {
        const result = await callBridge("generate", { input: await preprocess(userInput.image_blob_url) });
        const keypoints = KEYPOINT_NAMES.map((name, channel) => {
          let bestIndex = 0, confidence = -Infinity;
          for (let pixel = 0; pixel < 32 * 32; pixel++) { const score = result.scores[pixel * 19 + channel]; if (score > confidence) { confidence = score; bestIndex = pixel; } }
          return { name, x: ((bestIndex % 32) + 0.5) / 32, y: (Math.floor(bestIndex / 32) + 0.5) / 32, confidence, visible: confidence >= minimumConfidence };
        });
        respond(requestId, "generated", { status: "success", result: { keypoints, fullyAccelerated: result.fullyAccelerated } });
      } finally { generating = false; }
    } else if (type === "clearMemory") {
      await callBridge("clear", {}); initialized = false;
      respond(requestId, type, { status: "success" });
    } else throw new Error(`Unknown operation: ${type}`);
  } catch (error) { respond(requestId, "error", { message: error instanceof Error ? error.message : String(error), context: type }); }
});

self.postMessage({ type: "worker initialized", data: { success: true, workerVersion: "v4", modelId: CONFIG.MODEL_ID, runtimeName: "LiteRT.js", runtimeVersion: CONFIG.RUNTIME_VERSION } });
