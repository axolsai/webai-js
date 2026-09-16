const CONFIG = {
  MODEL_ID: "mobilenet-v2-litert",
  MODEL_REVISION: "a847f8dd803c5471b44e57fed3be772c0d382214",
  MODEL_URL: "https://huggingface.co/litert-community/MobileNet-v2/resolve/a847f8dd803c5471b44e57fed3be772c0d382214/mobilenet_v2.tflite",
  LABEL_URL: "https://huggingface.co/datasets/huggingface/label-files/resolve/998629752167c60819e6295813eda1c2db248fd4/imagenet-1k-id2label.json",
  LITERT_MODULE_URL: "https://cdn.jsdelivr.net/npm/@litertjs/core@2.5.3/+esm",
  LITERT_WASM_URL: "https://cdn.jsdelivr.net/npm/@litertjs/core@2.5.3/wasm/",
  RUNTIME_VERSION: "2.5.3",
  PRECISIONS: {
    fp32: { size: 24024229, weightsSize: 14079072, runtimeBytes: 9911589, sharedModelBytes: 33568, supportedDevices: ["wasm", "webgpu"], modelKeys: ["mobilenet_v2.tflite"], dtype: "fp32" },
  },
};

const MANIFEST = {
  contractVersion: "1.0",
  model: {
    id: CONFIG.MODEL_ID,
    displayName: "MobileNet V2 — LiteRT Image Classification",
    provider: "PyTorch Vision / Google LiteRT Community",
    providerUrl: "https://pytorch.org/vision/",
    license: "BSD-3-Clause",
    licenseUrl: "https://github.com/pytorch/vision/blob/main/LICENSE",
    lastUpdated: "2025-01-30T00:00:00.000Z",
    sourceRepository: "pytorch/vision",
    repository: "litert-community/MobileNet-v2",
    artifactRevision: CONFIG.MODEL_REVISION,
    artifactLastUpdated: "2026-06-30T05:44:12.000Z",
    task: "image-classification",
    description: "MobileNet V2 classifies one image into 1,000 ImageNet categories locally in the browser using LiteRT.js.",
    intendedUses: ["Classifying the primary object in a photograph.", "Prototyping efficient client-side image classification."],
    limitations: ["Only ImageNet-1k categories are available.", "It classifies the whole image and does not return boxes, segmentation, or captions.", "Small, obscured, unusual, or out-of-distribution subjects can be misclassified.", "Do not use predictions for identity, medical, safety, or other high-impact decisions."],
  },
  runtime: { engine: { name: "LiteRT.js", version: CONFIG.RUNTIME_VERSION, workerBridge: "classic-worker-inside-module-worker" }, precisions: CONFIG.PRECISIONS },
  input: { description: "Provide exactly one browser blob URL containing a decodable image.", alternatives: [{ name: "image blob", fields: [{ name: "image_blob_url", type: "string(blob URL)", required: true, description: "Image fetched and decoded by the worker." }] }] },
  config: { model: [], generation: [{ name: "topK", type: "integer", default: 5, minimum: 1, maximum: 20, description: "Number of predictions returned." }] },
  output: { description: "Ranked ImageNet predictions plus LiteRT acceleration status.", fields: [{ name: "predictions", type: "Array<{ label: string, classIndex: integer, score: number }>", required: true }, { name: "fullyAccelerated", type: "boolean", required: true }], example: { predictions: [{ label: "golden retriever", classIndex: 207, score: 0.82 }], fullyAccelerated: true } },
  operations: ["checkModelSupports", "init", "download", "generate", "clearMemory"],
  benchmark: {
    report: "benchmarks/mobilenet-v2-litert.json",
    testedAt: "2026-09-08T17:40:57.000Z",
    fixture: { file: "modnet-portrait-pexels-5965592.jpg", url: "https://f005.backblazeb2.com/file/hubters-webai-public/assets/benchmarks/modnet-portrait-pexels-5965592.jpg", size: 78124, sha1: "38212739a4a4f0be25d84863cf991d7fb6c6d818", width: 1024, height: 683, license: "Pexels License" },
    environment: "Headless Chrome 151 on macOS; LiteRT.js 2.5.3",
    invalidatedBy: ["model artifact revision change", "LiteRT.js version change", "fixture or quality-gate change", "browser/backend behavior change"],
    results: [
      { precision: "fp32", device: "wasm", status: "pass", runtime: { loadMs: 6547.8, inferenceMs: 28, totalMs: 6575.8, grade: "excellent" }, quality: { grade: "pass", expectedTopClass: 869, actualTopClass: 869 }, fullyAccelerated: true },
      { precision: "fp32", device: "webgpu", status: "pass", runtime: { loadMs: 6382, inferenceMs: 22, totalMs: 6404, grade: "excellent" }, quality: { grade: "pass", expectedTopClass: 869, actualTopClass: 869, maximumTopFiveLogitDifference: 0.000011 }, fullyAccelerated: true },
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
        if (JSON.stringify(inputShape) !== JSON.stringify([1, 3, 224, 224])) throw new Error(`Unexpected input shape: ${JSON.stringify(inputShape)}`);
        if (outputShape.at(-1) !== 1000) throw new Error(`Unexpected output shape: ${JSON.stringify(outputShape)}`);
        fullyAccelerated = model.isFullyAccelerated;
        reply(id, "result", { fullyAccelerated, input: inputShape, output: outputShape });
      } else if (type === "download") {
        await fetchBytes(data.modelUrl, id);
        reply(id, "result", { status: "success" });
      } else if (type === "generate") {
        if (!model) throw new Error("Call init before generate.");
        const input = new api.Tensor(data.input, [1, 3, 224, 224]);
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
let labels;
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

async function getLabels() {
  if (!labels) {
    const response = await fetch(CONFIG.LABEL_URL);
    if (!response.ok) throw new Error(`Label download failed: ${response.status}`);
    labels = await response.json();
  }
  return labels;
}

async function preprocess(blobUrl) {
  const response = await fetch(blobUrl);
  if (!response.ok) throw new Error(`Image fetch failed: ${response.status}`);
  const bitmap = await createImageBitmap(await response.blob());
  const side = Math.min(bitmap.width, bitmap.height);
  const canvas = new OffscreenCanvas(224, 224);
  const context = canvas.getContext("2d", { willReadFrequently: true });
  context.drawImage(bitmap, (bitmap.width - side) / 2, (bitmap.height - side) / 2, side, side, 0, 0, 224, 224);
  bitmap.close();
  const rgba = context.getImageData(0, 0, 224, 224).data;
  const input = new Float32Array(3 * 224 * 224);
  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];
  const plane = 224 * 224;
  for (let pixel = 0; pixel < plane; pixel++) for (let channel = 0; channel < 3; channel++) input[channel * plane + pixel] = (rgba[pixel * 4 + channel] / 255 - mean[channel]) / std[channel];
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
        await Promise.all([getLabels(), callBridge("init", { moduleUrl: CONFIG.LITERT_MODULE_URL, wasmUrl: CONFIG.LITERT_WASM_URL, modelUrl: CONFIG.MODEL_URL, device: data.device }, (value) => respond(requestId, "downloadProgress", value))]);
        selectedPrecision = data.precision; selectedDevice = data.device; initialized = true;
        respond(requestId, type, { status: "success", precision: selectedPrecision, device: selectedDevice });
      } finally { initializing = false; }
    } else if (type === "download") {
      if (!CONFIG.PRECISIONS[data?.precision]) throw new Error(`Unsupported precision: ${data?.precision}`);
      await callBridge("download", { modelUrl: CONFIG.MODEL_URL }, (value) => respond(requestId, "downloadProgress", value));
      respond(requestId, type, { status: "success" });
    } else if (type === "generate") {
      if (!initialized) throw new Error("Call init before generate.");
      if (generating) throw new Error("Generation already in progress.");
      const userInput = data?.userInput;
      if (!userInput || Object.keys(userInput).filter((key) => userInput[key] != null).length !== 1 || typeof userInput.image_blob_url !== "string" || !userInput.image_blob_url.startsWith("blob:")) throw new Error("userInput must contain exactly one valid image_blob_url.");
      const topK = data?.generateConfig?.topK ?? 5;
      if (!Number.isInteger(topK) || topK < 1 || topK > 20) throw new Error("generateConfig.topK must be an integer from 1 through 20.");
      generating = true;
      try {
        const result = await callBridge("generate", { input: await preprocess(userInput.image_blob_url) });
        const predictions = result.scores.map((score, classIndex) => ({ label: labels[String(classIndex)] ?? `class_${classIndex}`, classIndex, score })).sort((a, b) => b.score - a.score).slice(0, topK);
        respond(requestId, "generated", { status: "success", result: { predictions, fullyAccelerated: result.fullyAccelerated } });
      } finally { generating = false; }
    } else if (type === "clearMemory") {
      await callBridge("clear", {}); initialized = false;
      respond(requestId, type, { status: "success" });
    } else throw new Error(`Unknown operation: ${type}`);
  } catch (error) { respond(requestId, "error", { message: error instanceof Error ? error.message : String(error), context: type }); }
});

self.postMessage({ type: "worker initialized", data: { success: true, workerVersion: "v4", modelId: CONFIG.MODEL_ID, runtimeName: "LiteRT.js", runtimeVersion: CONFIG.RUNTIME_VERSION } });
