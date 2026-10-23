const CONFIG = {
  MODEL_ID: "mediapipe-selfie-segmenter",
  MODEL_URL: "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite",
  MODEL_VERSION: "selfie-segmenter-float16-generation-1683436453600523",
  MODEL_BYTES: 249537,
  RUNTIME_VERSION: "1.0.1",
  BUNDLE_URL: "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/vision_bundle.js",
  WASM_URL: "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm",
  RUNTIME_BYTES: 12235800,
  PRECISIONS: {
    fp16: { size: 12485337, weightsSize: 249537, runtimeBytes: 12235800, supportedDevices: ["wasm"], modelKeys: ["selfie_segmenter.tflite"], modelQuantization: "float16", runtimePrecision: "float32" },
  },
};

const MANIFEST = {
  contractVersion: "1.0",
  model: {
    id: CONFIG.MODEL_ID,
    displayName: "MediaPipe Selfie Segmenter",
    provider: "Google MediaPipe",
    providerUrl: "https://ai.google.dev/edge/mediapipe",
    license: "Apache-2.0",
    licenseUrl: "https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE",
    lastUpdated: "2023-05-07T05:14:13.000Z",
    sourceRepository: "google-ai-edge/mediapipe",
    repository: "mediapipe-models/image_segmenter/selfie_segmenter",
    artifactRevision: CONFIG.MODEL_VERSION,
    artifactLastUpdated: "2023-05-07T05:14:13.000Z",
    task: "portrait-segmentation",
    description: "Segments the prominent person from the background in one image locally in the browser with Google's official MediaPipe Selfie Segmenter and returns a transparent PNG plus an alpha-mask PNG.",
    intendedUses: ["Removing or replacing a webcam background.", "Applying portrait-only visual effects to selfies and video-call frames."],
    limitations: ["Segments the whole prominent person, not only the face.", "Fine hair, motion blur, occlusion, poor lighting, and multiple people can produce inaccurate edges.", "The output is a visual estimate and must not be used for identification, safety-critical decisions, or biometric analysis."],
  },
  runtime: { engine: { name: "MediaPipe Tasks Vision", version: CONFIG.RUNTIME_VERSION, backend: "WASM CPU", workerBridge: "classic-worker-inside-module-worker" }, precisions: CONFIG.PRECISIONS },
  input: { description: "Provide either one browser blob URL or one raw RGBA frame.", alternatives: [{ name: "image blob", fields: [{ name: "image_blob_url", type: "string(blob URL)", required: true, description: "Image fetched and decoded inside the worker." }] }, { name: "raw RGBA frame", fields: [{ name: "image_data", type: "{ data: Uint8ClampedArray, width: integer, height: integer }", required: true, description: "Realtime path that avoids image encoding and decoding." }] }] },
  config: { model: [], generation: [{ name: "useImageConversion", type: "boolean", default: true, description: "True uses the backward-compatible blob/PNG path; false requires raw RGBA image_data and defaults to a transferable alpha mask." }, { name: "threshold", type: "number|null", default: null, minimum: 0, maximum: 1, description: "Optional hard alpha threshold; null preserves the soft confidence mask." }, { name: "outputFormat", type: "'png'|'alpha-mask'", default: "png when useImageConversion is true; alpha-mask when false", description: "Optional explicit output override." }] },
  output: { description: "WebAI wraps either the PNG result or realtime alpha-mask result below as { result, runtime: { durationMs } }. PNG blob URLs stay valid until the next generation or clearMemory call.", fields: [{ name: "result", type: "string(blob URL)", required: false, description: "PNG mode: transparent source image." }, { name: "result_mask", type: "string(blob URL)", required: false, description: "PNG mode: grayscale person mask." }, { name: "alpha", type: "Uint8ClampedArray", required: false, description: "Alpha-mask mode: transferable person-confidence bytes at maskWidth × maskHeight." }, { name: "maskWidth", type: "integer", required: false, description: "Alpha-mask mode: raw mask width." }, { name: "maskHeight", type: "integer", required: false, description: "Alpha-mask mode: raw mask height." }, { name: "width", type: "integer", required: true, description: "Source-image width in pixels." }, { name: "height", type: "integer", required: true, description: "Source-image height in pixels." }, { name: "device", type: "'wasm'", required: true, description: "Measured MediaPipe execution backend." }, { name: "precision", type: "'fp16'", required: true, description: "Quantization of the selected TFLite artifact." }], example: { alpha: "Uint8ClampedArray(65536)", maskWidth: 256, maskHeight: 256, width: 384, height: 288, device: "wasm", precision: "fp16" } },
  operations: ["checkModelSupports", "init", "download", "generate", "clearMemory"],
  benchmark: {
    report: "benchmarks/mediapipe-selfie-segmenter.json",
    testedAt: "2026-09-16T08:40:05.446Z",
    fixture: { file: "modnet-portrait-pexels-5965592.jpg", localPath: "/test-images/modnet-portrait-pexels-5965592.jpg", url: "https://f005.backblazeb2.com/file/hubters-webai-public/assets/benchmarks/modnet-portrait-pexels-5965592.jpg", size: 78124, sha1: "38212739a4a4f0be25d84863cf991d7fb6c6d818", width: 1024, height: 683, license: "Pexels License", licenseUrl: "https://www.pexels.com/license/" },
    environment: "Headless Chrome 151 on macOS; MediaPipe Tasks Vision 1.0.1 CPU/WASM",
    invalidatedBy: ["model artifact generation change", "MediaPipe Tasks Vision version change", "fixture or quality-gate change", "browser/backend behavior change"],
    results: [{ precision: "fp16", device: "wasm", status: "pass", runtime: { loadMs: 4605.1, inferenceMs: 69.6, totalMs: 4944.4, megapixelsPerSecond: 10.049, grade: "good" }, quality: { passed: true, grade: "excellent", meanAbsoluteError: 0, intersectionOverUnion: 1, foregroundRatio: 0.251478 }, cacheCleanup: { supported: true, deleted: 1 } }, { precision: "fp16", device: "webgpu", status: "fail", error: "Unsupported device: webgpu (MediaPipe worker uses the measured CPU/WASM delegate)", runtime: { loadMs: null, inferenceMs: null, totalMs: 3.6, grade: "failed" }, cacheCleanup: { supported: true, deleted: 0 } }],
  },
};

function mediaPipeBridge() {
  let segmenter;
  const originalConsoleError = console.error;
  console.error = (...args) => {
    if (!args.map(String).join(" ").includes("INFO: Created TensorFlow Lite XNNPACK delegate for CPU")) originalConsoleError(...args);
  };
  const reply = (id, type, data, transfer = []) => self.postMessage({ id, type, data }, transfer);
  self.onmessage = async ({ data: message }) => {
    const { id, type, data } = message;
    try {
      if (type === "init") {
        segmenter?.close();
        if (!self.Vision) importScripts(data.bundleUrl);
        const fileset = await Vision.FilesetResolver.forVisionTasks(data.wasmUrl);
        segmenter = await Vision.ImageSegmenter.createFromOptions(fileset, { baseOptions: { modelAssetBuffer: new Uint8Array(data.model), delegate: "CPU" }, runningMode: "IMAGE", outputConfidenceMasks: true, outputCategoryMask: false });
        reply(id, "result", { status: "success" });
      } else if (type === "generate") {
        if (!segmenter) throw new Error("Call init before generate.");
        const pixels = new Uint8ClampedArray(data.pixels);
        const result = segmenter.segment(new ImageData(pixels, data.width, data.height));
        try {
          const personMask = result.confidenceMasks?.at(-1);
          if (!personMask) throw new Error("MediaPipe returned no person confidence mask.");
          const confidence = personMask.getAsFloat32Array();
          const alpha = new Uint8ClampedArray(confidence.length);
          for (let i = 0; i < confidence.length; i++) alpha[i] = Math.round(Math.max(0, Math.min(1, confidence[i])) * 255);
          reply(id, "result", { alpha: alpha.buffer, maskWidth: personMask.width, maskHeight: personMask.height }, [alpha.buffer]);
        } finally { result.close(); }
      } else if (type === "clear") {
        segmenter?.close(); segmenter = null;
        reply(id, "result", { status: "success" });
      } else throw new Error(`Unknown bridge operation: ${type}`);
    } catch (error) { reply(id, "error", { message: error instanceof Error ? error.message : String(error) }); }
  };
}

const bridgeUrl = URL.createObjectURL(new Blob([`(${mediaPipeBridge.toString()})();`], { type: "text/javascript" }));
const bridge = new Worker(bridgeUrl);
URL.revokeObjectURL(bridgeUrl);
const pending = new Map();
let initialized = false, initializing = false, generating = false, modelBytes, selectedPrecision, selectedDevice;
let generatedUrls = [];

bridge.addEventListener("message", ({ data }) => {
  const operation = pending.get(data.id);
  if (!operation) return;
  pending.delete(data.id);
  if (data.type === "error") operation.reject(new Error(data.data.message)); else operation.resolve(data.data);
});
function callBridge(type, data, transfer = []) { const id = crypto.randomUUID(); return new Promise((resolve, reject) => { pending.set(id, { resolve, reject }); bridge.postMessage({ id, type, data }, transfer); }); }
function respond(requestId, type, data, transfer = []) { self.postMessage({ requestId, type, data }, transfer); }

async function fetchModel(progress) {
  if (modelBytes) return modelBytes;
  const cache = await caches.open("webai-mediapipe-model-cache-v1");
  const cached = await cache.match(CONFIG.MODEL_URL);
  if (cached) { modelBytes = new Uint8Array(await cached.arrayBuffer()); return modelBytes; }
  const response = await fetch(CONFIG.MODEL_URL);
  if (!response.ok) throw new Error(`Model download failed: ${response.status} ${response.statusText}`);
  const cacheCopy = response.clone(), total = Number(response.headers.get("content-length")) || CONFIG.MODEL_BYTES, reader = response.body.getReader(), chunks = [];
  let loaded = 0;
  while (true) { const { done, value } = await reader.read(); if (done) break; chunks.push(value); loaded += value.byteLength; progress?.({ status: "progress", loaded, total, progress: total ? loaded / total * 100 : 0 }); }
  await cache.put(CONFIG.MODEL_URL, cacheCopy);
  modelBytes = new Uint8Array(loaded); let offset = 0;
  for (const chunk of chunks) { modelBytes.set(chunk, offset); offset += chunk.byteLength; }
  return modelBytes;
}

async function decodeImage(blobUrl) {
  const response = await fetch(blobUrl);
  if (!response.ok) throw new Error(`Image fetch failed: ${response.status}`);
  const bitmap = await createImageBitmap(await response.blob());
  try { const canvas = new OffscreenCanvas(bitmap.width, bitmap.height), context = canvas.getContext("2d", { willReadFrequently: true }); context.drawImage(bitmap, 0, 0); const image = context.getImageData(0, 0, bitmap.width, bitmap.height); return { pixels: image.data, width: bitmap.width, height: bitmap.height }; }
  finally { bitmap.close(); }
}

async function renderOutput(image, bridgeResult, threshold) {
  const maskCanvas = new OffscreenCanvas(bridgeResult.maskWidth, bridgeResult.maskHeight), maskContext = maskCanvas.getContext("2d"), maskImage = maskContext.createImageData(bridgeResult.maskWidth, bridgeResult.maskHeight), values = new Uint8ClampedArray(bridgeResult.alpha);
  for (let i = 0; i < values.length; i++) { const alpha = threshold == null ? values[i] : (values[i] / 255 >= threshold ? 255 : 0); const p = i * 4; maskImage.data[p] = 255; maskImage.data[p + 1] = 255; maskImage.data[p + 2] = 255; maskImage.data[p + 3] = alpha; }
  maskContext.putImageData(maskImage, 0, 0);
  const output = new OffscreenCanvas(image.width, image.height), context = output.getContext("2d");
  context.putImageData(new ImageData(image.pixels, image.width, image.height), 0, 0); context.globalCompositeOperation = "destination-in"; context.drawImage(maskCanvas, 0, 0, image.width, image.height);
  const maskOutput = new OffscreenCanvas(image.width, image.height), maskOutputContext = maskOutput.getContext("2d"); maskOutputContext.fillStyle = "black"; maskOutputContext.fillRect(0, 0, image.width, image.height); maskOutputContext.drawImage(maskCanvas, 0, 0, image.width, image.height);
  const [resultBlob, maskBlob] = await Promise.all([output.convertToBlob({ type: "image/png" }), maskOutput.convertToBlob({ type: "image/png" })]);
  for (const url of generatedUrls) URL.revokeObjectURL(url);
  generatedUrls = [URL.createObjectURL(resultBlob), URL.createObjectURL(maskBlob)];
  return { result: generatedUrls[0], result_mask: generatedUrls[1], mimeType: "image/png", size: resultBlob.size, maskSize: maskBlob.size, width: image.width, height: image.height, device: selectedDevice, precision: selectedPrecision };
}

self.addEventListener("message", async ({ data: message = {} }) => {
  const { requestId, type, data } = message;
  try {
    if (type === "checkModelSupports") respond(requestId, type, { manifest: MANIFEST, supportedModes: ["webai"], supportedPrecisions: Object.keys(CONFIG.PRECISIONS), supportedPrecisionsDevicesMap: CONFIG.PRECISIONS, doesSupportStreamGeneration: false, externalInterrupt: true, workerVersion: "v4", transformersJsVersion: null, runtimeName: "MediaPipe Tasks Vision", runtimeVersion: CONFIG.RUNTIME_VERSION, cacheModelId: MANIFEST.model.repository });
    else if (type === "init") {
      if (initializing || generating) throw new Error("Worker is busy.");
      const precision = data?.precision ?? "fp16", device = data?.device ?? "wasm", entry = CONFIG.PRECISIONS[precision];
      if (!entry) throw new Error(`Unsupported precision: ${precision}`); if (!entry.supportedDevices.includes(device)) throw new Error(`Unsupported device: ${device}`);
      initializing = true; initialized = false;
      try { const bytes = await fetchModel((value) => respond(requestId, "downloadProgress", value)), copy = bytes.slice(); await callBridge("init", { bundleUrl: CONFIG.BUNDLE_URL, wasmUrl: CONFIG.WASM_URL, model: copy.buffer }, [copy.buffer]); selectedPrecision = precision; selectedDevice = device; initialized = true; respond(requestId, type, { status: "success", precision, device }); }
      finally { initializing = false; }
    } else if (type === "download") { const precision = data?.precision ?? "fp16"; if (!CONFIG.PRECISIONS[precision]) throw new Error(`Unsupported precision: ${precision}`); await fetchModel((value) => respond(requestId, "downloadProgress", value)); respond(requestId, type, { status: "success" }); }
    else if (type === "generate") {
      if (!initialized) throw new Error("Call init before generate."); if (generating) throw new Error("Generation already in progress.");
      const userInput = data?.userInput, threshold = data?.generateConfig?.threshold ?? null, useImageConversion = data?.generateConfig?.useImageConversion ?? true, outputFormat = data?.generateConfig?.outputFormat ?? (useImageConversion ? "png" : "alpha-mask");
      if (typeof useImageConversion !== "boolean") throw new Error("generateConfig.useImageConversion must be boolean.");
      if (!userInput || Object.keys(userInput).filter((key) => userInput[key] != null).length !== 1) throw new Error("userInput must contain exactly one supported image input.");
      const raw = userInput.image_data;
      const isBlob = typeof userInput.image_blob_url === "string" && userInput.image_blob_url.startsWith("blob:");
      const isRaw = raw && (raw.data instanceof Uint8ClampedArray || raw.data instanceof Uint8Array) && Number.isInteger(raw.width) && raw.width > 0 && Number.isInteger(raw.height) && raw.height > 0 && raw.data.byteLength === raw.width * raw.height * 4;
      if (useImageConversion ? !isBlob : !isRaw) throw new Error(useImageConversion ? "useImageConversion=true requires image_blob_url." : "useImageConversion=false requires raw image_data.");
      if (threshold !== null && (typeof threshold !== "number" || threshold < 0 || threshold > 1)) throw new Error("generateConfig.threshold must be null or a number from 0 through 1.");
      if (outputFormat !== "png" && outputFormat !== "alpha-mask") throw new Error("generateConfig.outputFormat must be png or alpha-mask.");
      generating = true;
      try {
        const image = isRaw ? { pixels: new Uint8ClampedArray(raw.data.buffer, raw.data.byteOffset, raw.data.byteLength), width: raw.width, height: raw.height } : await decodeImage(userInput.image_blob_url);
        const canTransferDirectly = outputFormat === "alpha-mask" && image.pixels.byteOffset === 0 && image.pixels.byteLength === image.pixels.buffer.byteLength;
        const pixels = canTransferDirectly ? image.pixels : image.pixels.slice();
        const bridgeResult = await callBridge("generate", { pixels: pixels.buffer, width: image.width, height: image.height }, [pixels.buffer]);
        if (outputFormat === "alpha-mask") {
          const alpha = new Uint8ClampedArray(bridgeResult.alpha);
          if (threshold !== null) for (let i = 0; i < alpha.length; i++) alpha[i] = alpha[i] / 255 >= threshold ? 255 : 0;
          respond(requestId, "generated", { status: "success", result: { alpha, maskWidth: bridgeResult.maskWidth, maskHeight: bridgeResult.maskHeight, width: image.width, height: image.height, device: selectedDevice, precision: selectedPrecision } }, [alpha.buffer]);
        } else {
          const result = await renderOutput(image, bridgeResult, threshold);
          respond(requestId, "generated", { status: "success", result });
        }
      }
      finally { generating = false; }
    } else if (type === "clearMemory") { await callBridge("clear", {}); for (const url of generatedUrls) URL.revokeObjectURL(url); generatedUrls = []; modelBytes = null; initialized = false; selectedPrecision = null; selectedDevice = null; respond(requestId, type, { status: "success" }); }
    else throw new Error(`Unknown operation: ${type}`);
  } catch (error) { respond(requestId, "error", { message: error instanceof Error ? error.message : String(error), context: type }); }
});

self.postMessage({ type: "worker initialized", data: { success: true, workerVersion: "v4", modelId: CONFIG.MODEL_ID, runtimeName: "MediaPipe Tasks Vision", runtimeVersion: CONFIG.RUNTIME_VERSION } });
