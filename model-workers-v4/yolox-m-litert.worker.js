const CONFIG = {
  MODEL_ID: "yolox-m-litert",
  MODEL_REVISION: "f092a1457c4c39ca924f8815df2a42fe51bfb85e",
  MODEL_URLS: {
    fp16: "https://huggingface.co/litert-community/yolox-m-litert/resolve/f092a1457c4c39ca924f8815df2a42fe51bfb85e/yolox_m.tflite",
  },
  LITERT_MODULE_URL: "https://cdn.jsdelivr.net/npm/@litertjs/core@2.5.3/+esm",
  LITERT_WASM_URL: "https://cdn.jsdelivr.net/npm/@litertjs/core@2.5.3/wasm/",
  RUNTIME_VERSION: "2.5.3",
  PRECISIONS: {
    fp16: { size: 60953781, weightsSize: 51042192, runtimeBytes: 9911589, sharedModelBytes: 0, supportedDevices: ["wasm", "webgpu"], modelKeys: ["yolox_m.tflite"], dtype: "fp16" },
  },
};

const COCO_LABELS = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"];

const MANIFEST = {
  contractVersion: "1.0",
  model: { id: CONFIG.MODEL_ID, displayName: "YOLOX-M — LiteRT Object Detection", provider: "Megvii", providerUrl: "https://en.megvii.com/", license: "Apache-2.0", licenseUrl: "https://github.com/Megvii-BaseDetection/YOLOX/blob/main/LICENSE", lastUpdated: "2023-04-19T00:00:00.000Z", sourceRepository: "Megvii-BaseDetection/YOLOX", repository: "litert-community/yolox-m-litert", artifactRevision: CONFIG.MODEL_REVISION, artifactLastUpdated: "2026-09-06T12:55:52.000Z", task: "object-detection", description: "YOLOX-M detects and localizes 80 COCO object categories locally in the browser using a GPU-clean LiteRT graph.", intendedUses: ["Finding common COCO objects in photographs.", "Prototyping local object-detection interfaces."], limitations: ["Only the 80 COCO categories are recognized.", "Small, crowded, occluded, blurred, or unusual objects can be missed or mislabeled.", "Bounding boxes and confidence scores are estimates and must not drive safety-critical decisions."] },
  runtime: { engine: { name: "LiteRT.js", version: CONFIG.RUNTIME_VERSION, workerBridge: "classic-worker-inside-module-worker" }, precisions: CONFIG.PRECISIONS },
  input: { description: "Provide exactly one browser blob URL containing a decodable image.", alternatives: [{ name: "image blob", fields: [{ name: "image_blob_url", type: "string(blob URL)", required: true, description: "Image is letterboxed to 640×640, converted RGB→BGR, and kept in the 0–255 range." }] }] },
  config: { model: [], generation: [{ name: "scoreThreshold", type: "number", default: 0.35, minimum: 0, maximum: 1 }, { name: "iouThreshold", type: "number", default: 0.45, minimum: 0, maximum: 1 }, { name: "maxDetections", type: "integer", default: 20, minimum: 1, maximum: 100 }] },
  output: { description: "COCO detections with normalized boxes after per-class non-maximum suppression.", fields: [{ name: "detections", type: "Array<{ label: string, classIndex: integer, score: number, box: { xmin: number, ymin: number, xmax: number, ymax: number } }>", required: true }, { name: "fullyAccelerated", type: "boolean", required: true }], example: { detections: [{ label: "person", classIndex: 0, score: 0.9, box: { xmin: 0.1, ymin: 0.1, xmax: 0.8, ymax: 0.9 } }], fullyAccelerated: true } },
  operations: ["checkModelSupports", "init", "download", "generate", "clearMemory"],
  benchmark: { report: "benchmarks/yolox-m-litert.json", testedAt: "2026-09-08T18:10:00.000Z", fixture: { file: "modnet-portrait-pexels-5965592.jpg", url: "https://f005.backblazeb2.com/file/hubters-webai-public/assets/benchmarks/modnet-portrait-pexels-5965592.jpg", size: 78124, sha1: "38212739a4a4f0be25d84863cf991d7fb6c6d818", width: 1024, height: 683, license: "Pexels License" }, environment: "Headless Chrome 151 on macOS; LiteRT.js 2.5.3", invalidatedBy: ["model artifact revision change", "LiteRT.js version change", "fixture or quality-gate change", "browser/backend behavior change"], results: [{ precision: "fp16", device: "wasm", status: "pass", runtime: { loadMs: 10978.9, inferenceMs: 5773.8, totalMs: 16752.7, grade: "very-slow" }, quality: { grade: "reference", label: "person", score: 0.939385 }, fullyAccelerated: false }, { precision: "fp16", device: "webgpu", status: "pass", runtime: { loadMs: 11170.9, inferenceMs: 892.1, totalMs: 12063, grade: "slow" }, quality: { grade: "excellent", label: "person", scoreDifference: 0.000001, maximumBoxDifference: 0.000001 }, fullyAccelerated: true }] },
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
        if (JSON.stringify(inputShape) !== JSON.stringify([1, 640, 640, 3])) throw new Error(`Unexpected input shape: ${JSON.stringify(inputShape)}`);
        if (JSON.stringify(outputShape) !== JSON.stringify([1, 8400, 85])) throw new Error(`Unexpected output shape: ${JSON.stringify(outputShape)}`);
        fullyAccelerated = model.isFullyAccelerated;
        reply(id, "result", { fullyAccelerated, input: inputShape, output: outputShape });
      } else if (type === "download") {
        await fetchBytes(data.modelUrl, id);
        reply(id, "result", { status: "success" });
      } else if (type === "generate") {
        if (!model) throw new Error("Call init before generate.");
        const input = new api.Tensor(data.input, [1, 640, 640, 3]);
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
  const scale = Math.min(640 / bitmap.width, 640 / bitmap.height);
  const resizedWidth = Math.round(bitmap.width * scale), resizedHeight = Math.round(bitmap.height * scale);
  const canvas = new OffscreenCanvas(640, 640);
  const context = canvas.getContext("2d", { willReadFrequently: true });
  context.fillStyle = "rgb(114,114,114)"; context.fillRect(0, 0, 640, 640); context.drawImage(bitmap, 0, 0, resizedWidth, resizedHeight); bitmap.close();
  const rgba = context.getImageData(0, 0, 640, 640).data, input = new Float32Array(640 * 640 * 3);
  for (let pixel = 0; pixel < 640 * 640; pixel++) { input[pixel * 3] = rgba[pixel * 4 + 2]; input[pixel * 3 + 1] = rgba[pixel * 4 + 1]; input[pixel * 3 + 2] = rgba[pixel * 4]; }
  return { input, resizedWidth, resizedHeight };
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
      const scoreThreshold = data?.generateConfig?.scoreThreshold ?? 0.35, iouThreshold = data?.generateConfig?.iouThreshold ?? 0.45, maxDetections = data?.generateConfig?.maxDetections ?? 20;
      if (![scoreThreshold,iouThreshold].every(value=>Number.isFinite(value)&&value>=0&&value<=1)) throw new Error("scoreThreshold and iouThreshold must be from 0 through 1.");
      if (!Number.isInteger(maxDetections)||maxDetections<1||maxDetections>100) throw new Error("maxDetections must be an integer from 1 through 100.");
      generating = true;
      try {
        const prepared = await preprocess(userInput.image_blob_url);
        const result = await callBridge("generate", { input: prepared.input });
        const candidates=[];let anchor=0;
        for(const stride of [8,16,32]){const size=640/stride;for(let gy=0;gy<size;gy++)for(let gx=0;gx<size;gx++,anchor++){const offset=anchor*85,obj=result.scores[offset+4];let classIndex=0,classScore=0;for(let c=0;c<80;c++){const score=result.scores[offset+5+c];if(score>classScore){classScore=score;classIndex=c}}const score=obj*classScore;if(score<scoreThreshold)continue;const cx=(result.scores[offset]+gx)*stride,cy=(result.scores[offset+1]+gy)*stride,w=Math.exp(result.scores[offset+2])*stride,h=Math.exp(result.scores[offset+3])*stride;candidates.push({label:COCO_LABELS[classIndex],classIndex,score,box:{xmin:Math.max(0,(cx-w/2)/prepared.resizedWidth),ymin:Math.max(0,(cy-h/2)/prepared.resizedHeight),xmax:Math.min(1,(cx+w/2)/prepared.resizedWidth),ymax:Math.min(1,(cy+h/2)/prepared.resizedHeight)}})}}
        const iou=(a,b)=>{const x1=Math.max(a.xmin,b.xmin),y1=Math.max(a.ymin,b.ymin),x2=Math.min(a.xmax,b.xmax),y2=Math.min(a.ymax,b.ymax),intersection=Math.max(0,x2-x1)*Math.max(0,y2-y1),areaA=(a.xmax-a.xmin)*(a.ymax-a.ymin),areaB=(b.xmax-b.xmin)*(b.ymax-b.ymin);return intersection/(areaA+areaB-intersection||1)};
        const detections=[];for(const candidate of candidates.sort((a,b)=>b.score-a.score)){if(detections.length>=maxDetections)break;if(detections.some(kept=>kept.classIndex===candidate.classIndex&&iou(kept.box,candidate.box)>iouThreshold))continue;detections.push(candidate)}
        respond(requestId, "generated", { status: "success", result: { detections, fullyAccelerated: result.fullyAccelerated } });
      } finally { generating = false; }
    } else if (type === "clearMemory") {
      await callBridge("clear", {}); initialized = false;
      respond(requestId, type, { status: "success" });
    } else throw new Error(`Unknown operation: ${type}`);
  } catch (error) { respond(requestId, "error", { message: error instanceof Error ? error.message : String(error), context: type }); }
});

self.postMessage({ type: "worker initialized", data: { success: true, workerVersion: "v4", modelId: CONFIG.MODEL_ID, runtimeName: "LiteRT.js", runtimeVersion: CONFIG.RUNTIME_VERSION } });
