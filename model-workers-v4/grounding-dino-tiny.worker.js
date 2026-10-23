import { env, pipeline, RawImage } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0";

const ID = "grounding-dino-tiny";
const REPOSITORY = "onnx-community/grounding-dino-tiny-ONNX";
const REVISION = "ff690b0a8050566c290287545bd059350f3e9096";
const SHARED_BYTES = 946941;
const makePrecision = (weightsSize, dtype, suffix) => ({
  weightsSize,
  size: weightsSize + SHARED_BYTES,
  dtype,
  modelKeys: [`model${suffix}.onnx`],
  candidateDevices: ["wasm", "webgpu"],
});
const PRECISIONS = {
  fp32: makePrecision(718761381, "fp32", ""),
  fp16: makePrecision(360393267, "fp16", "_fp16"),
  int8: makePrecision(203824481, "int8", "_int8"),
  q8: makePrecision(203824675, "q8", "_quantized"),
  uint8: makePrecision(203824675, "uint8", "_uint8"),
  q4: makePrecision(227229745, "q4", "_q4"),
  q4f16: makePrecision(151069879, "q4f16", "_q4f16"),
  bnb4: makePrecision(218129274, "bnb4", "_bnb4"),
};

// Populated only from browser combinations that pass inference and the quality gate.
const SUPPORTED = {"fp32":{"weightsSize":718761381,"size":719708322,"dtype":"fp32","modelKeys":["model.onnx"],"candidateDevices":["wasm","webgpu"],"supportedDevices":["wasm","webgpu"]},"int8":{"weightsSize":203824481,"size":204771422,"dtype":"int8","modelKeys":["model_int8.onnx"],"candidateDevices":["wasm","webgpu"],"supportedDevices":["wasm","webgpu"]},"q8":{"weightsSize":203824675,"size":204771616,"dtype":"q8","modelKeys":["model_quantized.onnx"],"candidateDevices":["wasm","webgpu"],"supportedDevices":["wasm","webgpu"]},"uint8":{"weightsSize":203824675,"size":204771616,"dtype":"uint8","modelKeys":["model_uint8.onnx"],"candidateDevices":["wasm","webgpu"],"supportedDevices":["wasm","webgpu"]},"q4":{"weightsSize":227229745,"size":228176686,"dtype":"q4","modelKeys":["model_q4.onnx"],"candidateDevices":["wasm","webgpu"],"supportedDevices":["wasm","webgpu"]},"bnb4":{"weightsSize":218129274,"size":219076215,"dtype":"bnb4","modelKeys":["model_bnb4.onnx"],"candidateDevices":["wasm","webgpu"],"supportedDevices":["wasm","webgpu"]}};
const ACTIVE_PRECISIONS = SUPPORTED;

const BENCHMARK_RESULTS = [{"precision":"fp32","device":"wasm","status":"pass","runtime":{"loadMs":60710.6,"inferenceMs":13371.6,"totalMs":74231,"grade":"very-slow","imagesPerSecond":0.0748},"quality":{"valid":true,"bestIoU":0.7976,"falsePositiveCount":0,"grade":"grounded-detection-pass"},"cleanup":{"modelDisposed":true,"cacheClearedBeforeCase":true,"cacheClearedAfterCase":true,"deletedEntries":5}},{"precision":"fp32","device":"webgpu","status":"pass","runtime":{"loadMs":68605.4,"inferenceMs":5013.1,"totalMs":73757.1,"grade":"very-slow","imagesPerSecond":0.1995},"quality":{"valid":true,"bestIoU":0.7976,"falsePositiveCount":0,"grade":"grounded-detection-pass"},"cleanup":{"modelDisposed":true,"cacheClearedBeforeCase":true,"cacheClearedAfterCase":true,"deletedEntries":5}},{"precision":"fp16","device":"wasm","status":"fail","runtime":{"loadMs":null,"inferenceMs":null,"totalMs":36948,"grade":"failed"},"failure":"onnx-session-error","cleanup":{"modelDisposed":true,"cacheClearedBeforeCase":true,"cacheClearedAfterCase":true,"deletedEntries":5}},{"precision":"fp16","device":"webgpu","status":"fail","runtime":{"loadMs":35093,"inferenceMs":4141.3,"totalMs":39341.5,"grade":"slow","imagesPerSecond":0.2415},"quality":{"valid":false,"bestIoU":0,"falsePositiveCount":0,"grade":"failed"},"failure":"quality-invalid-output","cleanup":{"modelDisposed":true,"cacheClearedBeforeCase":true,"cacheClearedAfterCase":true,"deletedEntries":5}},{"precision":"int8","device":"wasm","status":"pass","runtime":{"loadMs":15430.1,"inferenceMs":13308.8,"totalMs":28870.4,"grade":"very-slow","imagesPerSecond":0.0751},"quality":{"valid":true,"bestIoU":0.7914,"falsePositiveCount":0,"grade":"grounded-detection-pass"},"cleanup":{"modelDisposed":true,"cacheClearedBeforeCase":true,"cacheClearedAfterCase":true,"deletedEntries":6}},{"precision":"int8","device":"webgpu","status":"pass","runtime":{"loadMs":21755.5,"inferenceMs":14438,"totalMs":36323.8,"grade":"very-slow","imagesPerSecond":0.0693},"quality":{"valid":true,"bestIoU":0.7865,"falsePositiveCount":0,"grade":"grounded-detection-pass"},"cleanup":{"modelDisposed":true,"cacheClearedBeforeCase":true,"cacheClearedAfterCase":true,"deletedEntries":6}},{"precision":"q8","device":"wasm","status":"pass","runtime":{"loadMs":17387.4,"inferenceMs":13376,"totalMs":30906.4,"grade":"very-slow","imagesPerSecond":0.0748},"quality":{"valid":true,"bestIoU":0.8163,"falsePositiveCount":0,"grade":"grounded-detection-pass"},"cleanup":{"modelDisposed":true,"cacheClearedBeforeCase":true,"cacheClearedAfterCase":true,"deletedEntries":6}},{"precision":"q8","device":"webgpu","status":"pass","runtime":{"loadMs":16102.3,"inferenceMs":14540.5,"totalMs":31065.7,"grade":"very-slow","imagesPerSecond":0.0688},"quality":{"valid":true,"bestIoU":0.8163,"falsePositiveCount":0,"grade":"grounded-detection-pass"},"cleanup":{"modelDisposed":true,"cacheClearedBeforeCase":true,"cacheClearedAfterCase":true,"deletedEntries":6}},{"precision":"uint8","device":"wasm","status":"pass","runtime":{"loadMs":21907.2,"inferenceMs":13459.6,"totalMs":35490.7,"grade":"very-slow","imagesPerSecond":0.0743},"quality":{"valid":true,"bestIoU":0.8163,"falsePositiveCount":0,"grade":"grounded-detection-pass"},"cleanup":{"modelDisposed":true,"cacheClearedBeforeCase":true,"cacheClearedAfterCase":true,"deletedEntries":6}},{"precision":"uint8","device":"webgpu","status":"pass","runtime":{"loadMs":23051.7,"inferenceMs":14656.1,"totalMs":37966.6,"grade":"very-slow","imagesPerSecond":0.0682},"quality":{"valid":true,"bestIoU":0.8163,"falsePositiveCount":0,"grade":"grounded-detection-pass"},"cleanup":{"modelDisposed":true,"cacheClearedBeforeCase":true,"cacheClearedAfterCase":true,"deletedEntries":6}},{"precision":"q4","device":"wasm","status":"pass","runtime":{"loadMs":19242,"inferenceMs":13201.8,"totalMs":32558.7,"grade":"very-slow","imagesPerSecond":0.0757},"quality":{"valid":true,"bestIoU":0.783,"falsePositiveCount":0,"grade":"grounded-detection-pass"},"cleanup":{"modelDisposed":true,"cacheClearedBeforeCase":true,"cacheClearedAfterCase":true,"deletedEntries":6}},{"precision":"q4","device":"webgpu","status":"pass","runtime":{"loadMs":23187.2,"inferenceMs":2605.2,"totalMs":25899.6,"grade":"slow","imagesPerSecond":0.3838},"quality":{"valid":true,"bestIoU":0.783,"falsePositiveCount":0,"grade":"grounded-detection-pass"},"cleanup":{"modelDisposed":true,"cacheClearedBeforeCase":true,"cacheClearedAfterCase":true,"deletedEntries":6}},{"precision":"q4f16","device":"wasm","status":"fail","runtime":{"loadMs":null,"inferenceMs":null,"totalMs":20978.7,"grade":"failed"},"failure":"onnx-session-error","cleanup":{"modelDisposed":true,"cacheClearedBeforeCase":true,"cacheClearedAfterCase":true,"deletedEntries":6}},{"precision":"q4f16","device":"webgpu","status":"fail","runtime":{"loadMs":17378.5,"inferenceMs":1608.8,"totalMs":19184.6,"grade":"slow","imagesPerSecond":0.6216},"quality":{"valid":false,"bestIoU":0,"falsePositiveCount":0,"grade":"failed"},"failure":"quality-invalid-output","cleanup":{"modelDisposed":true,"cacheClearedBeforeCase":true,"cacheClearedAfterCase":true,"deletedEntries":6}},{"precision":"bnb4","device":"wasm","status":"pass","runtime":{"loadMs":31170.3,"inferenceMs":12612,"totalMs":43886.9,"grade":"very-slow","imagesPerSecond":0.0793},"quality":{"valid":true,"bestIoU":0.7861,"falsePositiveCount":0,"grade":"grounded-detection-pass"},"cleanup":{"modelDisposed":true,"cacheClearedBeforeCase":true,"cacheClearedAfterCase":true,"deletedEntries":6}},{"precision":"bnb4","device":"webgpu","status":"pass","runtime":{"loadMs":28194.8,"inferenceMs":13127.6,"totalMs":41459.2,"grade":"very-slow","imagesPerSecond":0.0762},"quality":{"valid":true,"bestIoU":0.7861,"falsePositiveCount":0,"grade":"grounded-detection-pass"},"cleanup":{"modelDisposed":true,"cacheClearedBeforeCase":true,"cacheClearedAfterCase":true,"deletedEntries":6}}];

const MANIFEST = {
  contractVersion: "1.0",
  model: {
    id: ID,
    displayName: "Grounding DINO Tiny — Open-Vocabulary Detection",
    provider: "IDEA Research",
    providerUrl: "https://github.com/IDEA-Research/GroundingDINO",
    license: "Apache-2.0",
    licenseUrl: "https://www.apache.org/licenses/LICENSE-2.0.txt",
    lastUpdated: "2024-05-12T09:03:39.000Z",
    sourceRevision: "a2bb814dd30d776dcf7e30523b00659f4f141c71",
    sourceRepository: "IDEA-Research/grounding-dino-tiny",
    repository: REPOSITORY,
    artifactRevision: REVISION,
    artifactLastUpdated: "2025-01-05T18:09:57.000Z",
    task: "zero-shot-object-detection",
    description: "A compact Grounding DINO open-vocabulary detector that runs in the browser and locates objects described by user-provided text labels, returning confidence scores and pixel-coordinate bounding boxes.",
    intendedUses: [
      "Prototype prompt-conditioned object detection for images.",
      "Locate uncommon object categories that are not present in a fixed detector label list.",
      "Build browser image annotation, search, and accessibility experiments.",
    ],
    limitations: [
      "Detection quality depends strongly on label wording; short lowercase noun phrases usually work better than abstract or ambiguous prompts.",
      "Small, occluded, blurred, unusual, or densely packed objects may be missed or assigned inaccurate boxes.",
      "Scores are not calibrated probabilities, and different label sets can change the returned detections.",
      "The model may reflect biases in its training data and must not be used alone for surveillance, identity, safety-critical, or other high-impact decisions.",
    ],
  },
  runtime: { engine: { name: "Transformers.js", version: "4.2.0" }, precisions: SUPPORTED },
  input: {
    description: "One browser-readable image URL and 1–32 object descriptions to search for.",
    alternatives: [{
      name: "image and candidate labels",
      fields: [
        { name: "image", type: "string(URL)", required: true, minLength: 1, format: "url", description: "HTTP(S), data, or blob URL readable by this worker." },
        { name: "candidate_labels", type: "string[]", required: true, minItems: 1, maxItems: 32, items: { type: "string", minLength: 1, maxLength: 100 }, description: "Concrete object descriptions. The worker trims, lowercases, and adds terminal periods as required by Grounding DINO." },
      ],
    }],
  },
  config: {
    model: [{ name: "threshold", type: "number", default: 0.3, minimum: 0, maximum: 1, description: "Minimum detection score returned by post-processing." }],
    generation: [],
  },
  output: {
    description: "Prompt-matched detections with scores and pixel-coordinate boxes, plus the original image dimensions.",
    fields: [
      { name: "detections", type: "Array<{label:string,score:number,box:{xmin:number,ymin:number,xmax:number,ymax:number}}>", required: true },
      { name: "width", type: "integer", required: true },
      { name: "height", type: "integer", required: true },
    ],
    example: { detections: [{ label: "a person", score: 0.71, box: { xmin: 230, ymin: 90, xmax: 740, ymax: 683 } }], width: 1024, height: 683 },
  },
  operations: ["checkModelSupports", "init", "download", "generate", "clearMemory"],
  benchmark: {
    report: "benchmarks/grounding-dino-tiny.json",
    testedAt: "2026-10-23T13:29:12.152Z",
    fixture: {
      file: "modnet-portrait-pexels-5965592.jpg",
      url: "https://f005.backblazeb2.com/file/hubters-webai-public/assets/benchmarks/modnet-portrait-pexels-5965592.jpg",
      size: 78124,
      sha1: "38212739a4a4f0be25d84863cf991d7fb6c6d818",
      width: 1024,
      height: 683,
      source: "https://www.pexels.com/photo/woman-wearing-pink-turtleneck-sweater-5965592/",
      license: "Pexels License",
      licenseUrl: "https://www.pexels.com/license/",
    },
    candidates: PRECISIONS,
    qualityGate: "For prompts 'a person' and 'a dog', output must contain a finite 'a person' detection with score >= 0.3 and IoU >= 0.4 against the checked-in subject box, return no 'a dog' false positive above threshold, and contain only valid boxes.",
    environment: "Isolated headless Chrome on macOS; Transformers.js 4.2.0",
    invalidatedBy: ["artifact revision change", "runtime version change", "fixture or quality-gate change", "browser/backend behavior change"],
    results: BENCHMARK_RESULTS,
  },
};

const state = { initializing: false, initialized: false, generating: false };
let detector = null;
const send = (requestId, type, data) => self.postMessage({ requestId, type, data });
const capabilities = {
  supportedModes: ["webai"],
  supportedPrecisions: Object.keys(SUPPORTED),
  supportedPrecisionsDevicesMap: SUPPORTED,
  doesSupportStreamGeneration: false,
  externalInterrupt: true,
  workerVersion: "v4",
  transformersJsVersion: "4.2.0",
  cacheModelId: REPOSITORY,
};

function configure(workerConfig = {}) {
  env.allowRemoteModels = workerConfig.allowRemoteModels ?? true;
  env.allowLocalModels = workerConfig.allowLocalModels ?? false;
  env.useBrowserCache = workerConfig.useBrowserCache ?? true;
  if (workerConfig.remoteHost) env.remoteHost = workerConfig.remoteHost;
  if (workerConfig.remotePathTemplate) env.remotePathTemplate = workerConfig.remotePathTemplate;
  if (workerConfig.localModelPath) env.localModelPath = workerConfig.localModelPath;
}

async function clear() {
  if (detector?.dispose) await detector.dispose();
  detector = null;
  state.initialized = false;
}

async function load(data, requestId) {
  const precision = data?.precision;
  const device = data?.device;
  const specification = ACTIVE_PRECISIONS[precision];
  if (!specification) throw Error(`Unsupported precision: ${precision}`);
  const devices = specification.supportedDevices ?? specification.candidateDevices;
  if (!devices.includes(device)) throw Error(`Precision ${precision} is only supported on: ${devices.join(", ")}`);
  if (device === "webgpu" && !self.navigator?.gpu) throw Error("WebGPU is unavailable");
  await clear();
  detector = await pipeline("zero-shot-object-detection", REPOSITORY, {
    revision: REVISION,
    dtype: precision,
    device,
    progress_callback: (value) => send(requestId, "downloadProgress", { ...value, progress: value.status === "ready" ? 100 : value.progress }),
  });
}

function normalizeLabels(labels) {
  if (!Array.isArray(labels) || labels.length < 1 || labels.length > 32) throw Error("userInput.candidate_labels must contain 1 to 32 strings");
  return labels.map((label, index) => {
    if (typeof label !== "string" || !label.trim() || label.length > 100) throw Error(`candidate_labels[${index}] must be a non-empty string of at most 100 characters`);
    const normalized = label.trim().toLowerCase();
    return normalized.endsWith(".") ? normalized : `${normalized}.`;
  });
}

function validateConfig(data) {
  for (const [key, fields] of [["modelConfig", MANIFEST.config.model], ["generateConfig", MANIFEST.config.generation]]) {
    const config = data?.[key];
    if (config === undefined) continue;
    if (!config || typeof config !== "object" || Array.isArray(config)) throw Error(`${key} must be an object`);
    for (const name of Object.keys(config)) if (!fields.some((field) => field.name === name)) throw Error(`Unsupported ${key} option: ${name}`);
  }
}

async function generate(data) {
  const image = data?.userInput?.image;
  if (typeof image !== "string" || !image.trim()) throw Error("userInput.image must be a non-empty URL");
  let parsed;
  try { parsed = new URL(image); } catch { throw Error("userInput.image must be a valid URL"); }
  if (!["http:", "https:", "data:", "blob:"].includes(parsed.protocol)) throw Error("userInput.image must use http:, https:, data:, or blob:");
  const labels = normalizeLabels(data?.userInput?.candidate_labels);
  const threshold = data?.modelConfig?.threshold ?? MANIFEST.config.model[0].default;
  if (!Number.isFinite(threshold) || threshold < 0 || threshold > 1) throw Error("modelConfig.threshold must be from 0 through 1");
  const raw = await RawImage.read(image);
  const detections = await detector(raw, labels.join(" "), { threshold, percentage: false });
  return { detections, width: raw.width, height: raw.height };
}

self.addEventListener("message", async ({ data: message = {} }) => {
  const { requestId, type, data } = message;
  try {
    if (type === "checkModelSupports") {
      configure(data?.workerConfig);
      send(requestId, type, { ...capabilities, manifest: MANIFEST });
      return;
    }
    if (type === "init" || type === "download") {
      if (state.initializing || state.generating) throw Error("Worker is busy");
      state.initializing = true;
      state.initialized = false;
      try {
        const precision = data?.precision;
        const device = type === "download" ? ACTIVE_PRECISIONS[precision]?.supportedDevices?.[0] ?? ACTIVE_PRECISIONS[precision]?.candidateDevices?.[0] : data?.device;
        await load({ ...data, precision, device }, requestId);
        if (type === "download") await clear(); else state.initialized = true;
        send(requestId, type, { status: "success" });
      } finally { state.initializing = false; }
      return;
    }
    if (type === "generate") {
      if (state.initializing || !state.initialized || !detector) throw Error("Call init before generate");
      if (state.generating) throw Error("Generation already in progress");
      state.generating = true;
      try {
        validateConfig(data);
        send(requestId, "generated", { status: "success", result: await generate(data) });
      } finally { state.generating = false; }
      return;
    }
    if (type === "generateStream") throw Error("Stream generation is not supported");
    if (type === "clearMemory") {
      if (state.initializing || state.generating) throw Error("Worker is busy");
      state.initializing = true;
      try { await clear(); send(requestId, type, { status: "success" }); }
      finally { state.initializing = false; }
      return;
    }
    throw Error(`Unknown operation: ${type}`);
  } catch (error) {
    send(requestId, "error", { message: error instanceof Error ? error.message : String(error), context: type });
  }
});

self.postMessage({ type: "worker initialized", data: { success: true, workerVersion: "v4", modelId: ID } });
