import {
  env,
  AutoProcessor,
  AutoTokenizer,
  SiglipModel,
  ZeroShotImageClassificationPipeline,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0";

const ID = "siglip-base-patch16-224";
const REPO = "onnx-community/siglip-base-patch16-224-ONNX";
const REV = "9f0328bdd0eb4f62d0f6c10625f43d882b32102b";
const TOKENIZER_REPO = "google/siglip-base-patch16-224";
const TOKENIZER_REV = "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed";
const SHARED_BYTES = 918 + 394 + 2399357 + 711;

function candidate(weightsSize, dtype, filename) {
  return { weightsSize, size: weightsSize + SHARED_BYTES, modelKeys: [filename], auxiliaryRepositories: [{ repository: TOKENIZER_REPO, revision: TOKENIZER_REV, modelKeys: ["tokenizer.json", "tokenizer_config.json"] }], dtype, candidateDevices: ["wasm", "webgpu"] };
}

const PRECISIONS = {
  fp32: candidate(813080024, "fp32", "model.onnx"),
  q8: candidate(210058412, "q8", "model_quantized.onnx"),
  int8: candidate(210058412, "int8", "model_int8.onnx"),
  uint8: candidate(210058502, "uint8", "model_uint8.onnx"),
  q4: candidate(223867620, "q4", "model_q4.onnx"),
  bnb4: candidate(212956802, "bnb4", "model_bnb4.onnx"),
};

const SUPPORTED = {
  q8: { ...PRECISIONS.q8, supportedDevices: ["wasm", "webgpu"] },
  int8: { ...PRECISIONS.int8, supportedDevices: ["wasm", "webgpu"] },
  uint8: { ...PRECISIONS.uint8, supportedDevices: ["wasm", "webgpu"] },
  q4: { ...PRECISIONS.q4, supportedDevices: ["wasm", "webgpu"] },
  bnb4: { ...PRECISIONS.bnb4, supportedDevices: ["wasm", "webgpu"] },
};

const MANIFEST = {
  contractVersion: "1.0",
  model: {
    id: ID,
    displayName: "SigLIP Base Patch16 224 — Zero-Shot Image Classification",
    provider: "Google Research",
    providerUrl: "https://github.com/google-research/big_vision",
    license: "Apache-2.0",
    licenseUrl: "https://www.apache.org/licenses/LICENSE-2.0.txt",
    lastUpdated: "2024-09-26T08:20:18.000Z",
    sourceRepository: TOKENIZER_REPO,
    repository: REPO,
    artifactRevision: REV,
    artifactLastUpdated: "2026-09-08T14:57:51.000Z",
    task: "zero-shot-image-classification",
    description: "A SigLIP vision-language model that compares a 224×224 image with developer-supplied English labels and returns an independent sigmoid score for each label in the browser.",
    intendedUses: ["Zero-shot image categorization with custom labels.", "Image-text matching and exploratory content organization."],
    limitations: [
      "Scores are independent sigmoid probabilities and do not sum to one.",
      "Accuracy depends heavily on label wording, image framing, resolution, and visual domain.",
      "The model was trained on English image-text pairs; non-English labels are not validated by this worker.",
      "Predictions can reflect training-data bias and must not be used as authoritative safety, identity, medical, or legal judgments.",
    ],
  },
  runtime: { engine: { name: "Transformers.js", version: "4.2.0" }, precisions: SUPPORTED },
  input: {
    description: "One browser-readable image URL and at least two non-empty English candidate labels.",
    alternatives: [
      { name: "image_blob_url", type: "string", required: true },
      { name: "candidate_labels", type: "string[]", required: true },
    ],
  },
  config: { model: [], generation: [{ name: "hypothesis_template", type: "string", default: "This is a photo of {}", description: "English template containing one {} placeholder." }] },
  output: { description: "Candidate labels sorted by descending independent sigmoid score.", fields: [{ name: "predictions", type: "Array<{label:string,score:number}>", required: true }], example: { predictions: [{ label: "a portrait photo of a person", score: 0.91 }] } },
  operations: ["checkModelSupports", "init", "download", "generate", "clearMemory"],
  benchmark: {
    report: "benchmarks/siglip-base-patch16-224.json",
    testedAt: "2026-09-11T06:37:00.000Z",
    fixture: { file: "modnet-portrait-pexels-5965592.jpg", url: "https://f005.backblazeb2.com/file/hubters-webai-public/assets/benchmarks/modnet-portrait-pexels-5965592.jpg", size: 78124, sha1: "38212739a4a4f0be25d84863cf991d7fb6c6d818", width: 1024, height: 683, license: "Pexels License", labels: { file: "siglip-base-patch16-224.json", url: "https://f005.backblazeb2.com/file/hubters-webai-public/assets/benchmarks/siglip-base-patch16-224.json", size: 460, sha1: "fecc3379e545c31c5fc0dbcd8ca8777189f661e1" } },
    candidates: PRECISIONS,
    qualityGate: "Finite scores in [0,1], unique labels, expected person label ranked first, and its score is at least 10 times the strongest negative.",
    invalidatedBy: ["artifact or tokenizer revision change", "runtime version change", "fixture, labels, or quality-gate change", "browser/backend behavior change"],
    results: [
      { precision: "fp32", device: "wasm", status: "fail", failure: "cold load exceeded the bounded 15-minute case timeout" },
      { precision: "fp32", device: "webgpu", status: "fail", failure: "network error during the 813 MB cold load" },
      { precision: "q8", device: "wasm", status: "pass", runtime: { imagesPerSecond: 0.6303, grade: "very-slow" }, quality: { topToNegativeRatio: 409.14 } },
      { precision: "q8", device: "webgpu", status: "pass", runtime: { imagesPerSecond: 0.4859, grade: "very-slow" }, quality: { topToNegativeRatio: 466.89 } },
      { precision: "int8", device: "wasm", status: "pass", runtime: { imagesPerSecond: 0.6603, grade: "very-slow" }, quality: { topToNegativeRatio: 409.14 } },
      { precision: "int8", device: "webgpu", status: "pass", runtime: { imagesPerSecond: 0.6132, grade: "very-slow" }, quality: { topToNegativeRatio: 466.89 } },
      { precision: "uint8", device: "wasm", status: "pass", runtime: { imagesPerSecond: 0.6706, grade: "very-slow" }, quality: { topToNegativeRatio: 304.03 } },
      { precision: "uint8", device: "webgpu", status: "pass", runtime: { imagesPerSecond: 0.6223, grade: "very-slow" }, quality: { topToNegativeRatio: 501.96 } },
      { precision: "q4", device: "wasm", status: "pass", runtime: { imagesPerSecond: 0.5572, grade: "very-slow" }, quality: { topToNegativeRatio: 163.49 } },
      { precision: "q4", device: "webgpu", status: "pass", runtime: { imagesPerSecond: 3.5997, grade: "good" }, quality: { topToNegativeRatio: 163.49 } },
      { precision: "bnb4", device: "wasm", status: "pass", runtime: { imagesPerSecond: 0.6258, grade: "very-slow" }, quality: { topToNegativeRatio: 305.15 } },
      { precision: "bnb4", device: "webgpu", status: "pass", runtime: { imagesPerSecond: 0.6051, grade: "very-slow" }, quality: { topToNegativeRatio: 305.15 } },
    ],
  },
};

const state = { initializing: false, initialized: false, generating: false };
let classifier = null;
const send = (requestId, type, data) => self.postMessage({ requestId, type, data });
const caps = { supportedModes: ["webai"], supportedPrecisions: Object.keys(SUPPORTED), supportedPrecisionsDevicesMap: SUPPORTED, doesSupportStreamGeneration: false, externalInterrupt: true, workerVersion: "v4", transformersJsVersion: "4.2.0", cacheModelId: REPO };

function configure(config = {}) {
  env.allowRemoteModels = config.allowRemoteModels ?? true;
  env.allowLocalModels = config.allowLocalModels ?? false;
  env.useBrowserCache = config.useBrowserCache ?? true;
  if (config.remoteHost) env.remoteHost = config.remoteHost;
  if (config.remotePathTemplate) env.remotePathTemplate = config.remotePathTemplate;
  if (config.localModelPath) env.localModelPath = config.localModelPath;
}

function validateInit(data) {
  const precision = data?.precision, device = data?.device, supported = SUPPORTED[precision];
  if (!supported) throw Error(`Unsupported precision: ${precision}`);
  if (!supported.supportedDevices.includes(device)) throw Error(`Precision ${precision} is only supported on: ${supported.supportedDevices.join(", ")}`);
  if (device === "webgpu" && !self.navigator?.gpu) throw Error("WebGPU is unavailable");
  return { precision, device };
}

async function clear() {
  if (classifier?.dispose) await classifier.dispose();
  classifier = null;
  state.initialized = false;
}

async function load(data, requestId) {
  const { precision, device } = validateInit(data);
  await clear();
  const options = { revision: REV, progress_callback: (value) => send(requestId, "downloadProgress", value) };
  const [tokenizer, processor, model] = await Promise.all([
    AutoTokenizer.from_pretrained(TOKENIZER_REPO, { revision: TOKENIZER_REV, progress_callback: options.progress_callback }),
    AutoProcessor.from_pretrained(REPO, options),
    SiglipModel.from_pretrained(REPO, { ...options, dtype: precision, device }),
  ]);
  classifier = new ZeroShotImageClassificationPipeline({ task: "zero-shot-image-classification", tokenizer, processor, model });
}

function validateGeneration(data) {
  const image = data?.userInput?.image_blob_url;
  const labels = data?.userInput?.candidate_labels;
  if (typeof image !== "string" || !image.trim()) throw Error("userInput.image_blob_url must be a non-empty string");
  if (!Array.isArray(labels) || labels.length < 2 || labels.length > 100 || labels.some((label) => typeof label !== "string" || !label.trim())) throw Error("userInput.candidate_labels must contain 2 through 100 non-empty strings");
  const candidateLabels = labels.map((label) => label.trim());
  if (new Set(candidateLabels).size !== candidateLabels.length) throw Error("candidate labels must be unique");
  const hypothesisTemplate = data?.generateConfig?.hypothesis_template ?? "This is a photo of {}";
  if (typeof hypothesisTemplate !== "string" || hypothesisTemplate.length > 500 || !hypothesisTemplate.includes("{}")) throw Error("generateConfig.hypothesis_template must contain {} and be at most 500 characters");
  return { image: image.trim(), candidateLabels, hypothesisTemplate };
}

async function generate(data) {
  const { image, candidateLabels, hypothesisTemplate } = validateGeneration(data);
  const output = await classifier(image, candidateLabels, { hypothesis_template: hypothesisTemplate });
  if (!Array.isArray(output) || output.length !== candidateLabels.length || output.some((item) => typeof item?.label !== "string" || !Number.isFinite(item?.score))) throw Error("Model returned invalid predictions");
  return { predictions: output.map(({ label, score }) => ({ label, score })) };
}

self.addEventListener("message", async ({ data: { requestId, type, data } = {} }) => {
  try {
    if (type === "checkModelSupports") { configure(data?.workerConfig); send(requestId, type, { ...caps, manifest: MANIFEST }); return; }
    if (type === "download") { if (state.initializing || state.generating) throw Error("Worker is busy"); const supported = SUPPORTED[data?.precision]; if (!supported) throw Error(`Unsupported precision: ${data?.precision}`); const device = supported.supportedDevices.includes("wasm") ? "wasm" : supported.supportedDevices[0]; await load({ ...data, device }, requestId); await clear(); send(requestId, type, { status: "success" }); return; }
    if (type === "init") { if (state.initializing || state.generating) throw Error("Worker is busy"); state.initializing = true; try { await load(data, requestId); state.initialized = true; send(requestId, type, { status: "success" }); } finally { state.initializing = false; } return; }
    if (type === "generate") { if (!state.initialized || !classifier) throw Error("Call init before generate"); if (state.generating) throw Error("Generation already in progress"); state.generating = true; try { send(requestId, "generated", { status: "success", result: await generate(data) }); } finally { state.generating = false; } return; }
    if (type === "generateStream") throw Error("Stream generation is not supported for image classification");
    if (type === "clearMemory") { if (state.generating) throw Error("Cannot clear memory while generation is active"); await clear(); send(requestId, type, { status: "success" }); return; }
    throw Error(`Unknown operation: ${type}`);
  } catch (error) { send(requestId, "error", { message: error instanceof Error ? error.message : String(error), context: type }); }
});

self.postMessage({ type: "worker initialized", data: { success: true, workerVersion: "v4", modelId: ID } });
