import { AutoModel, AutoTokenizer, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0";

const ID = "embeddinggemma-300m";
const REPO = "onnx-community/embeddinggemma-300m-ONNX";
const REV = "5090578d9565bb06545b4552f76e6bc2c93e4a66";
const SHARED_BYTES = 26171811;
const candidate = (weightsSize, dtype, modelKeys, options = {}) => ({ weightsSize, size: weightsSize + SHARED_BYTES, modelKeys, dtype, candidateDevices: ["wasm", "webgpu"], ...options });
const PRECISIONS = {
  fp32: candidate(1235001020, "fp32", ["model.onnx", "model.onnx_data"]),
  fp16: candidate(618089375, "fp16", ["model_fp16.onnx", "model_fp16.onnx_data"]),
  q8: candidate(309458498, "q8", ["model_quantized.onnx", "model_quantized.onnx_data"]),
  q4: candidate(197245082, "q4", ["model_q4.onnx", "model_q4.onnx_data"]),
  q4f16: candidate(176115397, "q4f16", ["model_q4f16.onnx", "model_q4f16.onnx_data"]),
  no_gather_q4: candidate(195158084, "fp32", ["model_no_gather_q4.onnx", "model_no_gather_q4.onnx_data"], { modelFileName: "model_no_gather_q4" }),
};
const SUPPORTED = {
  fp32: { ...PRECISIONS.fp32, supportedDevices: ["wasm", "webgpu"] },
  fp16: { ...PRECISIONS.fp16, supportedDevices: ["wasm"] },
  q8: { ...PRECISIONS.q8, supportedDevices: ["wasm", "webgpu"] },
  q4: { ...PRECISIONS.q4, supportedDevices: ["webgpu"] },
};

const MANIFEST = {
  contractVersion: "1.0",
  model: {
    id: ID, displayName: "EmbeddingGemma 300M", provider: "Google DeepMind", providerUrl: "https://ai.google.dev/gemma/docs/embeddinggemma",
    license: "LicenseRef-Gemma", licenseUrl: "https://ai.google.dev/gemma/terms", lastUpdated: "2025-09-25T15:11:17.000Z",
    sourceRepository: "google/embeddinggemma-300m", repository: REPO, artifactRevision: REV, artifactLastUpdated: "2025-09-04T15:43:56.000Z",
    task: "feature-extraction",
    description: "A 300-million-parameter multilingual Google embedding model that creates normalized 768-dimensional text vectors locally in the browser, with optional Matryoshka truncation to 512, 256, or 128 dimensions.",
    intendedUses: ["Semantic search and retrieval across more than 100 trained languages.", "Text similarity, clustering, classification, question answering, fact-verification retrieval, and code retrieval."],
    limitations: ["Retrieval quality depends on using the documented query/document prefixes and varies by language, domain, ambiguity, and input quality.", "Inputs longer than 2,048 tokens are truncated by the model.", "Embedding similarity is not factual verification and can reproduce training-data gaps or bias.", "FP16 activations and derivatives are explicitly unsupported upstream; quantization and Matryoshka truncation can alter ranking margins."],
  },
  runtime: { engine: { name: "Transformers.js", version: "4.2.0" }, precisions: SUPPORTED },
  input: { description: "One or more non-empty, preformatted query or document strings.", alternatives: [{ name: "texts", type: "string[]", required: true }] },
  config: { model: [{ name: "dimensions", type: "integer", default: 768, enum: [768, 512, 256, 128], description: "Matryoshka output dimension; truncated vectors are re-normalized." }], generation: [] },
  output: { description: "One finite normalized embedding per input string.", fields: [{ name: "result", type: "number[][]", required: true }], example: { result: [[0.01, -0.02]] } },
  operations: ["checkModelSupports", "init", "download", "generate", "clearMemory"],
  benchmark: { report: "benchmarks/embeddinggemma-300m.json", testedAt: "2026-09-12T14:42:07.406Z", fixture: { file: "embeddinggemma-300m.json", url: "https://f005.backblazeb2.com/file/hubters-webai-public/assets/benchmarks/embeddinggemma-300m.json", size: 634, sha1: "9a1d261338de718b559f5544e81e488477672779" }, candidates: PRECISIONS, qualityGate: "Finite normalized vectors with the correct dimensions, distinct rows, and the relevant-document cosine exceeding the unrelated-document cosine by at least 0.1.", invalidatedBy: ["artifact revision change", "runtime version change", "fixture or quality-gate change", "browser/backend behavior change"], results: [
    { precision: "fp32", device: "wasm", status: "pass", runtime: { embeddingsPerSecond: 9.3575, grade: "slow" }, quality: { margin: 0.6141 } },
    { precision: "fp32", device: "webgpu", status: "pass", runtime: { embeddingsPerSecond: 11.9379, grade: "good" }, quality: { margin: 0.6141 } },
    { precision: "fp16", device: "wasm", status: "pass", runtime: { embeddingsPerSecond: 8.8836, grade: "slow" }, quality: { margin: 0.6141 } },
    { precision: "fp16", device: "webgpu", status: "fail", failure: "WebGPU LayerNorm shader validation failed during inference." },
    { precision: "q8", device: "wasm", status: "pass", runtime: { embeddingsPerSecond: 6.7325, grade: "slow" }, quality: { margin: 0.6085 } },
    { precision: "q8", device: "webgpu", status: "pass", runtime: { embeddingsPerSecond: 17.1527, grade: "good" }, quality: { margin: 0.6085 } },
    { precision: "q4", device: "wasm", status: "fail", failure: "WASM has no implementation for GatherBlockQuantized." },
    { precision: "q4", device: "webgpu", status: "pass", runtime: { embeddingsPerSecond: 37.2208, grade: "excellent" }, quality: { margin: 0.5983 } },
    { precision: "q4f16", device: "wasm", status: "fail", failure: "WASM has no GatherBlockQuantized kernel." },
    { precision: "q4f16", device: "webgpu", status: "fail", failure: "WebGPU LayerNorm shader validation failed during inference." },
    { precision: "no_gather_q4", device: "wasm", status: "pass", runtime: { embeddingsPerSecond: 1.9249, grade: "very-slow" }, quality: { margin: 0.5969 }, advertised: false },
    { precision: "no_gather_q4", device: "webgpu", status: "pass", runtime: { embeddingsPerSecond: 15.9574, grade: "good" }, quality: { margin: 0.5969 }, advertised: false },
  ] },
};

const state = { initializing: false, initialized: false, generating: false };
let model = null, tokenizer = null;
const caps = { supportedModes: ["webai"], supportedPrecisions: Object.keys(SUPPORTED), supportedPrecisionsDevicesMap: SUPPORTED, doesSupportStreamGeneration: false, externalInterrupt: true, workerVersion: "v4", transformersJsVersion: "4.2.0", cacheModelId: REPO };
const send = (requestId, type, data) => self.postMessage({ requestId, type, data });
function configure(c = {}) { env.allowRemoteModels = c.allowRemoteModels ?? true; env.allowLocalModels = c.allowLocalModels ?? false; env.useBrowserCache = c.useBrowserCache ?? true; if (c.remoteHost) env.remoteHost = c.remoteHost; if (c.remotePathTemplate) env.remotePathTemplate = c.remotePathTemplate; if (c.localModelPath) env.localModelPath = c.localModelPath; }
function validateInit(data, allowCandidates = false) { const p = data?.precision, d = data?.device, s = (allowCandidates ? PRECISIONS : SUPPORTED)[p]; if (!s) throw Error(`Unsupported precision: ${p}`); if (!s.candidateDevices.includes(d) && !s.supportedDevices?.includes(d)) throw Error(`Precision ${p} is not available on ${d}`); if (d === "webgpu" && !self.navigator?.gpu) throw Error("WebGPU is unavailable"); return { p, d, s }; }
async function clear() { if (model?.dispose) await model.dispose(); model = null; tokenizer = null; state.initialized = false; }
async function load(data, requestId) { const { d, s } = validateInit(data, data?.benchmark === true); await clear(); const options = { revision: REV, dtype: s.dtype, device: d, progress_callback: value => send(requestId, "downloadProgress", value) }; if (s.modelFileName) options.model_file_name = s.modelFileName; [tokenizer, model] = await Promise.all([AutoTokenizer.from_pretrained(REPO, { revision: REV, progress_callback: options.progress_callback }), AutoModel.from_pretrained(REPO, options)]); }
async function generate(data) {
  const texts = data?.userInput?.texts;
  if (!Array.isArray(texts) || !texts.length || texts.some(text => typeof text !== "string" || !text.trim())) throw Error("userInput.texts must be a non-empty array of non-empty strings");
  const dimensions = data?.modelConfig?.dimensions ?? 768;
  if (![768, 512, 256, 128].includes(dimensions)) throw Error("dimensions must be 768, 512, 256, or 128");
  const inputs = await tokenizer(texts, { padding: true, truncation: true, max_length: 2048 });
  const output = await model(inputs), tensor = output?.sentence_embedding;
  if (!tensor?.tolist) throw Error("Model returned no sentence_embedding tensor");
  const rows = tensor.tolist().map(row => { const values = row.slice(0, dimensions); const norm = Math.sqrt(values.reduce((sum, value) => sum + value * value, 0)); if (!(norm > 0) || values.some(value => !Number.isFinite(value))) throw Error("Model returned an invalid embedding"); return values.map(value => value / norm); });
  return { result: rows };
}
self.addEventListener("message", async ({ data: { requestId, type, data } = {} }) => { try { if (type === "checkModelSupports") { configure(data?.workerConfig); send(requestId, type, { ...caps, manifest: MANIFEST }); return; } if (type === "download") { await load(data, requestId); await clear(); send(requestId, type, { status: "success" }); return; } if (type === "init") { if (state.initializing || state.generating) throw Error("Worker is busy"); state.initializing = true; try { await load(data, requestId); state.initialized = true; send(requestId, type, { status: "success" }); } finally { state.initializing = false; } return; } if (type === "generate") { if (!state.initialized || !model || !tokenizer) throw Error("Call init before generate"); if (state.generating) throw Error("Generation already in progress"); state.generating = true; try { send(requestId, "generated", { status: "success", result: await generate(data) }); } finally { state.generating = false; } return; } if (type === "generateStream") throw Error("Stream generation is not supported for embeddings"); if (type === "clearMemory") { await clear(); send(requestId, type, { status: "success" }); return; } throw Error(`Unknown operation: ${type}`); } catch (error) { send(requestId, "error", { message: error instanceof Error ? error.message : String(error), context: type }); } });
self.postMessage({ type: "worker initialized", data: { success: true, workerVersion: "v4", modelId: ID } });
