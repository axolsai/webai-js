import { AutoModelForSequenceClassification, AutoTokenizer, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0";

const ID = "bge-reranker-base";
const REPO = "Xenova/bge-reranker-base";
const REV = "280bcc27a84e0b898c251e06fddb25171bd9b101";
const SHARED_BYTES = 22169516;
const make = (weightsSize, dtype, suffix) => ({ weightsSize, size: weightsSize + SHARED_BYTES, dtype, modelKeys: [`model${suffix}.onnx`], candidateDevices: ["wasm", "webgpu"] });
const PRECISIONS = {
  fp32: make(1112459588, "fp32", ""),
  fp16: make(556462150, "fp16", "_fp16"),
  int8: make(278825308, "int8", "_int8"),
  uint8: make(278825337, "uint8", "_uint8"),
  q8: make(279301077, "q8", "_quantized"),
  q4: make(825849673, "q4", "_q4"),
  q4f16: make(434321936, "q4f16", "_q4f16"),
  bnb4: make(820541785, "bnb4", "_bnb4"),
};
// Replaced from the complete browser matrix before release.
const SUPPORTED = {
  fp32: { ...PRECISIONS.fp32, supportedDevices: ["wasm","webgpu"] },
  int8: { ...PRECISIONS.int8, supportedDevices: ["wasm","webgpu"] },
  uint8: { ...PRECISIONS.uint8, supportedDevices: ["wasm","webgpu"] },
  q8: { ...PRECISIONS.q8, supportedDevices: ["wasm","webgpu"] },
  q4: { ...PRECISIONS.q4, supportedDevices: ["wasm","webgpu"] },
  bnb4: { ...PRECISIONS.bnb4, supportedDevices: ["wasm","webgpu"] },
};
const MANIFEST = {
  contractVersion: "1.0",
  model: {
    id: ID,
    displayName: "BGE Reranker Base — Cross-Encoder Ranking",
    provider: "BAAI",
    providerUrl: "https://www.baai.ac.cn/",
    license: "MIT",
    licenseUrl: "https://opensource.org/license/mit/",
    lastUpdated: "2024-06-24T14:10:03.000Z",
    sourceRevision:"2cfc18c9415c912f9d8155881c133215df768a70",sourceRepository: "BAAI/bge-reranker-base",
    repository: REPO,
    artifactRevision: REV,
    artifactLastUpdated: "2025-06-30T19:07:27.000Z",
    task: "text-ranking",
    description: "A Chinese-and-English BAAI cross-encoder, converted by Xenova for Transformers.js, that scores each query-passage pair and returns passages ordered by estimated relevance.",
    intendedUses: ["Rerank a small candidate set produced by semantic or lexical retrieval.", "Improve the ordering of Chinese or English search and RAG passages."],
    limitations: ["Every query-passage pair requires model inference, making this slower than embedding retrieval for large corpora.", "Even the smallest tested weight file is about 279 MB and was not retained by Cache API in the isolated Chrome audit profile, so affected browser sessions redownload it.", "Scores are relative ranking signals, not calibrated probabilities or factuality judgments.", "Long passages are truncated and relevant content beyond the token limit can be missed.", "Ranking can reflect training-data bias and degrade on unsupported languages, specialized domains, malformed text, or adversarial content.", "Do not use rankings as the sole basis for high-impact decisions."],
  },
  runtime: { engine: { name: "Transformers.js", version: "4.2.0" }, precisions: SUPPORTED },
  input: {
    description: "One non-empty query and one or more non-empty candidate passages.",
    alternatives: [{ name: "query and documents", description: "Scores every document against the same query.", fields: [{ name: "query", type: "string", required: true }, { name: "documents", type: "string[]", required: true }] }],
  },
  config: {
    model: [{ name: "max_length", type: "integer", default: 512, minimum: 8, maximum: 512 }, { name: "top_k", type: "integer|null", default: null, minimum: 1 }, { name: "normalize_scores", type: "boolean", default: true }],
    generation: [],
  },
  output: {
    description: "Candidate passages sorted by descending relevance, including raw logits, optional sigmoid scores, and original indices.",
    fields: [{ name: "results", type: "Array<{document:string,score:number,logit:number,originalIndex:number}>" }],
    example: { results: [{ document: "Paris is the capital of France.", score: 0.99, logit: 4.6, originalIndex: 0 }] },
  },
  operations: ["checkModelSupports", "init", "download", "generate", "clearMemory"],
  benchmark: {
    report: "benchmarks/bge-reranker-base.json",
    testedAt: "2026-09-14T20:55:17.075Z",
    fixture: { file: "bge-m3.json", url: "https://f005.backblazeb2.com/file/hubters-webai-public/assets/benchmarks/bge-m3.json", size: 500, sha1: "75a7770d06fc312ddeb39987712a8b4bbe6f1e08", source:"Original WebAI semantic retrieval fixture.", license:"MIT", licenseUrl:"https://opensource.org/license/mit/", preprocessing:"First text is the query; remaining texts are candidate documents." },
    candidates: PRECISIONS,
    qualityGate: "Finite scores, deterministic descending order, and the France passage must outrank the unrelated whale passage by at least 0.25 sigmoid probability.",
    note: "Measurements are specific to the tested browser and hardware.",
    environment: "Isolated headless Chrome on macOS; Transformers.js 4.2.0",
    invalidatedBy: ["artifact revision change", "runtime version change", "fixture or quality-gate change", "browser/backend behavior change"],
    results: [{"precision":"fp32","device":"wasm","status":"pass","runtime":{"pairsPerSecond":9.62,"grade":"good"},"quality":{"grade":"ranking-pass","margin":0.9997}},{"precision":"fp32","device":"webgpu","status":"pass","runtime":{"pairsPerSecond":4.575,"grade":"slow"},"quality":{"grade":"ranking-pass","margin":0.9997}},{"precision":"fp16","device":"wasm","status":"fail","failure":"Error: Can't create a session. ERROR_CODE: 1, ERROR_MESSAGE: /mnt/vss/_work/1/s/onnxruntime/core/graph/graph_utils.cc:30 int onnxruntime::graph_utils::GetIndexFromName(const Node &, const std::string &, bool) itr != node_args.end() was false. Attempting to get index by a name which does not exist:InsertedPrecisionFreeCast_/roberta/encoder/layer.11/output/LayerNorm/Constant_output_0for node: /roberta/embeddings/LayerNorm/Mul/SimplifiedLayerNormFusion/\n"},{"precision":"fp16","device":"webgpu","status":"fail","failure":"quality gate failed: top=1, margin=-0.24457595644260421"},{"precision":"int8","device":"wasm","status":"pass","runtime":{"pairsPerSecond":8.981,"grade":"good"},"quality":{"grade":"ranking-pass","margin":0.9999}},{"precision":"int8","device":"webgpu","status":"pass","runtime":{"pairsPerSecond":6.527,"grade":"slow"},"quality":{"grade":"ranking-pass","margin":0.9998}},{"precision":"uint8","device":"wasm","status":"pass","runtime":{"pairsPerSecond":9.083,"grade":"good"},"quality":{"grade":"ranking-pass","margin":0.9998}},{"precision":"uint8","device":"webgpu","status":"pass","runtime":{"pairsPerSecond":6.487,"grade":"slow"},"quality":{"grade":"ranking-pass","margin":0.9998}},{"precision":"q8","device":"wasm","status":"pass","runtime":{"pairsPerSecond":9.021,"grade":"good"},"quality":{"grade":"ranking-pass","margin":0.9996}},{"precision":"q8","device":"webgpu","status":"pass","runtime":{"pairsPerSecond":5.402,"grade":"slow"},"quality":{"grade":"ranking-pass","margin":0.9996}},{"precision":"q4","device":"wasm","status":"pass","runtime":{"pairsPerSecond":6.2,"grade":"slow"},"quality":{"grade":"ranking-pass","margin":0.9998}},{"precision":"q4","device":"webgpu","status":"pass","runtime":{"pairsPerSecond":8.361,"grade":"good"},"quality":{"grade":"ranking-pass","margin":0.9998}},{"precision":"q4f16","device":"wasm","status":"fail","failure":"Error: Can't create a session. ERROR_CODE: 1, ERROR_MESSAGE: /mnt/vss/_work/1/s/onnxruntime/core/graph/graph_utils.cc:30 int onnxruntime::graph_utils::GetIndexFromName(const Node &, const std::string &, bool) itr != node_args.end() was false. Attempting to get index by a name which does not exist:InsertedPrecisionFreeCast_/roberta/embeddings/LayerNorm/Constant_output_0for node: /roberta/embeddings/LayerNorm/Mul/SimplifiedLayerNormFusion/\n"},{"precision":"q4f16","device":"webgpu","status":"fail","failure":"quality gate failed: top=1, margin=-0.24198654101080747"},{"precision":"bnb4","device":"wasm","status":"pass","runtime":{"pairsPerSecond":7.66,"grade":"slow"},"quality":{"grade":"ranking-pass","margin":0.9996}},{"precision":"bnb4","device":"webgpu","status":"pass","runtime":{"pairsPerSecond":3.876,"grade":"slow"},"quality":{"grade":"ranking-pass","margin":0.9996}}],
  },
};

const state = { initializing: false, initialized: false, generating: false };
let tokenizer = null;
let model = null;
const caps = { supportedModes: ["webai"], supportedPrecisions: Object.keys(SUPPORTED), supportedPrecisionsDevicesMap: SUPPORTED, doesSupportStreamGeneration: false, externalInterrupt: true, workerVersion: "v4", transformersJsVersion: "4.2.0", cacheModelId: REPO };
const send = (requestId, type, data) => self.postMessage({ requestId, type, data });
function configure(c = {}) { env.allowRemoteModels = c.allowRemoteModels ?? true; env.allowLocalModels = c.allowLocalModels ?? false; env.useBrowserCache = c.useBrowserCache ?? true; if (c.remoteHost) env.remoteHost = c.remoteHost; if (c.remotePathTemplate) env.remotePathTemplate = c.remotePathTemplate; if (c.localModelPath) env.localModelPath = c.localModelPath; }
function validateInit(data) { const p = data?.precision, d = data?.device, s = SUPPORTED[p]; if (!s) throw Error(`Unsupported precision: ${p}`); const devices = s.supportedDevices ?? s.candidateDevices; if (!devices.includes(d)) throw Error(`Precision ${p} is only supported on: ${devices.join(", ")}`); if (d === "webgpu" && !self.navigator?.gpu) throw Error("WebGPU is unavailable"); return { p, d }; }
async function clear() { if (model?.dispose) await model.dispose(); model = null; tokenizer = null; state.initialized = false; }
async function load(data, id) { const { p, d } = validateInit(data); await clear(); const progress_callback = value => send(id, "downloadProgress", value); tokenizer = await AutoTokenizer.from_pretrained(REPO, { revision: REV, progress_callback }); model = await AutoModelForSequenceClassification.from_pretrained(REPO, { revision: REV, dtype: p, device: d, progress_callback }); }
function options(data, count) { const c = { ...Object.fromEntries(MANIFEST.config.model.map(({name, default: value}) => [name, value])), ...data?.modelConfig }; if (!Number.isInteger(c.max_length) || c.max_length < 8 || c.max_length > 512) throw Error("max_length must be an integer from 8 through 512"); if (c.top_k !== null && (!Number.isInteger(c.top_k) || c.top_k < 1 || c.top_k > count)) throw Error(`top_k must be null or an integer from 1 through ${count}`); if (typeof c.normalize_scores !== "boolean") throw Error("normalize_scores must be boolean"); return c; }
async function generate(data) { const query = data?.userInput?.query, documents = data?.userInput?.documents; if (typeof query !== "string" || !query.trim()) throw Error("userInput.query must be a non-empty string"); if (!Array.isArray(documents) || !documents.length || documents.some(x => typeof x !== "string" || !x.trim())) throw Error("userInput.documents must be a non-empty array of non-empty strings"); const c = options(data, documents.length); const inputs = tokenizer(new Array(documents.length).fill(query), { text_pair: documents, padding: true, truncation: true, max_length: c.max_length }); const outputs = await model(inputs); const logits = Array.from(outputs.logits.data, Number); if (logits.length !== documents.length || logits.some(x => !Number.isFinite(x))) throw Error("Model returned invalid ranking logits"); const ranked = documents.map((document, originalIndex) => ({ document, score: c.normalize_scores ? 1 / (1 + Math.exp(-logits[originalIndex])) : logits[originalIndex], logit: logits[originalIndex], originalIndex })).sort((a, b) => b.score - a.score || a.originalIndex - b.originalIndex); return { results: c.top_k === null ? ranked : ranked.slice(0, c.top_k) }; }
function validateConfig(data) {
  for (const [key, fields] of [["modelConfig", MANIFEST.config.model], ["generateConfig", MANIFEST.config.generation]]) {
    const config = data?.[key];
    if (config === undefined) continue;
    if (!config || typeof config !== "object" || Array.isArray(config)) throw Error(`${key} must be an object`);
    for (const name of Object.keys(config)) if (!fields.some(field => field.name === name)) throw Error(`Unsupported ${key} option: ${name}`);
  }
}
self.addEventListener("message", async ({ data: message = {} }) => {
  const { requestId, type, data } = message;
  try {
    if (type === "checkModelSupports") {
      configure(data?.workerConfig);
      send(requestId, type, { ...caps, manifest: MANIFEST });
      return;
    }
    if (type === "init" || type === "download") {
      if (state.initializing || state.generating) throw Error("Worker is busy");
      state.initializing = true;
      state.initialized = false;
      try {
        const precision = data?.precision;
        const device = type === "download" ? SUPPORTED[precision]?.supportedDevices[0] : data?.device;
        await load({ ...data, precision, device }, requestId);
        if (type === "download") await clear();
        else state.initialized = true;
        send(requestId, type, { status: "success" });
      } finally { state.initializing = false; }
      return;
    }
    if (type === "generate") {
      if (state.initializing || !state.initialized || !model) throw Error("Call init before generate");
      if (state.generating) throw Error("Generation already in progress");
      state.generating = true;
      try { validateConfig(data); send(requestId, "generated", { status: "success", result: await generate(data) }); }
      finally { state.generating = false; }
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
