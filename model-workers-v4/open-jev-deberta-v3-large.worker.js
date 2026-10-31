import { AutoModel, AutoTokenizer, Tensor, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0";

const ID = "open-jev-deberta-v3-large";
const REPO = "onnx-community/open-jev-deberta-v3-large-ONNX";
const REV = "7c79f25b5ac496089f448a969c801872ad59d31c";
const SHARED_BYTES = 11126810;
const make = (weightsSize, dtype, keys) => ({ weightsSize, size: weightsSize + SHARED_BYTES, dtype, modelKeys: keys, candidateDevices: ["wasm", "webgpu"] });
const PRECISIONS = {
  fp32: make(1749199550, "fp32", ["model.onnx", "model.onnx_data", "model.onnx_data_1"]),
  fp16: make(874883488, "fp16", ["model_fp16.onnx", "model_fp16.onnx_data"]),
  q4: make(478017229, "q4", ["model_q4.onnx", "model_q4.onnx_data"]),
  q4f16: make(348390087, "q4f16", ["model_q4f16.onnx", "model_q4f16.onnx_data"]),
};
const SUPPORTED = {
  fp32: { ...PRECISIONS.fp32, supportedDevices: ["wasm", "webgpu"] },
  fp16: { ...PRECISIONS.fp16, supportedDevices: ["wasm", "webgpu"] },
  q4: { ...PRECISIONS.q4, supportedDevices: ["webgpu"] },
  q4f16: { ...PRECISIONS.q4f16, supportedDevices: ["webgpu"] },
};

const MANIFEST = {
  contractVersion: "1.0",
  model: {
    id: ID,
    displayName: "Open Jev DeBERTa V3 Large — Typed Decisions",
    provider: "Kotoba Labs",
    providerUrl: "https://huggingface.co/com-kotobalabs",
    license: "Apache-2.0",
    licenseUrl: "https://www.apache.org/licenses/LICENSE-2.0.txt",
    lastUpdated: "2026-09-18T10:02:28.000Z",
    sourceRepository: "com-kotobalabs/open-jev-deberta-v3-large",
    sourceRevision: "19bf9a64815add579fbf6c907bef584d9277a8e4",
    repository: REPO,
    artifactRevision: REV,
    artifactLastUpdated: "2026-09-20T12:30:02.000Z",
    task: "typed-decision-classification",
    description: "An independent open Jev-shaped English decision model that runs locally through Transformers.js and returns calibrated distributions for categorical choices, ordered scores, and yes/no questions without generating text.",
    intendedUses: [
      "Route or classify English inputs among an application-provided set of choices.",
      "Estimate ordered ratings and yes/no probabilities for human-reviewed workflows.",
      "Prototype deterministic structured decision interfaces in the browser.",
    ],
    limitations: [
      "This independent reproduction is not TypeSafe AI's proprietary Jev model and is not affiliated with TypeSafe AI.",
      "The source reports 0.854 in-domain accuracy but 0.690 on new instructions and option sets; application-specific evaluation and recalibration are required.",
      "It is English-only, has a 512-token total context, and was trained on banking support, movie reviews, and Wikipedia yes/no data.",
      "Unseen ordered scales are a weak case, and out-of-domain confidence may be over-confident.",
      "It cannot generate explanations, and its probabilities do not establish truth, safety, or suitability for consequential decisions.",
      "The model is large for browser use: even the smallest artifact is roughly 348 MB before shared tokenizer files.",
      "The isolated public-API audit did not retain the q4f16 artifact in Chrome's Cache API, so some browser profiles may redownload it between sessions.",
    ],
  },
  runtime: { engine: { name: "Transformers.js", version: "4.2.0" }, precisions: SUPPORTED },
  input: {
    description: "One non-empty English state and one or more typed questions. The state is token-truncated; the complete encoded request must fit the configured total token limit.",
    alternatives: [{
      name: "typed decisions",
      fields: [
        { name: "state", type: "string", required: true, minLength: 1, description: "English program or workflow state; token-truncated to state_max_tokens." },
        { name: "questions", type: "Array<TypedDecisionQuestion>", required: true, minItems: 1, items: { type: "object", description: "type is choice, score, or noul; instructions is non-empty; choice has 2–255 options, score has 2–10 ordered options, and noul always uses no/yes." } },
      ],
    }],
  },
  config: {
    model: [
      { name: "max_length", type: "integer", default: 512, minimum: 8, maximum: 512, description: "Maximum total encoded tokens; requests whose questions do not fit are rejected." },
      { name: "state_max_tokens", type: "integer", default: 256, minimum: 1, maximum: 256, description: "Maximum state tokens retained before questions are appended." },
      { name: "temperature", type: "number", default: 1.05, minimum: 0.01, maximum: 10, description: "Softmax calibration temperature; 1.05 is fitted by the source model." },
    ],
    generation: [],
  },
  output: {
    description: "One typed result per question, preserving input order and including the complete calibrated option distribution.",
    fields: [{ name: "decisions", type: "Array<{type:string,choice?:string,score?:number,noul?:number,confidence:number,probabilities:Array<{option:string,probability:number}>}>" }],
    example: { decisions: [{ type: "choice", choice: "refund & dispute", confidence: 0.82, probabilities: [{ option: "refund & dispute", probability: 0.82 }, { option: "other", probability: 0.18 }] }] },
  },
  operations: ["checkModelSupports", "init", "download", "generate", "clearMemory"],
  benchmark: {
    report: "benchmarks/open-jev-deberta-v3-large.json",
    testedAt: "2026-09-22T14:47:49.157Z",
    fixture: { file: "open-jev-deberta-v3-large.json", url: "https://f005.backblazeb2.com/file/hubters-webai-public/assets/benchmarks/open-jev-deberta-v3-large.json", size: 819, sha1: "bf1e5a03646050ac2c9b0fa51bf8f01b72f95ece", source: "Adapted from the pinned source model card usage example.", sourceUrl: "https://huggingface.co/com-kotobalabs/open-jev-deberta-v3-large/blob/19bf9a64815add579fbf6c907bef584d9277a8e4/README.md#use", license: "Apache-2.0", licenseUrl: "https://www.apache.org/licenses/LICENSE-2.0.txt", preprocessing: "Pinned SentencePiece tokenizer; state truncates to 256 tokens; questions and options use the artifact's marker-token span interface." },
    candidates: PRECISIONS,
    qualityGate: "All checked-in typed questions must return finite normalized distributions, the expected top choice/yes-no decision, and an ordered-score value within the fixture tolerance; outputs must be non-degenerate.",
    environment: "Isolated headless Google Chrome 151 on macOS; Transformers.js 4.2.0",
    invalidatedBy: ["artifact revision change", "runtime version change", "fixture or quality-gate change", "browser/backend behavior change"],
    results: [{"precision":"fp32","device":"wasm","status":"pass","runtime":{"loadMs":182152.3,"inferenceMs":1475.1,"totalMs":185097.3,"grade":"slow","questionsPerSecond":2.0338},"quality":{"normalized":true,"finite":true,"expected":true,"nondegenerate":true,"grade":"typed-decision-pass"},"cleanup":{"modelDisposed":true,"cacheClearedBeforeCase":true,"cacheClearedAfterCase":true,"deletedEntries":4}},{"precision":"fp32","device":"webgpu","status":"pass","runtime":{"loadMs":181271.3,"inferenceMs":445.4,"totalMs":183191.6,"grade":"good","questionsPerSecond":6.7355},"quality":{"normalized":true,"finite":true,"expected":true,"nondegenerate":true,"grade":"typed-decision-pass"},"cleanup":{"modelDisposed":true,"cacheClearedBeforeCase":true,"cacheClearedAfterCase":true,"deletedEntries":4}},{"precision":"fp16","device":"wasm","status":"pass","runtime":{"loadMs":110955.2,"inferenceMs":2716.9,"totalMs":115036.4,"grade":"slow","questionsPerSecond":1.1042},"quality":{"normalized":true,"finite":true,"expected":true,"nondegenerate":true,"grade":"typed-decision-pass"},"cleanup":{"modelDisposed":true,"cacheClearedBeforeCase":true,"cacheClearedAfterCase":true,"deletedEntries":4}},{"precision":"fp16","device":"webgpu","status":"pass","runtime":{"loadMs":97316.4,"inferenceMs":965.6,"totalMs":99586.3,"grade":"good","questionsPerSecond":3.1069},"quality":{"normalized":true,"finite":true,"expected":true,"nondegenerate":true,"grade":"typed-decision-pass"},"cleanup":{"modelDisposed":true,"cacheClearedBeforeCase":true,"cacheClearedAfterCase":true,"deletedEntries":4}},{"precision":"q4","device":"wasm","status":"fail","runtime":{"loadMs":null,"inferenceMs":null,"totalMs":64089.3,"grade":"failed"},"error":"Could not find a WASM implementation for GatherBlockQuantized(1).","cleanup":{"modelDisposed":true,"cacheClearedBeforeCase":true,"cacheClearedAfterCase":true,"deletedEntries":4}},{"precision":"q4","device":"webgpu","status":"pass","runtime":{"loadMs":58865.6,"inferenceMs":745.3,"totalMs":61101.1,"grade":"good","questionsPerSecond":4.0252},"quality":{"normalized":true,"finite":true,"expected":true,"nondegenerate":true,"grade":"typed-decision-pass"},"cleanup":{"modelDisposed":true,"cacheClearedBeforeCase":true,"cacheClearedAfterCase":true,"deletedEntries":4}},{"precision":"q4f16","device":"wasm","status":"fail","runtime":{"loadMs":null,"inferenceMs":null,"totalMs":43624.3,"grade":"failed"},"error":"No CPUExecutionProvider kernel was found for GatherBlockQuantized(1).","cleanup":{"modelDisposed":true,"cacheClearedBeforeCase":true,"cacheClearedAfterCase":true,"deletedEntries":4}},{"precision":"q4f16","device":"webgpu","status":"pass","runtime":{"loadMs":45219.2,"inferenceMs":387.9,"totalMs":47054.6,"grade":"good","questionsPerSecond":7.734},"quality":{"normalized":true,"finite":true,"expected":true,"nondegenerate":true,"grade":"typed-decision-pass"},"cleanup":{"modelDisposed":true,"cacheClearedBeforeCase":true,"cacheClearedAfterCase":true,"deletedEntries":4}}],
  },

  samples: [
  {
    "id": "standard",
    "label": "Try sample text",
    "description": "Standard sample for Open Jev DeBERTa V3 Large — Typed Decisions.",
    "userInput": {
      "state": "I was charged twice for the same order and nobody answers my emails. I want my money back now.",
      "questions": [
        {
          "type": "choice",
          "instructions": "Which product area is the message about?",
          "options": [
            "fees & charges",
            "pin & security",
            "refund & dispute",
            "top-up",
            "exchange & fiat",
            "atm & cash",
            "transfer",
            "card",
            "account & identity",
            "other"
          ]
        },
        {
          "type": "score",
          "instructions": "How positive is the sentiment of this message?",
          "options": [
            "very negative",
            "negative",
            "neutral",
            "positive",
            "very positive"
          ]
        },
        {
          "type": "noul",
          "instructions": "The customer is asking for a refund."
        }
      ]
    },
    "assets": [],
    "modelConfig": {},
    "generateConfig": {},
    "source": "Adapted from the pinned source model card usage example.",
    "license": "Apache-2.0",
    "fixture": {
      "file": "model-samples/open-jev-deberta-v3-large.json",
      "url": "https://f005.backblazeb2.com/file/hubters-webai-public/assets/benchmarks/open-jev-deberta-v3-large.json",
      "size": 819,
      "sha1": "bf1e5a03646050ac2c9b0fa51bf8f01b72f95ece"
    },
    "benchmarkReport": "benchmarks/open-jev-deberta-v3-large.json"
  }
],
};

const state = { initializing: false, initialized: false, generating: false };
let tokenizer = null;
let model = null;
let markers = null;
const capabilities = { supportedModes: ["webai"], supportedPrecisions: Object.keys(SUPPORTED), supportedPrecisionsDevicesMap: SUPPORTED, doesSupportStreamGeneration: false, externalInterrupt: true, workerVersion: "v4", transformersJsVersion: "4.2.0", cacheModelId: REPO };
const send = (requestId, type, data) => self.postMessage({ requestId, type, data });

function configure(config = {}) {
  env.allowRemoteModels = config.allowRemoteModels ?? true;
  env.allowLocalModels = config.allowLocalModels ?? false;
  env.useBrowserCache = config.useBrowserCache ?? true;
  if (config.remoteHost) env.remoteHost = config.remoteHost;
  if (config.remotePathTemplate) env.remotePathTemplate = config.remotePathTemplate;
  if (config.localModelPath) env.localModelPath = config.localModelPath;
}

function validateInit(data, supported = SUPPORTED) {
  const precision = data?.precision;
  const device = data?.device;
  const entry = supported[precision];
  if (!entry) throw Error(`Unsupported precision: ${precision}`);
  const devices = entry.supportedDevices ?? entry.candidateDevices;
  if (!devices.includes(device)) throw Error(`Precision ${precision} is only supported on: ${devices.join(", ")}`);
  if (device === "webgpu" && !self.navigator?.gpu) throw Error("WebGPU is unavailable");
  return { precision, device };
}

async function clear() {
  if (model?.dispose) await model.dispose();
  model = null;
  tokenizer = null;
  markers = null;
  state.initialized = false;
}

function tokenIds(text) {
  return Array.from(tokenizer(text, { add_special_tokens: false }).input_ids.data, Number);
}

async function load(data, requestId, supported = SUPPORTED) {
  const { precision, device } = validateInit(data, supported);
  await clear();
  const progress_callback = value => send(requestId, "downloadProgress", value);
  tokenizer = await AutoTokenizer.from_pretrained(REPO, { revision: REV, progress_callback });
  model = await AutoModel.from_pretrained(REPO, { revision: REV, dtype: precision, device, progress_callback });
  markers = Object.fromEntries(["[CLS]", "[SEP]", "[STATE]", "[Q]", "[OPT]"].map(value => [value, tokenIds(value)[0]]));
  if (Object.values(markers).some(value => !Number.isInteger(value))) throw Error("Artifact marker tokens are unavailable");
}

function validatedConfig(data) {
  const known = new Set(MANIFEST.config.model.map(field => field.name));
  const supplied = data?.modelConfig;
  if (supplied !== undefined && (!supplied || typeof supplied !== "object" || Array.isArray(supplied))) throw Error("modelConfig must be an object");
  for (const name of Object.keys(supplied ?? {})) if (!known.has(name)) throw Error(`Unsupported modelConfig option: ${name}`);
  if (data?.generateConfig !== undefined && (!data.generateConfig || typeof data.generateConfig !== "object" || Array.isArray(data.generateConfig) || Object.keys(data.generateConfig).length)) throw Error("generateConfig does not accept options");
  const config = { max_length: 512, state_max_tokens: 256, temperature: 1.05, ...supplied };
  if (!Number.isInteger(config.max_length) || config.max_length < 8 || config.max_length > 512) throw Error("max_length must be an integer from 8 through 512");
  if (!Number.isInteger(config.state_max_tokens) || config.state_max_tokens < 1 || config.state_max_tokens > 256) throw Error("state_max_tokens must be an integer from 1 through 256");
  if (!Number.isFinite(config.temperature) || config.temperature < 0.01 || config.temperature > 10) throw Error("temperature must be from 0.01 through 10");
  return config;
}

function validateQuestions(value) {
  if (!Array.isArray(value) || !value.length) throw Error("userInput.questions must be a non-empty array");
  return value.map((question, index) => {
    if (!question || typeof question !== "object" || Array.isArray(question)) throw Error(`questions[${index}] must be an object`);
    if (!["choice", "score", "noul"].includes(question.type)) throw Error(`questions[${index}].type must be choice, score, or noul`);
    if (typeof question.instructions !== "string" || !question.instructions.trim()) throw Error(`questions[${index}].instructions must be a non-empty string`);
    const options = question.type === "noul" ? ["no", "yes"] : question.options;
    if (!Array.isArray(options) || options.some(option => typeof option !== "string" || !option.trim())) throw Error(`questions[${index}].options must contain non-empty strings`);
    const maximum = question.type === "score" ? 10 : 255;
    if (options.length < 2 || options.length > maximum) throw Error(`questions[${index}].options must contain 2 through ${maximum} values`);
    if (new Set(options).size !== options.length) throw Error(`questions[${index}].options must be unique`);
    if (question.type === "noul" && question.options !== undefined && (question.options.length !== 2 || question.options[0] !== "no" || question.options[1] !== "yes")) throw Error(`questions[${index}].options, when present for noul, must be ["no", "yes"]`);
    return { type: question.type, instructions: question.instructions.trim(), options: options.map(option => option.trim()) };
  });
}

const int64 = (values, dims) => new Tensor("int64", BigInt64Array.from(values, BigInt), dims);
function softmax(values, temperature) {
  const maximum = Math.max(...values);
  const exponentials = values.map(value => Math.exp((value - maximum) / temperature));
  const total = exponentials.reduce((sum, value) => sum + value, 0);
  return exponentials.map(value => value / total);
}

async function generate(data) {
  const input = data?.userInput;
  if (!input || typeof input !== "object" || Array.isArray(input)) throw Error("userInput must be an object");
  if (typeof input.state !== "string" || !input.state.trim()) throw Error("userInput.state must be a non-empty string");
  const questions = validateQuestions(input.questions);
  const config = validatedConfig(data);
  const tokens = [markers["[CLS]"], markers["[STATE]"], ...tokenIds(input.state.trim()).slice(0, config.state_max_tokens)];
  const segments = tokens.map(() => -1);
  const pairQuestions = [];
  const pairOptions = [];
  const groups = [];
  const totalPairs = questions.reduce((sum, question) => sum + question.options.length, 0);
  questions.forEach((question, questionIndex) => {
    const instructionTokens = tokenIds(question.instructions);
    tokens.push(markers["[Q]"], ...instructionTokens);
    segments.push(-1, ...instructionTokens.map(() => totalPairs + questionIndex));
    groups.push(question.options.map(option => {
      const optionTokens = tokenIds(option);
      const pairIndex = pairOptions.length;
      tokens.push(markers["[OPT]"], ...optionTokens);
      segments.push(-1, ...optionTokens.map(() => pairIndex));
      pairQuestions.push(totalPairs + questionIndex);
      pairOptions.push(pairIndex);
      return pairIndex;
    }));
  });
  tokens.push(markers["[SEP]"]);
  segments.push(-1);
  if (tokens.length > config.max_length) throw Error(`Encoded request uses ${tokens.length} tokens; reduce the state, questions, or options to fit max_length ${config.max_length}`);
  const output = await model({
    input_ids: int64(tokens, [1, tokens.length]),
    attention_mask: int64(tokens.map(() => 1), [1, tokens.length]),
    seg: int64(segments, [1, segments.length]),
    pair_q: int64(pairQuestions, [1, pairQuestions.length]),
    pair_opt: int64(pairOptions, [1, pairOptions.length]),
  });
  const logits = Array.from(output.logits.to("float32").data, Number);
  if (logits.length !== pairOptions.length || logits.some(value => !Number.isFinite(value))) throw Error("Model returned invalid decision logits");
  const decisions = groups.map((group, index) => {
    const question = questions[index];
    const probabilities = softmax(group.map(pair => logits[pair]), config.temperature);
    const distribution = question.options.map((option, optionIndex) => ({ option, probability: probabilities[optionIndex] }));
    const confidence = Math.max(...probabilities);
    if (question.type === "choice") return { type: question.type, choice: question.options[probabilities.indexOf(confidence)], confidence, probabilities: distribution };
    if (question.type === "score") return { type: question.type, score: probabilities.reduce((sum, probability, optionIndex) => sum + probability * optionIndex, 0), confidence, probabilities: distribution };
    return { type: question.type, noul: probabilities[1], confidence, probabilities: distribution };
  });
  return { decisions };
}

self.addEventListener("message", async ({ data: message = {} }) => {
  const { requestId, type, data } = message;
  try {
    if (type === "checkModelSupports") { configure(data?.workerConfig); send(requestId, type, { ...capabilities, manifest: MANIFEST }); return; }
    if (type === "init" || type === "download") {
      if (state.initializing || state.generating) throw Error("Worker is busy");
      state.initializing = true;
      state.initialized = false;
      try {
        const precision = data?.precision;
        const device = type === "download" ? SUPPORTED[precision]?.supportedDevices[0] : data?.device;
        await load({ ...data, precision, device }, requestId);
        if (type === "download") await clear(); else state.initialized = true;
        send(requestId, type, { status: "success" });
      } finally { state.initializing = false; }
      return;
    }
    if (type === "generate") {
      if (state.initializing || !state.initialized || !model) throw Error("Call init before generate");
      if (state.generating) throw Error("Generation already in progress");
      state.generating = true;
      try { send(requestId, "generated", { status: "success", result: await generate(data) }); }
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
