import { AutoTokenizer, PreTrainedModel, Tensor, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0";

const ID = "laya-typed-decisions";
const REPO = "m1rhan/laya-typed-decisions-ONNX";
const REV = "7388ca40228f08fd72fb3d2fdbd29153e9bc63d4";
const SHARED_BYTES = 3585649;
const make = (weightsSize, dtype, key) => ({ weightsSize, size: weightsSize + SHARED_BYTES, dtype, modelKeys: [key], candidateDevices: ["wasm", "webgpu"] });
const PRECISIONS = {
  q4: make(448638677, "q4", "model_q4.onnx"),
};
const SUPPORTED = {
  q4: { ...PRECISIONS.q4, supportedDevices: ["wasm", "webgpu"] },
};

const MANIFEST = {
  contractVersion: "1.0",
  model: {
    id: ID,
    displayName: "Laya Typed Decisions — Q4",
    provider: "Convai Innovations",
    providerUrl: "https://huggingface.co/convaiinnovations",
    license: "Apache-2.0",
    licenseUrl: "https://www.apache.org/licenses/LICENSE-2.0.txt",
    lastUpdated: "2026-09-20T01:43:36.000Z",
    sourceRepository: "convaiinnovations/laya (typed-decisions checkpoint)",
    sourceRevision: "1c5edc17a7acd8701df6fc341c0d179f1c62c982",
    repository: REPO,
    artifactRevision: REV,
    artifactLastUpdated: "2026-09-22T12:10:34.000Z",
    task: "typed-decision-classification",
    description: "An English Laya System One specialist fine-tuned for typed decisions in agent observability, customer service, invoice processing, and security incidents, exported as a Q4 ONNX model for local Transformers.js inference.",
    intendedUses: [
      "Classify and score English agent traces, customer-service cases, invoices, and security incidents.",
      "Return application-provided choices, ordered ratings, and yes/no probabilities for human-reviewed workflow automation.",
      "Prototype deterministic structured-decision interfaces in a browser.",
    ],
    limitations: [
      "This open model is not TypeSafe AI's proprietary Jev model and is not affiliated with TypeSafe AI.",
      "The source reports 0.766 accuracy on its four synthetic workflow families, but performance outside those English domains may fall back to the weaker general checkpoint behavior.",
      "The source reports ECE 0.213 and warns that the checkpoint remains over-confident; recalibrate probabilities on held-out application data before thresholding them.",
      "The Q4 publisher reports six-of-six argmax parity cases, not task accuracy; conversion parity does not prove that a decision is correct.",
      "The source model's option-count calibration includes a value below the SDK's safe range for 11+ choices; this adapter clamps it to 0.5 and treats affected confidence as uncalibrated.",
      "The model is English-only and choice questions should remain below roughly 20 options because they share a fixed question-head token budget.",
      "Long states are truncated to fit the 1,024-token context; option text is truncated within a fixed 256-token question-head budget.",
      "It cannot generate explanations, and its outputs must not be the sole basis for safety-critical or consequential decisions.",
      "The Q4 first download is approximately 452 MB including tokenizer assets and may not persist in every browser cache profile.",
    ],
  },
  runtime: { engine: { name: "Transformers.js", version: "4.2.0" }, precisions: SUPPORTED },
  input: {
    description: "One non-empty text state and one or more typed questions. Each question is encoded as a separate model row against the same state.",
    alternatives: [{
      name: "English typed decisions",
      fields: [
        { name: "state", type: "string", required: true, minLength: 1, description: "English workflow state; truncated at the token boundary when needed." },
        { name: "questions", type: "Array<TypedDecisionQuestion>", required: true, minItems: 1, items: { type: "object", description: "type is choice, score, or noul; instructions is non-empty; choice has 2–255 options, score has 2–10 ordered options, and noul always uses no/yes." } },
      ],
    }],
  },
  config: {
    model: [
      { name: "max_length", type: "integer", default: 1024, minimum: 32, maximum: 1024, description: "Maximum encoded tokens for each question/state row." },
      { name: "head_max_length", type: "integer", default: 256, minimum: 32, maximum: 256, description: "Question-and-options token budget before state tokens are appended." },
    ],
    generation: [],
  },
  output: {
    description: "One typed result per question, preserving input order and including the full option distribution and Laya action probability.",
    fields: [{ name: "decisions", type: "Array<{type:string,choice?:string,score?:number,noul?:number,confidence:number,answerConfidence:number,actProbability:number,probabilities:Array<{option:string,probability:number}>}>" }],
    example: { decisions: [{ type: "choice", choice: "refund", confidence: 0.6, answerConfidence: 0.8, actProbability: 0.9, probabilities: [{ option: "refund", probability: 0.8 }, { option: "other", probability: 0.2 }] }] },
  },
  operations: ["checkModelSupports", "init", "download", "generate", "clearMemory"],
  benchmark: {
    report: "benchmarks/laya-typed-decisions.json",
    testedAt: "2026-09-25T09:51:15.016Z",
    fixture: { file: "laya-typed-decisions.json", url: "https://f005.backblazeb2.com/file/hubters-webai-public/assets/benchmarks/laya-typed-decisions.json", size: 2242, sha1: "cee490dc2efc280f3d781dd6c7ec47c6d23ba737", source: "WebAI-authored regression cases representing the source model's four intended workflow families.", license: "CC0-1.0", licenseUrl: "https://creativecommons.org/publicdomain/zero/1.0/", preprocessing: "Pinned ModernBERT tokenizer; each question/options prefix uses Laya's marker format and the shared state is truncated to the remaining context." },
    candidates: PRECISIONS,
    qualityGate: "Every agent-observability, customer-service, invoice, and security case must return finite normalized non-degenerate distributions and the checked-in expected choice/yes-no result or ordered-score interval.",
    environment: "Isolated headless Google Chrome on macOS; Transformers.js 4.2.0",
    invalidatedBy: ["artifact revision change", "runtime version change", "fixture or quality-gate change", "browser/backend behavior change"],
    results: [{"precision":"q4","device":"wasm","status":"pass","runtime":{"loadMs":53499.4,"inferenceMs":9939.8,"totalMs":64657.9,"grade":"very-slow","questionsPerSecond":0.8048},"quality":{"grade":"typed-decision-pass","cases":[{"domain":"agent-observability","type":"choice","pass":true},{"domain":"agent-observability","type":"noul","pass":true},{"domain":"customer-service","type":"choice","pass":true},{"domain":"customer-service","type":"noul","pass":true},{"domain":"invoice-processing","type":"choice","pass":true},{"domain":"invoice-processing","type":"score","pass":true},{"domain":"security-incidents","type":"choice","pass":true},{"domain":"security-incidents","type":"noul","pass":true}]},"cleanup":{"modelDisposed":true,"cacheClearedBeforeCase":true,"cacheClearedAfterCase":true,"deletedEntries":3}},{"precision":"q4","device":"webgpu","status":"pass","runtime":{"loadMs":53566.3,"inferenceMs":938.7,"totalMs":55818.6,"grade":"good","questionsPerSecond":8.5224},"quality":{"grade":"typed-decision-pass","cases":[{"domain":"agent-observability","type":"choice","pass":true},{"domain":"agent-observability","type":"noul","pass":true},{"domain":"customer-service","type":"choice","pass":true},{"domain":"customer-service","type":"noul","pass":true},{"domain":"invoice-processing","type":"choice","pass":true},{"domain":"invoice-processing","type":"score","pass":true},{"domain":"security-incidents","type":"choice","pass":true},{"domain":"security-incidents","type":"noul","pass":true}]},"cleanup":{"modelDisposed":true,"cacheClearedBeforeCase":true,"cacheClearedAfterCase":true,"deletedEntries":3}}],
  },
};

const state = { initializing: false, initialized: false, generating: false };
let tokenizer = null;
let model = null;
const capabilities = { supportedModes: ["webai"], supportedPrecisions: Object.keys(SUPPORTED), supportedPrecisionsDevicesMap: SUPPORTED, doesSupportStreamGeneration: false, externalInterrupt: true, workerVersion: "v4", transformersJsVersion: "4.2.0", cacheModelId: REPO };
const send = (requestId, type, data) => self.postMessage({ requestId, type, data });
const QTYPES = { choice: 0, score: 1, noul: 2 };
const SPECIAL = { pad: 50283, sep: 50282, cls: 50281, mask: 50284, maskToken: "[MASK]" };
const TEMPERATURE = [1.0148024559020996, 1.0374259948730469, 1.0575125217437744];
const TEMPERATURE_BY_OPTIONS = { "choice:3-5": 1.7601518630981445, "choice:6-10": 1.0000158548355103, "score:3-5": 1.2514300346374512, "noul:2": 1.983399510383606, "choice:11+": 0.5, "choice:2": 1.9063563346862793 };

function configure(config = {}) {
  env.allowRemoteModels = config.allowRemoteModels ?? true;
  env.allowLocalModels = config.allowLocalModels ?? false;
  env.useBrowserCache = config.useBrowserCache ?? true;
  if (config.remoteHost) env.remoteHost = config.remoteHost;
  if (config.remotePathTemplate) env.remotePathTemplate = config.remotePathTemplate;
  if (config.localModelPath) env.localModelPath = config.localModelPath;
}

function validateInit(data, supported = SUPPORTED) {
  const precision = data?.precision, device = data?.device, entry = supported[precision];
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
  state.initialized = false;
}

async function load(data, requestId, supported = SUPPORTED) {
  const { precision, device } = validateInit(data, supported);
  await clear();
  const progress_callback = value => send(requestId, "downloadProgress", value);
  tokenizer = await AutoTokenizer.from_pretrained(REPO, { revision: REV, progress_callback });
  model = await PreTrainedModel.from_pretrained(REPO, { revision: REV, dtype: precision, device, model_file_name: "model", subfolder: "onnx", progress_callback });
}

function tokenIds(text) { return Array.from(tokenizer(text, { add_special_tokens: false }).input_ids.data, Number); }
function optionsFor(question) {
  if (question.type === "choice") return question.options;
  if (question.type === "score") return question.options.map((option, index) => `level ${index}: ${option}`);
  return ["false: no, the statement does not hold", "true: yes, the statement holds"];
}
function validateQuestions(value) {
  if (!Array.isArray(value) || !value.length) throw Error("userInput.questions must be a non-empty array");
  return value.map((question, index) => {
    if (!question || typeof question !== "object" || Array.isArray(question)) throw Error(`questions[${index}] must be an object`);
    if (!["choice", "score", "noul"].includes(question.type)) throw Error(`questions[${index}].type must be choice, score, or noul`);
    if (typeof question.instructions !== "string" || !question.instructions.trim()) throw Error(`questions[${index}].instructions must be a non-empty string`);
    const options = question.type === "noul" ? ["false", "true"] : question.options;
    if (!Array.isArray(options) || options.some(option => typeof option !== "string" || !option.trim())) throw Error(`questions[${index}].options must contain non-empty strings`);
    const maximum = question.type === "score" ? 10 : 255;
    if (options.length < 2 || options.length > maximum) throw Error(`questions[${index}].options must contain 2 through ${maximum} values`);
    if (new Set(options).size !== options.length) throw Error(`questions[${index}].options must be unique`);
    if (question.type === "noul" && question.options !== undefined && (question.options.length !== 2 || question.options[0] !== "false" || question.options[1] !== "true")) throw Error(`questions[${index}].options, when present for noul, must be ["false", "true"]`);
    return { type: question.type, instructions: question.instructions.trim(), options: options.map(option => option.trim()) };
  });
}
function validatedConfig(data) {
  const supplied = data?.modelConfig;
  if (supplied !== undefined && (!supplied || typeof supplied !== "object" || Array.isArray(supplied))) throw Error("modelConfig must be an object");
  for (const name of Object.keys(supplied ?? {})) if (!["max_length", "head_max_length"].includes(name)) throw Error(`Unsupported modelConfig option: ${name}`);
  if (data?.generateConfig !== undefined && (!data.generateConfig || typeof data.generateConfig !== "object" || Array.isArray(data.generateConfig) || Object.keys(data.generateConfig).length)) throw Error("generateConfig does not accept options");
  const config = { max_length: 1024, head_max_length: 256, ...supplied };
  if (!Number.isInteger(config.max_length) || config.max_length < 32 || config.max_length > 1024) throw Error("max_length must be an integer from 32 through 1024");
  if (!Number.isInteger(config.head_max_length) || config.head_max_length < 32 || config.head_max_length > 256 || config.head_max_length >= config.max_length) throw Error("head_max_length must be an integer from 32 through 256 and smaller than max_length");
  return config;
}

function buildRow(question, sharedStateIds, config) {
  const rendered = optionsFor(question), instruction = question.instructions.split(SPECIAL.maskToken).join(" ");
  let headIds = tokenIds(`${question.type} question: ${instruction}`);
  let optionIds = rendered.map(option => [SPECIAL.mask, ...tokenIds(` ${option.split(SPECIAL.maskToken).join(" ")}`).slice(0, 48)]);
  let budget = config.head_max_length - optionIds.reduce((sum, option) => sum + option.length, 0);
  if (budget < 16) {
    const perOption = Math.max(4, Math.floor((config.head_max_length - 16) / optionIds.length));
    optionIds = optionIds.map(option => option.slice(0, perOption));
    budget = config.head_max_length - optionIds.reduce((sum, option) => sum + option.length, 0);
  }
  headIds = headIds.slice(0, Math.max(8, budget));
  const ids = [SPECIAL.cls, ...headIds, SPECIAL.sep], markers = [];
  for (const option of optionIds) { markers.push(ids.length); ids.push(...option); }
  ids.push(SPECIAL.sep);
  if (markers.length !== rendered.length) throw Error("Question options exceed the question-head token budget");
  const room = Math.max(0, config.max_length - ids.length - 1);
  ids.push(...sharedStateIds.slice(0, room), SPECIAL.sep);
  return { ids: ids.slice(0, config.max_length), markers, qtype: QTYPES[question.type] };
}
const int64 = (values, dims) => new Tensor("int64", BigInt64Array.from(values, BigInt), dims);
const boolTensor = (values, dims) => new Tensor("bool", Uint8Array.from(values, value => value ? 1 : 0), dims);
function softmax(values, temperature = 1) { const maximum = Math.max(...values), exponentials = values.map(value => Math.exp((value - maximum) / temperature)), total = exponentials.reduce((sum, value) => sum + value, 0); return exponentials.map(value => value / total); }
function temperatureFor(qtype, count) { const size = count <= 2 ? "2" : count <= 5 ? "3-5" : count <= 10 ? "6-10" : "11+"; return TEMPERATURE_BY_OPTIONS[`${["choice", "score", "noul"][qtype]}:${size}`] ?? TEMPERATURE[qtype]; }
function entropyConfidence(probabilities) { if (probabilities.length < 2) return 1; const entropy = -probabilities.reduce((sum, value) => sum + value * Math.log(Math.max(value, 1e-12)), 0); return Math.min(1, Math.max(0, 1 - entropy / Math.log(probabilities.length))); }

async function generate(data) {
  const input = data?.userInput;
  if (!input || typeof input !== "object" || Array.isArray(input)) throw Error("userInput must be an object");
  if (typeof input.state !== "string" || !input.state.trim()) throw Error("userInput.state must be a non-empty string");
  const questions = validateQuestions(input.questions), config = validatedConfig(data), sharedStateIds = tokenIds(input.state.trim().split(SPECIAL.maskToken).join(" "));
  const rows = questions.map(question => buildRow(question, sharedStateIds, config)), rawResults = [];
  // The published graph has a fixed batch dimension of one. Preserve the
  // contract's multi-question request and isolate that artifact detail here.
  for (const row of rows) {
    const sequenceLength = row.ids.length, optionCount = row.markers.length;
    const output = await model({
      input_ids: int64(row.ids, [1, sequenceLength]),
      attention_mask: int64(Array(sequenceLength).fill(1), [1, sequenceLength]),
      marker_pos: int64(row.markers, [1, optionCount]),
      marker_mask: boolTensor(Array(optionCount).fill(true), [1, optionCount]),
      qtype: int64([row.qtype], [1]),
    });
    const logitsTensor = output.logits, actionTensor = output.act_probs ?? output.act_logits;
    if (!logitsTensor || !actionTensor) throw Error(`Model returned unexpected outputs: ${Object.keys(output).join(", ")}`);
    rawResults.push({
      logits: Array.from(logitsTensor.to("float32").data, Number).slice(0, optionCount),
      actionValues: Array.from(actionTensor.to("float32").data, Number),
      actionIsProbability: Boolean(output.act_probs),
      actionWidth: actionTensor.dims.at(-1),
    });
  }
  const decisions = questions.map((question, rowIndex) => {
    const count = rows[rowIndex].markers.length, raw = rawResults[rowIndex], probabilities = softmax(raw.logits, temperatureFor(rows[rowIndex].qtype, count));
    if (probabilities.some(value => !Number.isFinite(value))) throw Error("Model returned invalid decision probabilities");
    let actProbability;
    if (raw.actionWidth === 2) actProbability = raw.actionIsProbability ? raw.actionValues[0] : softmax(raw.actionValues.slice(0, 2))[0];
    else actProbability = raw.actionValues[0];
    const distribution = question.options.map((option, index) => ({ option, probability: probabilities[index] }));
    const answerConfidence = Math.max(...probabilities), base = { type: question.type, confidence: entropyConfidence(probabilities), answerConfidence, actProbability, probabilities: distribution };
    if (question.type === "choice") return { ...base, choice: question.options[probabilities.indexOf(answerConfidence)] };
    if (question.type === "score") return { ...base, score: probabilities.reduce((sum, probability, index) => sum + probability * index, 0) };
    return { ...base, noul: probabilities[1] };
  });
  return { decisions };
}

self.addEventListener("message", async ({ data: message = {} }) => {
  const { requestId, type, data } = message;
  try {
    if (type === "checkModelSupports") { configure(data?.workerConfig); send(requestId, type, { ...capabilities, manifest: MANIFEST }); return; }
    if (type === "init" || type === "download") {
      if (state.initializing || state.generating) throw Error("Worker is busy");
      state.initializing = true; state.initialized = false;
      try { const precision = data?.precision, device = type === "download" ? SUPPORTED[precision]?.supportedDevices[0] : data?.device; await load({ ...data, precision, device }, requestId); if (type === "download") await clear(); else state.initialized = true; send(requestId, type, { status: "success" }); }
      finally { state.initializing = false; }
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
    if (type === "clearMemory") { if (state.initializing || state.generating) throw Error("Worker is busy"); state.initializing = true; try { await clear(); send(requestId, type, { status: "success" }); } finally { state.initializing = false; } return; }
    throw Error(`Unknown operation: ${type}`);
  } catch (error) { send(requestId, "error", { message: error instanceof Error ? error.message : String(error), context: type }); }
});

self.postMessage({ type: "worker initialized", data: { success: true, workerVersion: "v4", modelId: ID } });
