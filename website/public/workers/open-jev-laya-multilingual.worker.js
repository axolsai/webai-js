import { AutoTokenizer, PreTrainedModel, Tensor, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0";

const ID = "open-jev-laya-multilingual";
const REPO = "killkli/open-jev-laya-multilingual-onnx";
const REV = "643219b2b3a112ff1c40fb1143f050aa871f76f7";
const SHARED_BYTES = 34363920;
const make = (weightsSize, dtype, key) => ({ weightsSize, size: weightsSize + SHARED_BYTES, dtype, modelKeys: [key], candidateDevices: ["wasm", "webgpu"] });
const PRECISIONS = {
  fp32: make(1290793955, "fp32", "laya.onnx"),
  fp16: make(646982318, "fp16", "laya_fp16.onnx"),
};
const SUPPORTED = {
  fp32: { ...PRECISIONS.fp32, supportedDevices: ["wasm", "webgpu"] },
  fp16: { ...PRECISIONS.fp16, supportedDevices: ["wasm", "webgpu"] },
};

const MANIFEST = {
  contractVersion: "1.0",
  model: {
    id: ID,
    displayName: "Open Jev Laya Multilingual — Typed Decisions",
    provider: "Convai Innovations",
    providerUrl: "https://huggingface.co/convaiinnovations",
    license: "Apache-2.0",
    licenseUrl: "https://www.apache.org/licenses/LICENSE-2.0.txt",
    lastUpdated: "2026-09-23T08:09:01.000Z",
    sourceRepository: "convaiinnovations/laya (multilingual checkpoint)",
    sourceRevision: "5e7b2b1b8ca2ecdd3f2322d94069c9b6ce7e844b",
    repository: REPO,
    artifactRevision: REV,
    artifactLastUpdated: "2026-09-23T12:31:22.000Z",
    task: "multilingual-typed-decision-classification",
    description: "A multilingual Laya System One decision model exported for local Transformers.js inference. It returns distributions for categorical choices, ordered scores, and yes/no questions without generating text.",
    intendedUses: [
      "Route and classify multilingual workflow inputs among application-provided choices.",
      "Estimate ordered ratings and yes/no probabilities for human-reviewed multilingual workflows.",
      "Prototype deterministic structured-decision interfaces in a browser.",
    ],
    limitations: [
      "This open model is not TypeSafe AI's proprietary Jev model and is not affiliated with TypeSafe AI.",
      "The source describes support for more than 100 languages, but accuracy and calibration vary by language and domain and require application-specific evaluation.",
      "The ONNX publisher reports fp16/fp32 parity, not task accuracy; parity does not prove that a decision is correct.",
      "The exported checkpoint uses neutral temperature values and its decision probabilities have not been independently recalibrated for this adapter's workloads.",
      "Long states are truncated to fit the 1,024-token context; option text is truncated within a fixed 256-token question-head budget.",
      "It cannot generate explanations, and its outputs must not be the sole basis for safety-critical or consequential decisions.",
      "The fp16 first download is approximately 681 MB including tokenizer assets and may not persist in every browser cache profile.",
    ],
  },
  runtime: { engine: { name: "Transformers.js", version: "4.2.0" }, precisions: SUPPORTED },
  input: {
    description: "One non-empty text state and one or more typed questions. Each question is encoded as a separate model row against the same state.",
    alternatives: [{
      name: "multilingual typed decisions",
      fields: [
        { name: "state", type: "string", required: true, minLength: 1, description: "Text in any supported language; truncated at the token boundary when needed." },
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
    report: "benchmarks/open-jev-laya-multilingual.json",
    testedAt: "2026-09-25T08:11:04.378Z",
    fixture: { file: "open-jev-laya-multilingual.json", url: "https://f005.backblazeb2.com/file/hubters-webai-public/assets/benchmarks/open-jev-laya-multilingual.json", size: 1368, sha1: "ff4a5039dbc0c37a52717f0c82420ff45e0a2803", source: "WebAI-authored multilingual regression cases with unambiguous gold labels.", license: "CC0-1.0", licenseUrl: "https://creativecommons.org/publicdomain/zero/1.0/", preprocessing: "Pinned mmBERT tokenizer; each question/options prefix uses Laya's marker format and the shared state is truncated to the remaining context." },
    candidates: PRECISIONS,
    qualityGate: "Every English, Chinese, and Spanish case must return finite normalized non-degenerate distributions and the checked-in expected choice/yes-no result or ordered-score interval.",
    environment: "Isolated headless Google Chrome on macOS; Transformers.js 4.2.0",
    invalidatedBy: ["artifact revision change", "runtime version change", "fixture or quality-gate change", "browser/backend behavior change"],
    results: [{"precision":"fp32","device":"wasm","status":"pass","runtime":{"loadMs":141147.9,"inferenceMs":1615.8,"totalMs":144071.1,"grade":"good","questionsPerSecond":3.7133},"quality":{"grade":"multilingual-typed-decision-pass","cases":[{"language":"en","type":"choice","pass":true},{"language":"en","type":"noul","pass":true},{"language":"zh","type":"choice","pass":true},{"language":"zh","type":"noul","pass":true},{"language":"es","type":"score","pass":true},{"language":"es","type":"noul","pass":true}]},"cleanup":{"modelDisposed":true,"cacheClearedBeforeCase":true,"cacheClearedAfterCase":true,"deletedEntries":3}},{"precision":"fp32","device":"webgpu","status":"pass","runtime":{"loadMs":132489.8,"inferenceMs":524.2,"totalMs":134322.4,"grade":"excellent","questionsPerSecond":11.446},"quality":{"grade":"multilingual-typed-decision-pass","cases":[{"language":"en","type":"choice","pass":true},{"language":"en","type":"noul","pass":true},{"language":"zh","type":"choice","pass":true},{"language":"zh","type":"noul","pass":true},{"language":"es","type":"score","pass":true},{"language":"es","type":"noul","pass":true}]},"cleanup":{"modelDisposed":true,"cacheClearedBeforeCase":true,"cacheClearedAfterCase":true,"deletedEntries":3}},{"precision":"fp16","device":"wasm","status":"pass","runtime":{"loadMs":72630.7,"inferenceMs":1860.8,"totalMs":75768.9,"grade":"good","questionsPerSecond":3.2244},"quality":{"grade":"multilingual-typed-decision-pass","cases":[{"language":"en","type":"choice","pass":true},{"language":"en","type":"noul","pass":true},{"language":"zh","type":"choice","pass":true},{"language":"zh","type":"noul","pass":true},{"language":"es","type":"score","pass":true},{"language":"es","type":"noul","pass":true}]},"cleanup":{"modelDisposed":true,"cacheClearedBeforeCase":true,"cacheClearedAfterCase":true,"deletedEntries":3}},{"precision":"fp16","device":"webgpu","status":"pass","runtime":{"loadMs":79046.5,"inferenceMs":1648.2,"totalMs":81778.8,"grade":"good","questionsPerSecond":3.6403},"quality":{"grade":"multilingual-typed-decision-pass","cases":[{"language":"en","type":"choice","pass":true},{"language":"en","type":"noul","pass":true},{"language":"zh","type":"choice","pass":true},{"language":"zh","type":"noul","pass":true},{"language":"es","type":"score","pass":true},{"language":"es","type":"noul","pass":true}]},"cleanup":{"modelDisposed":true,"cacheClearedBeforeCase":true,"cacheClearedAfterCase":true,"deletedEntries":3}}],
  },
};

const state = { initializing: false, initialized: false, generating: false };
let tokenizer = null;
let model = null;
const capabilities = { supportedModes: ["webai"], supportedPrecisions: Object.keys(SUPPORTED), supportedPrecisionsDevicesMap: SUPPORTED, doesSupportStreamGeneration: false, externalInterrupt: true, workerVersion: "v4", transformersJsVersion: "4.2.0", cacheModelId: REPO };
const send = (requestId, type, data) => self.postMessage({ requestId, type, data });
const QTYPES = { choice: 0, score: 1, noul: 2 };
const SPECIAL = { pad: 0, sep: 1, cls: 2, mask: 4, maskToken: "<mask>" };

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
  model = await PreTrainedModel.from_pretrained(REPO, { revision: REV, dtype: precision, device, model_file_name: "laya", subfolder: "onnx", progress_callback });
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
function softmax(values) { const maximum = Math.max(...values), exponentials = values.map(value => Math.exp(value - maximum)), total = exponentials.reduce((sum, value) => sum + value, 0); return exponentials.map(value => value / total); }
function entropyConfidence(probabilities) { if (probabilities.length < 2) return 1; const entropy = -probabilities.reduce((sum, value) => sum + value * Math.log(Math.max(value, 1e-12)), 0); return Math.min(1, Math.max(0, 1 - entropy / Math.log(probabilities.length))); }

async function generate(data) {
  const input = data?.userInput;
  if (!input || typeof input !== "object" || Array.isArray(input)) throw Error("userInput must be an object");
  if (typeof input.state !== "string" || !input.state.trim()) throw Error("userInput.state must be a non-empty string");
  const questions = validateQuestions(input.questions), config = validatedConfig(data), sharedStateIds = tokenIds(input.state.trim().split(SPECIAL.maskToken).join(" "));
  const rows = questions.map(question => buildRow(question, sharedStateIds, config));
  const sequenceLength = Math.max(...rows.map(row => row.ids.length)), optionCount = Math.max(...rows.map(row => row.markers.length));
  const inputIds = rows.flatMap(row => [...row.ids, ...Array(sequenceLength - row.ids.length).fill(SPECIAL.pad)]);
  const attentionMask = rows.flatMap(row => [...Array(row.ids.length).fill(1), ...Array(sequenceLength - row.ids.length).fill(0)]);
  const markerPositions = rows.flatMap(row => [...row.markers, ...Array(optionCount - row.markers.length).fill(0)]);
  const markerMask = rows.flatMap(row => [...row.markers.map(() => true), ...Array(optionCount - row.markers.length).fill(false)]);
  const output = await model({
    input_ids: int64(inputIds, [rows.length, sequenceLength]),
    attention_mask: int64(attentionMask, [rows.length, sequenceLength]),
    marker_pos: int64(markerPositions, [rows.length, optionCount]),
    marker_mask: boolTensor(markerMask, [rows.length, optionCount]),
    qtype: int64(rows.map(row => row.qtype), [rows.length]),
  });
  const logitsTensor = output.logits, actionTensor = output.act_probs ?? output.act_logits;
  if (!logitsTensor || !actionTensor) throw Error(`Model returned unexpected outputs: ${Object.keys(output).join(", ")}`);
  const logits = Array.from(logitsTensor.to("float32").data, Number), actionValues = Array.from(actionTensor.to("float32").data, Number);
  const decisions = questions.map((question, rowIndex) => {
    const count = rows[rowIndex].markers.length, probabilities = softmax(logits.slice(rowIndex * optionCount, rowIndex * optionCount + count));
    if (probabilities.some(value => !Number.isFinite(value))) throw Error("Model returned invalid decision probabilities");
    let actProbability;
    if (actionTensor.dims.at(-1) === 2) {
      const pair = actionValues.slice(rowIndex * 2, rowIndex * 2 + 2);
      actProbability = output.act_probs ? pair[0] : softmax(pair)[0];
    } else actProbability = actionValues[rowIndex];
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
