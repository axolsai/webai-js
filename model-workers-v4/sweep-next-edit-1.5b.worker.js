import {
  AutoModelForCausalLM,
  AutoTokenizer,
  TextStreamer,
  env,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0";

const MODEL_CONFIG_FIELDS = [
  { name: "max_new_tokens", type: "integer", default: 512, minimum: 1, maximum: 2048, description: "Maximum number of tokens in the predicted updated file." },
  { name: "do_sample", type: "boolean", default: false, description: "Use sampling instead of deterministic greedy decoding." },
  { name: "temperature", type: "number", default: 1, minimum: 0, maximum: 2, description: "Sampling temperature; ignored by greedy decoding." },
  { name: "top_k", type: "integer", default: 50, minimum: 1, description: "Sampling candidate count." },
  { name: "top_p", type: "number", default: 1, minimum: 0, maximum: 1, description: "Nucleus-sampling probability mass." },
  { name: "repetition_penalty", type: "number", default: 1, minimum: 0.01, description: "Penalty applied to repeated tokens." },
];
const GENERATION_CONFIG_FIELDS = [
  { name: "skip_special_tokens", type: "boolean", default: true, description: "Remove tokenizer control tokens from the returned edit." },
];
const defaultsFrom = (fields) => Object.fromEntries(fields.map(({ name, default: value }) => [name, value]));
const CANDIDATE_PRECISIONS = {
  fp32: { weightsSize: 5_797_054_210, size: 5_800_122_008, modelKeys: ["model.onnx", "model.onnx_data", "model.onnx_data_1", "model.onnx_data_2"], dtype: "fp32", candidateDevices: ["wasm", "webgpu"] },
  fp16: { weightsSize: 2_898_635_284, size: 2_901_703_082, modelKeys: ["model_fp16.onnx", "model_fp16.onnx_data", "model_fp16.onnx_data_1"], dtype: "fp16", candidateDevices: ["wasm", "webgpu"] },
  q8: { weightsSize: 1_688_302_819, size: 1_691_370_617, modelKeys: ["model_quantized.onnx", "model_quantized.onnx_data"], dtype: "q8", candidateDevices: ["wasm", "webgpu"] },
  q4: { weightsSize: 943_286_437, size: 946_354_235, modelKeys: ["model_q4.onnx", "model_q4.onnx_data"], dtype: "q4", candidateDevices: ["wasm", "webgpu"] },
  q4f16: { weightsSize: 844_302_125, size: 847_369_923, modelKeys: ["model_q4f16.onnx", "model_q4f16.onnx_data"], dtype: "q4f16", candidateDevices: ["wasm", "webgpu"] },
};
const CONFIG = {
  MODEL_ID: "sweep-next-edit-1.5b",
  MODEL_REPOSITORY: "Xenova/sweep-next-edit-1.5B",
  MODEL_REVISION: "402c437f67cee450c00d3add669b43ae79e55e50",
  TRANSFORMERS_JS_VERSION: "4.2.0",
  DEFAULT_MODEL_CONFIG: defaultsFrom(MODEL_CONFIG_FIELDS),
  DEFAULT_GENERATION_CONFIG: defaultsFrom(GENERATION_CONFIG_FIELDS),
  // Artifact-derived candidates are benchmarked separately. Only measured passing
  // combinations may be copied into runtime.precisions and advertised below.
  PRECISIONS: {},
};

const BENCHMARK_RESULTS = [];
const MANIFEST = {
  contractVersion: "1.0",
  model: {
    id: CONFIG.MODEL_ID,
    displayName: "Sweep Next-Edit 1.5B — ONNX",
    provider: "Sweep AI",
    providerUrl: "https://sweep.dev/",
    license: "Apache-2.0",
    licenseUrl: "https://www.apache.org/licenses/LICENSE-2.0.txt",
    lastUpdated: "2026-01-22T20:40:28.000Z",
    sourceRepository: "sweepai/sweep-next-edit-1.5B",
    repository: CONFIG.MODEL_REPOSITORY,
    artifactRevision: CONFIG.MODEL_REVISION,
    artifactLastUpdated: "2026-01-24T02:26:04.000Z",
    task: "text-generation",
    description: "A 1.5-billion-parameter Qwen2.5-Coder-based model from Sweep AI that predicts the complete next state of a code file from repository context, recent diffs, and the file's original and current contents, running locally in a browser through Transformers.js and ONNX Runtime Web.",
    intendedUses: [
      "Predicting a likely next code edit from a developer's recent changes.",
      "Prototyping local code-edit autocomplete in browser developer tools and editors.",
      "Benchmarking causal language-model ONNX artifacts with the reusable Transformers.js LLM harness.",
    ],
    limitations: [
      "This is a specialized next-edit predictor, not a general conversational assistant.",
      "Predictions may be incorrect, incomplete, insecure, or destructive and require developer review before application.",
      "Quality depends on exact prompt formatting and useful repository and edit context; unrelated or truncated context can substantially reduce accuracy.",
      "The model can reproduce sensitive or licensed material supplied in its context; callers are responsible for reviewing inputs and outputs.",
      "The smallest ONNX candidate is approximately 844 MB before browser runtime assets and may exceed memory limits on mobile or constrained devices.",
    ],
  },
  runtime: { engine: { name: "Transformers.js", version: CONFIG.TRANSFORMERS_JS_VERSION }, precisions: CONFIG.PRECISIONS },
  input: {
    description: "Provide structured next-edit context, or an already formatted Sweep prompt. The output is a predicted complete updated file.",
    alternatives: [
      { name: "structured edit", fields: [
        { name: "file_path", type: "string", required: true },
        { name: "original_content", type: "string", required: true },
        { name: "current_content", type: "string", required: true },
        { name: "context_files", type: "Record<string, string>", required: false },
        { name: "recent_diffs", type: "Array<{ file_path: string, original: string, updated: string }>", required: false },
      ] },
      { name: "formatted prompt", fields: [{ name: "prompt", type: "string", required: true }] },
      { name: "messages", fields: [{ name: "messages", type: "Array<{ role: string, content: string }>", required: true, description: "The last message content is treated as an already formatted Sweep prompt." }] },
    ],
  },
  config: { model: MODEL_CONFIG_FIELDS, generation: GENERATION_CONFIG_FIELDS },
  output: {
    description: "The predicted updated file and a standard WebAI LLM transcript. WebAI wraps this native object in its public result envelope.",
    fields: [
      { name: "result", type: "string", description: "Predicted complete updated file." },
      { name: "messages", type: "array", description: "Input prompt transcript plus the generated assistant prediction." },
      { name: "structured_output", type: "boolean", description: "Always false for this worker." },
    ],
    example: { result: "def greet(name):\n    ...", messages: [{ role: "user", content: "<|file_sep|>..." }, { role: "assistant", content: "def greet(name):\n    ..." }], structured_output: false },
  },
  operations: ["checkModelSupports", "init", "download", "generate", "generateStream", "clearMemory"],
  benchmark: {
    report: "benchmarks/sweep-next-edit-1.5b.json",
    testedAt: "2026-09-09T07:47:49.698Z",
    fixture: {
      file: "sweep-next-edit-1.5b.json",
      url: "https://f005.backblazeb2.com/file/hubters-webai-public/assets/benchmarks/sweep-next-edit-1.5b.json",
      size: 1291,
      sha1: "98a29ff43b6bdcd2faba07f9ba47bb9d9ed36192",
    },
    environment: "Chrome 151.0.7922.176 on macOS arm64; Transformers.js 4.2.0",
    note: "No candidate is advertised because every tested precision/device pair failed model initialization. See the report for preserved errors.",
    invalidatedBy: ["model artifact revision change", "Transformers.js or ONNX Runtime version change", "fixture or quality-gate change", "browser/backend behavior change"],
    candidates: CANDIDATE_PRECISIONS,
    results: BENCHMARK_RESULTS,
  },
};
const CAPABILITIES = {
  supportedModes: ["webai"],
  supportedPrecisions: Object.keys(CONFIG.PRECISIONS),
  supportedPrecisionsDevicesMap: CONFIG.PRECISIONS,
  doesSupportStreamGeneration: true,
  externalInterrupt: true,
  workerVersion: "v4",
  transformersJsVersion: CONFIG.TRANSFORMERS_JS_VERSION,
  cacheModelId: CONFIG.MODEL_REPOSITORY,
};

let tokenizer = null;
let model = null;
const state = { initialized: false, initializing: false, generating: false };
function respond(requestId, type, data) { self.postMessage({ requestId, type, data }); }
function configure(workerConfig = {}) {
  env.allowRemoteModels = workerConfig.allowRemoteModels ?? true;
  env.allowLocalModels = workerConfig.allowLocalModels ?? false;
  env.useBrowserCache = workerConfig.useBrowserCache ?? true;
  if (workerConfig.remoteHost) env.remoteHost = workerConfig.remoteHost;
  if (workerConfig.remotePathTemplate) env.remotePathTemplate = workerConfig.remotePathTemplate;
  if (workerConfig.localModelPath) env.localModelPath = workerConfig.localModelPath;
}
function progress(requestId, value) {
  respond(requestId, "downloadProgress", { ...value, progress: value.status === "ready" ? 100 : value.progress });
}
function validateInit(data) {
  const precision = data?.precision;
  const device = data?.device;
  const spec = CONFIG.PRECISIONS[precision] ?? (data?.benchmark === true ? CANDIDATE_PRECISIONS[precision] : undefined);
  if (!spec) throw new Error(`Unsupported or unverified precision: ${precision}. Run the benchmark matrix before advertising it.`);
  const devices = spec.supportedDevices ?? spec.candidateDevices;
  if (!devices.includes(device)) throw new Error(`Precision ${precision} does not support device ${device}.`);
  if (device === "webgpu" && !self.navigator?.gpu) throw new Error("WebGPU is unavailable in this worker.");
  return { precision, device };
}
function buildPrompt(input = {}) {
  if (typeof input.prompt === "string" && input.prompt.trim()) return input.prompt;
  if (Array.isArray(input.messages) && input.messages.length) {
    const content = input.messages.at(-1)?.content;
    if (typeof content !== "string" || !content.trim()) throw new Error("The final userInput.messages entry must contain non-empty string content.");
    return content;
  }
  for (const key of ["file_path", "original_content", "current_content"]) {
    if (typeof input[key] !== "string" || !input[key].trim()) throw new Error(`userInput.${key} must be a non-empty string.`);
  }
  const parts = [];
  const contextFiles = input.context_files ?? {};
  if (!contextFiles || typeof contextFiles !== "object" || Array.isArray(contextFiles)) throw new Error("userInput.context_files must be an object.");
  for (const [path, content] of Object.entries(contextFiles)) {
    if (typeof content !== "string") throw new Error("Every context_files value must be a string.");
    parts.push(`<|file_sep|>${path}`, content);
  }
  const recentDiffs = input.recent_diffs ?? [];
  if (!Array.isArray(recentDiffs)) throw new Error("userInput.recent_diffs must be an array.");
  for (const diff of recentDiffs) {
    if (!diff || typeof diff.file_path !== "string" || typeof diff.original !== "string" || typeof diff.updated !== "string") throw new Error("Every recent diff needs string file_path, original, and updated fields.");
    parts.push(`<|file_sep|>${diff.file_path}.diff`, "original:", diff.original, "updated:", diff.updated);
  }
  parts.push(`<|file_sep|>original/${input.file_path}`, input.original_content, `<|file_sep|>current/${input.file_path}`, input.current_content, `<|file_sep|>updated/${input.file_path}`);
  return parts.join("\n");
}
function validateGeneration(data) {
  const prompt = buildPrompt(data?.userInput);
  const config = { ...CONFIG.DEFAULT_MODEL_CONFIG, ...(data?.modelConfig ?? {}) };
  const generateConfig = { ...CONFIG.DEFAULT_GENERATION_CONFIG, ...(data?.generateConfig ?? {}) };
  if (!Number.isInteger(config.max_new_tokens) || config.max_new_tokens < 1 || config.max_new_tokens > 2048) throw new Error("modelConfig.max_new_tokens must be an integer from 1 through 2048.");
  if (typeof config.do_sample !== "boolean") throw new Error("modelConfig.do_sample must be boolean.");
  if (!Number.isFinite(config.temperature) || config.temperature < 0 || config.temperature > 2) throw new Error("modelConfig.temperature must be from 0 through 2.");
  if (!Number.isInteger(config.top_k) || config.top_k < 1) throw new Error("modelConfig.top_k must be a positive integer.");
  if (!Number.isFinite(config.top_p) || config.top_p <= 0 || config.top_p > 1) throw new Error("modelConfig.top_p must be greater than 0 and at most 1.");
  if (!Number.isFinite(config.repetition_penalty) || config.repetition_penalty <= 0) throw new Error("modelConfig.repetition_penalty must be positive.");
  if (typeof generateConfig.skip_special_tokens !== "boolean") throw new Error("generateConfig.skip_special_tokens must be boolean.");
  return { prompt, config, generateConfig };
}
async function clearMemory() {
  if (model?.dispose) await model.dispose();
  model = null;
  tokenizer = null;
  state.initialized = false;
}
async function load(data, requestId) {
  const { precision, device } = validateInit(data);
  await clearMemory();
  tokenizer = await AutoTokenizer.from_pretrained(CONFIG.MODEL_REPOSITORY, { revision: CONFIG.MODEL_REVISION, progress_callback: (value) => progress(requestId, value) });
  model = await AutoModelForCausalLM.from_pretrained(CONFIG.MODEL_REPOSITORY, { revision: CONFIG.MODEL_REVISION, dtype: precision, device, progress_callback: (value) => progress(requestId, value) });
}
async function generate(data, requestId, streaming) {
  const { prompt, config, generateConfig } = validateGeneration(data);
  const inputs = await tokenizer(prompt, { add_special_tokens: false });
  const streamer = streaming ? new TextStreamer(tokenizer, {
    skip_prompt: true,
    skip_special_tokens: generateConfig.skip_special_tokens,
    callback_function: (text) => {
      if (!text) return;
      respond(requestId, "streamChunk", { result: text, text, structured_output: false });
    },
  }) : undefined;
  const stopTokenIds = [tokenizer.convert_tokens_to_ids("<|file_sep|>"), tokenizer.eos_token_id].filter(Number.isInteger);
  const outputs = await model.generate({ ...inputs, ...config, eos_token_id: stopTokenIds, pad_token_id: tokenizer.pad_token_id ?? tokenizer.eos_token_id, streamer });
  const promptLength = inputs.input_ids.dims.at(-1);
  const ids = outputs.tolist()[0].slice(promptLength);
  const output = tokenizer.decode(ids, { skip_special_tokens: generateConfig.skip_special_tokens });
  const messages = Array.isArray(data.userInput?.messages) ? [...data.userInput.messages] : [{ role: "user", content: prompt }];
  return { result: output, messages: [...messages, { role: "assistant", content: output }], structured_output: false };
}

self.addEventListener("message", async ({ data: message = {} }) => {
  const { requestId, type, data } = message;
  try {
    if (type === "checkModelSupports") {
      configure(data?.workerConfig);
      respond(requestId, type, { ...CAPABILITIES, manifest: MANIFEST });
    } else if (type === "download") {
      await load(data, requestId);
      await clearMemory();
      respond(requestId, type, { status: "success" });
    } else if (type === "init") {
      if (state.initializing || state.generating) throw new Error("Worker is busy.");
      state.initializing = true;
      try { await load(data, requestId); state.initialized = true; respond(requestId, type, { status: "success" }); }
      finally { state.initializing = false; }
    } else if (type === "generate" || type === "generateStream") {
      if (!state.initialized || !model || !tokenizer) throw new Error("Call init before generation.");
      if (state.generating) throw new Error("Generation is already in progress.");
      state.generating = true;
      try { respond(requestId, "generated", { status: "success", result: await generate(data, requestId, type === "generateStream") }); }
      finally { state.generating = false; }
    } else if (type === "clearMemory") {
      if (state.generating) throw new Error("Cannot clear memory while generation is active.");
      await clearMemory();
      respond(requestId, type, { status: "success" });
    } else throw new Error(`Unknown operation: ${type}`);
  } catch (error) {
    respond(requestId, "error", { message: error instanceof Error ? error.message : String(error), context: type });
  }
});

self.postMessage({ type: "worker initialized", data: { success: true, workerVersion: CAPABILITIES.workerVersion, modelId: MANIFEST.model.id } });
