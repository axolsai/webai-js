const CONFIG = {
  MODEL_ID: "gemma-4-e4b-it-litert-lm",
  MODEL_REPOSITORY: "litert-community/gemma-4-E4B-it-litert-lm",
  MODEL_REVISION: "2eee7ac325f20eb8c9ac1d0e972f7c84663062da",
  MODEL_FILE: "gemma-4-E4B-it-web.litertlm",
  MODEL_BYTES: 2_969_059_328,
  RUNTIME_BYTES: 21_688_351,
  RUNTIME_VERSION: "0.17.0",
  RUNTIME_URL: "https://cdn.jsdelivr.net/npm/@litert-lm/core@0.17.0/wasm/",
  CACHE_NAME: "transformers-cache",
  DEFAULT_MODEL_CONFIG: {
    max_new_tokens: 256,
    temperature: 0.7,
    do_sample: true,
    top_k: 40,
    top_p: 0.95,
    seed: 0,
  },
};

CONFIG.MODEL_URL = `https://huggingface.co/${CONFIG.MODEL_REPOSITORY}/resolve/${CONFIG.MODEL_REVISION}/${CONFIG.MODEL_FILE}`;

const MANIFEST = {
  contractVersion: "1.0",
  model: {
    id: CONFIG.MODEL_ID,
    displayName: "Gemma 4 E4B Instruct — LiteRT-LM",
    provider: "Google",
    providerUrl: "https://ai.google.dev/gemma",
    license: "Apache-2.0",
    licenseUrl: "https://www.apache.org/licenses/LICENSE-2.0.txt",
    lastUpdated: "2026-07-20T16:42:03.000Z",
    sourceRepository: "google/gemma-4-E4B-it",
    repository: CONFIG.MODEL_REPOSITORY,
    artifactRevision: CONFIG.MODEL_REVISION,
    artifactLastUpdated: "2026-08-07T02:18:35.000Z",
    task: "text-generation",
    description: "Gemma 4 E4B Instruct is a compact Google language model that performs text-only conversational generation locally in a WebGPU-capable browser through LiteRT-LM.",
    intendedUses: [
      "Local conversational assistants, drafting, rewriting, and summarization.",
      "Browser-side experiments where prompts and generated text should stay on the device after the model download.",
    ],
    limitations: [
      "Generated text can be incorrect, biased, unsafe, or fabricated and must be reviewed before consequential use.",
      "The web artifact is text-only even though the upstream Gemma model family supports other modalities.",
      "The approximately 2.99 GB web artifact and roughly 3.3 GB reported GPU-memory use require a capable desktop-class device.",
      "LiteRT-LM's current web model support is limited and this worker requires WebGPU; WASM-only inference is not advertised.",
    ],
  },
  runtime: {
    engine: { name: "LiteRT-LM JS", version: CONFIG.RUNTIME_VERSION },
    precisions: {
      q4: {
        supportedDevices: ["webgpu"],
        weightsSize: CONFIG.MODEL_BYTES,
        size: CONFIG.MODEL_BYTES + CONFIG.RUNTIME_BYTES,
        modelKeys: [CONFIG.MODEL_FILE],
        modelQuantization: "Gemma 4 mobile mixed 2-bit/4-bit/8-bit quantization-aware training; q4 is the WebAI selection label, not a claim that every tensor is 4-bit.",
        runtimePrecision: "Runtime-managed mixed precision",
        acceleration: "WebGPU-only web artifact",
      },
    },
  },
  input: {
    description: "A chat transcript whose final message is from the user.",
    alternatives: [{ name: "messages", type: "Array<{ role: 'system' | 'user' | 'assistant', content: string }>", required: true }],
  },
  config: {
    model: [
      { name: "max_new_tokens", type: "integer", default: 256, minimum: 1, maximum: 2048 },
      { name: "temperature", type: "number", default: 0.7, minimum: 0, maximum: 2 },
      { name: "do_sample", type: "boolean", default: true },
      { name: "top_k", type: "integer", default: 40, minimum: 1 },
      { name: "top_p", type: "number", default: 0.95, minimum: 0, maximum: 1 },
      { name: "seed", type: "integer", default: 0 },
    ],
    generation: [],
  },
  output: {
    description: "The generated assistant text and the updated chat transcript, matching WebAI's standard LLM result shape.",
    fields: [
      { name: "result", type: "string" },
      { name: "messages", type: "array" },
      { name: "structured_output", type: "boolean" },
      { name: "generationMetrics", type: "object", description: "Native LiteRT-LM prefill/decode throughput and time-to-first-token measurements." },
    ],
    example: { result: "Paris", messages: [{ role: "user", content: "What is the capital of France?" }, { role: "assistant", content: "Paris" }], structured_output: false },
  },
  operations: ["checkModelSupports", "init", "download", "generate", "generateStream", "clearMemory"],
  benchmark: {
    report: "benchmarks/gemma-4-e4b-it-litert-lm.json",
    testedAt: "2026-09-09T05:10:20.124Z",
    fixture: {
      file: "gemma-4-e4b-it-litert-lm.json",
      url: "https://f005.backblazeb2.com/file/hubters-webai-public/assets/benchmarks/gemma-4-e4b-it-litert-lm.json",
      size: 195,
      sha1: "a963a0f87228b86b7c9880699df48d4ab6c33482",
    },
    environment: "Codex in-app Chromium on macOS; LiteRT-LM JS 0.17.0; WebGPU",
    invalidatedBy: ["model artifact revision change", "LiteRT-LM version change", "fixture or quality-gate change", "browser/WebGPU behavior change"],
    results: [{
      precision: "q4",
      device: "webgpu",
      status: "pass",
      runtime: { loadMs: 183812.2, inferenceMs: 6132.6, timeToFirstTokenMs: 6087.25, prefillTokensPerSecond: 3.791, decodeTokensPerSecond: 48.193, totalMs: 189944.8, grade: "good" },
      quality: { grade: "exact-match", expected: "Paris", output: "Paris", nonEmpty: true, nonRepetitive: true },
      fullyAccelerated: true,
      cleanup: { modelDisposed: true, conversationsDisposed: true, modelCacheClearedBeforeCase: true, modelCacheClearedAfterCase: true },
    }],
  },
};

const CAPABILITIES = {
  supportedModes: ["webai"],
  supportedPrecisions: ["q4"],
  supportedPrecisionsDevicesMap: MANIFEST.runtime.precisions,
  doesSupportStreamGeneration: true,
  externalInterrupt: true,
  workerVersion: "v4",
  transformersJsVersion: null,
  cacheModelId: CONFIG.MODEL_REPOSITORY,
};

const state = { bridge: null, pending: new Map(), nextBridgeId: 0, initialized: false, initializing: false, generating: false };

function respond(requestId, type, data) { self.postMessage({ requestId, type, data }); }

function runtimeWorkerMain() {
  const MODEL = {
    url: "https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm/resolve/2eee7ac325f20eb8c9ac1d0e972f7c84663062da/gemma-4-E4B-it-web.litertlm",
    file: "gemma-4-E4B-it-web.litertlm",
    bytes: 2969059328,
    cache: "transformers-cache",
    wasm: "https://cdn.jsdelivr.net/npm/@litert-lm/core@0.17.0/wasm/",
    defaults: { max_new_tokens: 256, temperature: 0.7, do_sample: true, top_k: 40, top_p: 0.95, seed: 0 },
  };
  let api;
  let engine;

  function send(id, kind, data) { self.postMessage({ id, kind, data }); }
  async function cachedResponse() { return (await caches.open(MODEL.cache)).match(MODEL.url); }
  async function download(id) {
    if (await cachedResponse()) {
      send(id, "progress", { status: "progress", file: MODEL.file, loaded: MODEL.bytes, total: MODEL.bytes, progress: 100 });
      return;
    }
    const response = await fetch(MODEL.url);
    if (!response.ok || !response.body) throw new Error(`Model download failed with HTTP ${response.status}.`);
    const total = Number(response.headers.get("content-length")) || MODEL.bytes;
    let loaded = 0;
    const monitored = response.body.pipeThrough(new TransformStream({ transform(chunk, controller) {
      loaded += chunk.byteLength;
      send(id, "progress", { status: "progress", file: MODEL.file, loaded, total, progress: total > 0 ? loaded / total * 100 : 0 });
      controller.enqueue(chunk);
    } }));
    await (await caches.open(MODEL.cache)).put(MODEL.url, new Response(monitored, { headers: response.headers }));
    if (loaded !== MODEL.bytes) throw new Error(`Downloaded ${loaded} bytes; expected ${MODEL.bytes}.`);
  }
  function validateMessages(data) {
    const messages = data?.userInput?.messages;
    if (!Array.isArray(messages) || messages.length === 0) throw new Error("userInput.messages must be a non-empty array.");
    for (const message of messages) {
      if (!message || !["system", "user", "assistant"].includes(message.role) || typeof message.content !== "string" || !message.content.trim()) throw new Error("Each message needs a system, user, or assistant role and non-empty string content.");
    }
    if (messages.at(-1).role !== "user") throw new Error("The final message must have role 'user'.");
    const config = { ...MODEL.defaults, ...(data.modelConfig ?? {}) };
    if (!Number.isInteger(config.max_new_tokens) || config.max_new_tokens < 1 || config.max_new_tokens > 2048) throw new Error("modelConfig.max_new_tokens must be an integer from 1 through 2048.");
    if (!Number.isFinite(config.temperature) || config.temperature < 0 || config.temperature > 2) throw new Error("modelConfig.temperature must be from 0 through 2.");
    if (!Number.isInteger(config.top_k) || config.top_k < 1) throw new Error("modelConfig.top_k must be a positive integer.");
    if (!Number.isFinite(config.top_p) || config.top_p <= 0 || config.top_p > 1) throw new Error("modelConfig.top_p must be greater than 0 and at most 1.");
    if (!Number.isInteger(config.seed)) throw new Error("modelConfig.seed must be an integer.");
    return { messages, config };
  }
  function text(message) {
    if (typeof message?.content === "string") return message.content;
    return Array.isArray(message?.content) ? message.content.filter((part) => part?.type === "text").map((part) => part.text ?? "").join("") : "";
  }
  async function generate(id, data, streaming) {
    if (!engine) throw new Error("Call init before generation.");
    const { messages, config } = validateMessages(data);
    const samplerType = !config.do_sample || config.temperature === 0 ? api.SamplerType.GREEDY : config.top_p < 1 ? api.SamplerType.TOP_P : api.SamplerType.TOP_K;
    const conversation = await engine.createConversation({
      preface: messages.length > 1 ? { messages: messages.slice(0, -1) } : undefined,
      sessionConfig: { maxOutputTokens: config.max_new_tokens, samplerParams: { type: samplerType, k: config.top_k, p: config.top_p, temperature: config.temperature, seed: config.seed } },
    });
    try {
      let output = "";
      if (streaming) {
        for await (const chunk of conversation.sendMessageStreaming(messages.at(-1))) {
          const delta = text(chunk);
          if (delta) { output += delta; send(id, "stream", { result: delta, text: delta }); }
        }
      } else output = text(await conversation.sendMessage(messages.at(-1)));
      const generationMetrics = await conversation.getBenchmarkInfo();
      send(id, "result", { result: output, messages: [...messages, { role: "assistant", content: output }], structured_output: false, generationMetrics });
    } finally { await conversation.delete(); }
  }
  async function clear() {
    if (engine) { await engine.delete(); engine = null; }
    if (api) { api.unloadLiteRtLm(); api = null; }
  }
  self.onmessage = async ({ data: message }) => {
    const { id, type, data } = message;
    try {
      if (type === "download") await download(id);
      else if (type === "init") {
        await clear();
        await download(id);
        api = await import("https://cdn.jsdelivr.net/npm/@litert-lm/core@0.17.0/+esm");
        self.Module = {
          locateFile: (filename) => new URL(filename, MODEL.wasm).href,
          mainScriptUrlOrBlob: new URL("litertlm_wasm_internal.js", MODEL.wasm).href,
        };
        await api.loadLiteRtLm(MODEL.wasm);
        const cached = await cachedResponse();
        if (!cached?.body) throw new Error("The cached model stream is unavailable.");
        engine = await api.Engine.create({ model: cached.body, mainExecutorSettings: { maxNumTokens: 4096 }, benchmarkEnabled: true });
      } else if (type === "generate") { await generate(id, data, false); return; }
      else if (type === "generateStream") { await generate(id, data, true); return; }
      else if (type === "clear") await clear();
      else throw new Error(`Unknown bridge operation: ${type}`);
      send(id, "result", { status: "success" });
    } catch (error) { send(id, "error", { message: error instanceof Error ? error.message : String(error) }); }
  };
}

function ensureBridge() {
  if (state.bridge) return state.bridge;
  const source = `(${runtimeWorkerMain.toString()})()`;
  const url = URL.createObjectURL(new Blob([source], { type: "text/javascript" }));
  const bridge = new Worker(url);
  URL.revokeObjectURL(url);
  bridge.onmessage = ({ data: message }) => {
    const pending = state.pending.get(message.id);
    if (!pending) return;
    if (message.kind === "progress") pending.progress?.(message.data);
    else if (message.kind === "stream") pending.stream?.(message.data);
    else {
      state.pending.delete(message.id);
      if (message.kind === "error") pending.reject(new Error(message.data?.message ?? "LiteRT-LM bridge failed."));
      else pending.resolve(message.data);
    }
  };
  bridge.onerror = (event) => {
    for (const pending of state.pending.values()) pending.reject(new Error(event.message || "LiteRT-LM bridge crashed."));
    state.pending.clear();
  };
  state.bridge = bridge;
  return bridge;
}

function callBridge(type, data, progress, stream) {
  return new Promise((resolve, reject) => {
    const id = ++state.nextBridgeId;
    state.pending.set(id, { resolve, reject, progress, stream });
    ensureBridge().postMessage({ id, type, data });
  });
}

function validateInit(data) {
  if (data?.precision !== "q4") throw new Error("Gemma 4 E4B LiteRT-LM supports only the WebAI q4 selection.");
  if (data?.device !== "webgpu") throw new Error("Gemma 4 E4B LiteRT-LM requires WebGPU.");
  if (!self.navigator?.gpu) throw new Error("WebGPU is unavailable in this worker.");
}

async function clearMemory() {
  if (state.bridge) {
    await callBridge("clear");
    state.bridge.terminate();
    state.bridge = null;
  }
  state.initialized = false;
}

self.addEventListener("message", async ({ data: message = {} }) => {
  const { requestId, type, data } = message;
  try {
    switch (type) {
      case "checkModelSupports":
        respond(requestId, type, { ...CAPABILITIES, manifest: MANIFEST });
        break;
      case "download":
        await callBridge("download", data, (value) => respond(requestId, "downloadProgress", value));
        respond(requestId, type, { status: "success" });
        break;
      case "init":
        if (state.initializing || state.generating) throw new Error("Worker is busy.");
        state.initializing = true;
        try {
          validateInit(data);
          await clearMemory();
          await callBridge("init", data, (value) => respond(requestId, "downloadProgress", value));
          state.initialized = true;
          respond(requestId, type, { status: "success" });
        } finally {
          state.initializing = false;
        }
        break;
      case "generate":
      case "generateStream":
        if (!state.initialized || !state.bridge) throw new Error("Call init before generation.");
        if (state.generating) throw new Error("Generation is already in progress.");
        state.generating = true;
        try {
          const result = await callBridge(type, data, undefined, (value) => respond(requestId, "streamChunk", value));
          respond(requestId, "generated", { status: "success", result });
        } finally {
          state.generating = false;
        }
        break;
      case "clearMemory":
        if (state.generating) throw new Error("Cannot clear memory while generation is active.");
        await clearMemory();
        respond(requestId, type, { status: "success" });
        break;
      default:
        throw new Error(`Unknown operation: ${type}`);
    }
  } catch (error) {
    respond(requestId, "error", { message: error instanceof Error ? error.message : String(error), context: type });
  }
});

self.postMessage({ type: "worker initialized", data: { success: true, workerVersion: CAPABILITIES.workerVersion, modelId: MANIFEST.model.id } });
