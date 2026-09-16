// WebAI self-describing worker template — contract v1.0.
// Copy to public/workers/{modelId}.worker.js and replace the adapter methods/manifest.

const MANIFEST = {
  contractVersion: "1.0",
  model: {
    id: "replace-me",
    displayName: "Human-readable model name",
    provider: "Original model creator",
    providerUrl: "https://provider.example/",
    license: "SPDX-license-id",
    licenseUrl: "https://source.example/LICENSE",
    lastUpdated: "YYYY-MM-DDTHH:mm:ss.sssZ",
    sourceRepository: "original-provider/model",
    repository: "organization/model",
    artifactRevision: "immutable commit hash",
    artifactLastUpdated: "YYYY-MM-DDTHH:mm:ss.sssZ",
    task: "replace-me",
    description: "Describe what this worker does.",
    intendedUses: ["Describe a suitable use."],
    limitations: ["Describe an important accuracy, safety, or runtime limitation."],
  },
  runtime: { engine: { name: "replace-me", version: "exact-version" }, precisions: {} },
  input: { description: "Describe the accepted input.", alternatives: [] },
  config: { model: [], generation: [] },
  output: { description: "Describe the native model output.", fields: [], example: {} },
  operations: ["checkModelSupports", "init", "download", "generate", "clearMemory"],
  benchmark: {
    report: "benchmarks/MODEL.json",
    testedAt: "YYYY-MM-DDTHH:mm:ss.sssZ",
    fixture: { file: "fixture", url: "https://assets.example/fixture", size: 0, sha1: "replace", durationSeconds: 0 },
    environment: "Browser, OS, and pinned runtime version",
    invalidatedBy: ["model artifact revision change", "runtime version change", "fixture or quality-gate change", "browser/backend behavior change"],
    results: [],
  },
};

const CAPABILITIES = {
  supportedModes: ["webai"],
  supportedPrecisions: [],
  supportedPrecisionsDevicesMap: {},
  doesSupportStreamGeneration: false,
  externalInterrupt: true,
  workerVersion: "v4",
  transformersJsVersion: null,
  cacheModelId: MANIFEST.model.repository,
};

const state = { initialized: false, initializing: false, generating: false };

const adapter = {
  configure(_workerConfig) {},
  async init(_data, _progress) { throw new Error("Implement adapter.init"); },
  async download(_data, _progress) { throw new Error("Implement adapter.download"); },
  async generate(_data) { throw new Error("Implement adapter.generate"); },
  async clearMemory() {},
};

function respond(requestId, type, data) {
  self.postMessage({ requestId, type, data });
}

function progress(requestId, data) {
  respond(requestId, "downloadProgress", data);
}

self.addEventListener("message", async ({ data: message = {} }) => {
  const { requestId, type, data } = message;
  try {
    switch (type) {
      case "checkModelSupports":
        adapter.configure(data?.workerConfig);
        respond(requestId, type, { ...CAPABILITIES, manifest: MANIFEST });
        break;
      case "init":
        if (state.initializing || state.generating) throw new Error("Worker is busy");
        state.initializing = true;
        try {
          await adapter.init(data, (value) => progress(requestId, value));
          state.initialized = true;
          respond(requestId, type, { status: "success" });
        } finally { state.initializing = false; }
        break;
      case "download":
        await adapter.download(data, (value) => progress(requestId, value));
        respond(requestId, type, { status: "success" });
        break;
      case "generate":
        if (!state.initialized) throw new Error("Call init before generate");
        if (state.generating) throw new Error("Generation already in progress");
        state.generating = true;
        try {
          const result = await adapter.generate(data);
          respond(requestId, "generated", { status: "success", result });
        } finally { state.generating = false; }
        break;
      case "clearMemory":
        await adapter.clearMemory();
        state.initialized = false;
        respond(requestId, type, { status: "success" });
        break;
      default:
        throw new Error(`Unknown operation: ${type}`);
    }
  } catch (error) {
    respond(requestId, "error", {
      message: error instanceof Error ? error.message : String(error),
      context: type,
    });
  }
});

self.postMessage({
  type: "worker initialized",
  data: { success: true, workerVersion: CAPABILITIES.workerVersion, modelId: MANIFEST.model.id },
});
