import {
  AutoProcessor,
  VitMatteForImageMatting,
  RawImage,
  env,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0";
const ID = "vitmatte-base-composition-1k",
  REPO = "Xenova/vitmatte-base-composition-1k",
  REV = "1290b014b994e95ca1b9dd9c5f72c3b6d5b7236a";
const make = (weightsSize, dtype, suffix) => ({
  weightsSize,
  size: weightsSize + 2178,
  dtype,
  modelKeys: [`model${suffix}.onnx`],
  candidateDevices: ["wasm", "webgpu"],
});
const PRECISIONS = {
  fp32: make(387371620, "fp32", ""),
  q8: make(99003292, "q8", "_quantized"),
};
const SUPPORTED = {
  fp32: { ...PRECISIONS.fp32, supportedDevices: ["wasm", "webgpu"] },
  q8: { ...PRECISIONS.q8, supportedDevices: ["wasm", "webgpu"] },
};
const MANIFEST = {
  contractVersion: "1.0",
  model: {
    id: ID,
    displayName: "VitMatte Base Composition-1K — Trimap Matting",
    provider: "HUST Vision Lab",
    providerUrl: "https://github.com/hustvl/ViTMatte",
    license: "Apache-2.0",
    licenseUrl: "https://github.com/hustvl/ViTMatte/blob/main/LICENSE",
    lastUpdated: "2023-09-21T09:25:07.000Z",
    sourceRevision: "bf486d01a7d9e3dbcc8400f7942835caf0eaf76e",
    sourceRepository: "hustvl/vitmatte-base-composition-1k",
    repository: REPO,
    artifactRevision: REV,
    artifactLastUpdated: "2026-04-24T15:57:09.000Z",
    task: "image-matting",
    description:
      "A base-size HUST Vision Lab transformer trained on Composition-1K, converted by Xenova for browser inference, that refines a supplied foreground/background/unknown trimap into a soft image alpha matte.",
    intendedUses: [
      "Refine user-provided trimaps for image editing and foreground compositing.",
      "Prototype trimap-guided extraction of fine object boundaries.",
    ],
    limitations: [
      "Requires a same-size trimap; it does not independently discover the foreground.",
      "Incorrect foreground/background markings can force wrong alpha values and lose image content.",
      "Hair, transparent objects, blur, poor contrast, and out-of-domain scenes can produce halos or missing detail.",
      "The q8 artifact passed the broad regression gate but produced substantially higher error and less recovered foreground than FP32 on the pinned fixture.",
      "Alpha values are editing estimates, not semantic or factual classifications; do not use for high-impact decisions.",
    ],
  },
  runtime: {
    engine: { name: "Transformers.js", version: "4.2.0" },
    precisions: SUPPORTED,
  },
  input: {
    description:
      "Image and matching trimap URLs. Trimap grayscale values: 0 background, 128 unknown, 255 foreground.",
    alternatives: [
      {
        name: "image and trimap",
        fields: [
          {
            name: "image",
            type: "string(URL)",
            required: true,
            minLength: 1,
            format: "url",
          },
          {
            name: "trimap",
            type: "string(URL)",
            required: true,
            minLength: 1,
            format: "url",
          },
        ],
      },
    ],
  },
  config: { model: [], generation: [] },
  output: {
    description: "Soft alpha values in original image dimensions.",
    fields: [
      { name: "alpha", type: "Float32Array" },
      { name: "width", type: "integer" },
      { name: "height", type: "integer" },
    ],
    example: { alpha: "Float32Array", width: 960, height: 640 },
  },
  operations: [
    "checkModelSupports",
    "init",
    "download",
    "generate",
    "clearMemory",
  ],
  benchmark: {
    report: "benchmarks/vitmatte-base-composition-1k.json",
    testedAt: "2026-09-26T07:43:10.731Z",
    fixture: {
      file: "modnet-portrait-pexels-5965592.jpg",
      url: "https://f005.backblazeb2.com/file/hubters-webai-public/assets/benchmarks/modnet-portrait-pexels-5965592.jpg",
      size: 78124,
      sha1: "38212739a4a4f0be25d84863cf991d7fb6c6d818",
      width: 1024,
      height: 683,
      source:
        "https://www.pexels.com/photo/woman-wearing-pink-turtleneck-sweater-5965592/",
      license: "Pexels License",
      licenseUrl: "https://www.pexels.com/license/",
      preprocessing:
        "RGB image through pinned processor; the checked-in 0/128/255 trimap is derived from the MODNet reference with an 8-pixel sampled boundary.",
      reference: {
        file: "modnet-portrait-pexels-5965592-mask.png",
        url: "https://f005.backblazeb2.com/file/hubters-webai-public/assets/benchmarks/modnet-portrait-pexels-5965592-mask.png",
        size: 67541,
        sha1: "01ace4f95454ce44cc53c2c797a87a583c86cee8",
        width: 1024,
        height: 683,
      },
      trimap: {
        file: "vitmatte-portrait-5965592-trimap.png",
        url: "https://f005.backblazeb2.com/file/hubters-webai-public/assets/benchmarks/vitmatte-portrait-5965592-trimap.png",
        size: 22459,
        sha1: "64b06a53a2b0f33507093d2696911ec236a00396",
        width: 1024,
        height: 683,
      },
    },
    candidates: PRECISIONS,
    qualityGate:
      "Finite original-size alpha in [0,1], non-degenerate foreground, mean absolute error <= 0.2 versus the local MODNet regression reference; trimap preserves certain foreground/background and marks the boundary unknown.",
    note: "The MODNet reference is a regression proxy, not annotated ground truth.",
    environment: "Pending isolated-browser matrix; Transformers.js 4.2.0",
    invalidatedBy: [
      "artifact revision change",
      "runtime version change",
      "fixture or quality-gate change",
      "browser/backend behavior change",
    ],
    results: [
      {
        precision: "fp32",
        device: "wasm",
        status: "pass",
        runtime: {
          loadMs: 43907.60000002384,
          inferenceMs: 20086.5,
          totalMs: 65276.10000002384,
          grade: "very-slow",
          imagesPerSecond: 0.049784681253578274,
        },
        cleanup: {
          modelDisposed: true,
          cacheClearedBeforeCase: true,
          cacheClearedAfterCase: true,
          deletedEntries: 2,
        },
        quality: {
          valid: true,
          meanAbsoluteError: 0.0028033782470408327,
          foregroundRatio: 0.2549342857796486,
          grade: "matting-pass",
        },
        nativeOutput: { width: 1024, height: 683, alphaLength: 699392 },
      },
      {
        precision: "fp32",
        device: "webgpu",
        status: "pass",
        runtime: {
          loadMs: 45021.80000001192,
          inferenceMs: 968.5,
          totalMs: 47274.79999998212,
          grade: "good",
          imagesPerSecond: 1.0325245224574084,
        },
        cleanup: {
          modelDisposed: true,
          cacheClearedBeforeCase: true,
          cacheClearedAfterCase: true,
          deletedEntries: 2,
        },
        quality: {
          valid: true,
          meanAbsoluteError: 0.002803391196857619,
          foregroundRatio: 0.2549342857796486,
          grade: "matting-pass",
        },
        nativeOutput: { width: 1024, height: 683, alphaLength: 699392 },
      },
      {
        precision: "q8",
        device: "wasm",
        status: "pass",
        runtime: {
          loadMs: 15451,
          inferenceMs: 20787,
          totalMs: 37558.19999998808,
          grade: "very-slow",
          imagesPerSecond: 0.048106989945639105,
        },
        cleanup: {
          modelDisposed: true,
          cacheClearedBeforeCase: true,
          cacheClearedAfterCase: true,
          deletedEntries: 3,
        },
        quality: {
          valid: true,
          meanAbsoluteError: 0.14917271067533516,
          foregroundRatio: 0.10128511621522694,
          grade: "matting-pass",
          qualityNote:
            "Passes the broad regression threshold but is materially degraded versus FP32 on this fixture.",
        },
        nativeOutput: { width: 1024, height: 683, alphaLength: 699392 },
      },
      {
        precision: "q8",
        device: "webgpu",
        status: "pass",
        runtime: {
          loadMs: 17564.69999998808,
          inferenceMs: 17431.09999999404,
          totalMs: 36313.69999998808,
          grade: "very-slow",
          imagesPerSecond: 0.057368726012721055,
        },
        cleanup: {
          modelDisposed: true,
          cacheClearedBeforeCase: true,
          cacheClearedAfterCase: true,
          deletedEntries: 3,
        },
        quality: {
          valid: true,
          meanAbsoluteError: 0.148451668763005,
          foregroundRatio: 0.10273494692532943,
          grade: "matting-pass",
          qualityNote:
            "Passes the broad regression threshold but is materially degraded versus FP32 on this fixture.",
        },
        nativeOutput: { width: 1024, height: 683, alphaLength: 699392 },
      },
    ],
  },
};
const state = { initializing: false, initialized: false, generating: false };
let model = null,
  processor = null;
const send = (requestId, type, data) =>
  self.postMessage({ requestId, type, data });
const caps = {
  supportedModes: ["webai"],
  supportedPrecisions: Object.keys(SUPPORTED),
  supportedPrecisionsDevicesMap: SUPPORTED,
  doesSupportStreamGeneration: false,
  externalInterrupt: true,
  workerVersion: "v4",
  transformersJsVersion: "4.2.0",
  cacheModelId: REPO,
};
function configure(c = {}) {
  env.allowRemoteModels = c.allowRemoteModels ?? true;
  env.allowLocalModels = c.allowLocalModels ?? false;
  env.useBrowserCache = c.useBrowserCache ?? true;
  if (c.remoteHost) env.remoteHost = c.remoteHost;
  if (c.remotePathTemplate) env.remotePathTemplate = c.remotePathTemplate;
  if (c.localModelPath) env.localModelPath = c.localModelPath;
}
async function clear() {
  if (model?.dispose) await model.dispose();
  model = null;
  processor = null;
  state.initialized = false;
}
async function load(data, id) {
  const p = data?.precision,
    d = data?.device,
    s = SUPPORTED[p];
  if (!s) throw Error(`Unsupported precision: ${p}`);
  const devices = s.supportedDevices ?? s.candidateDevices;
  if (!devices.includes(d))
    throw Error(`Precision ${p} is only supported on: ${devices.join(", ")}`);
  if (d === "webgpu" && !self.navigator?.gpu)
    throw Error("WebGPU is unavailable");
  await clear();
  const progress_callback = (v) => send(id, "downloadProgress", v);
  model = await VitMatteForImageMatting.from_pretrained(REPO, {
    revision: REV,
    dtype: p,
    device: d,
    progress_callback,
  });
  processor = await AutoProcessor.from_pretrained(REPO, {
    revision: REV,
    progress_callback,
  });
}
async function generate(data) {
  const { image, trimap } = data?.userInput ?? {};
  if (
    typeof image !== "string" ||
    !image.trim() ||
    typeof trimap !== "string" ||
    !trimap.trim()
  )
    throw Error("userInput.image and trimap must be non-empty URLs");
  const raw = await RawImage.read(image),
    map = await RawImage.read(trimap);
  if (raw.width !== map.width || raw.height !== map.height)
    throw Error("Image and trimap dimensions must match");
  const inputs = await processor(raw, map),
    { alphas } = await model(inputs),
    [, , height, width] = alphas.dims;
  if (width < raw.width || height < raw.height)
    throw Error("Model returned undersized alpha");
  const alpha = new Float32Array(raw.width * raw.height);
  for (let y = 0; y < raw.height; y++)
    for (let x = 0; x < raw.width; x++) {
      const value = Number(alphas.data[y * width + x]);
      if (!Number.isFinite(value))
        throw Error("Model returned non-finite alpha");
      alpha[y * raw.width + x] = Math.max(0, Math.min(1, value));
    }
  return { alpha, width: raw.width, height: raw.height };
}
function validateConfig(data) {
  for (const [key, fields] of [
    ["modelConfig", MANIFEST.config.model],
    ["generateConfig", MANIFEST.config.generation],
  ]) {
    const config = data?.[key];
    if (config === undefined) continue;
    if (!config || typeof config !== "object" || Array.isArray(config))
      throw Error(`${key} must be an object`);
    for (const name of Object.keys(config))
      if (!fields.some((field) => field.name === name))
        throw Error(`Unsupported ${key} option: ${name}`);
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
        const device =
          type === "download"
            ? SUPPORTED[precision]?.supportedDevices[0]
            : data?.device;
        await load({ ...data, precision, device }, requestId);
        if (type === "download") await clear();
        else state.initialized = true;
        send(requestId, type, { status: "success" });
      } finally {
        state.initializing = false;
      }
      return;
    }
    if (type === "generate") {
      if (state.initializing || !state.initialized || !model)
        throw Error("Call init before generate");
      if (state.generating) throw Error("Generation already in progress");
      state.generating = true;
      try {
        validateConfig(data);
        send(requestId, "generated", {
          status: "success",
          result: await generate(data),
        });
      } finally {
        state.generating = false;
      }
      return;
    }
    if (type === "generateStream")
      throw Error("Stream generation is not supported");
    if (type === "clearMemory") {
      if (state.initializing || state.generating) throw Error("Worker is busy");
      state.initializing = true;
      try {
        await clear();
        send(requestId, type, { status: "success" });
      } finally {
        state.initializing = false;
      }
      return;
    }
    throw Error(`Unknown operation: ${type}`);
  } catch (error) {
    send(requestId, "error", {
      message: error instanceof Error ? error.message : String(error),
      context: type,
    });
  }
});
self.postMessage({
  type: "worker initialized",
  data: { success: true, workerVersion: "v4", modelId: ID },
});
