import {
  env,
  pipeline,
  RawImage,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0";

const ID = "swin2sr-realworld-sr-x4-64-bsrgan-psnr",
  REPO = "Xenova/swin2SR-realworld-sr-x4-64-bsrgan-psnr",
  REV = "d0e9926970c93e472ce2392373d72597fc849027";
const PRECISIONS = {
  fp32: {
    size: 52774772,
    weightsSize: 52772645,
    modelKeys: ["model.onnx"],
    candidateDevices: ["wasm", "webgpu"],
  },
  fp16: {
    size: 32359649,
    weightsSize: 32357522,
    modelKeys: ["model_fp16.onnx"],
    candidateDevices: ["wasm", "webgpu"],
  },
  uint8: {
    size: 19001760,
    weightsSize: 18999633,
    modelKeys: ["model_uint8.onnx"],
    candidateDevices: ["wasm", "webgpu"],
  },
  q8: {
    size: 21440749,
    weightsSize: 21438622,
    modelKeys: ["model_quantized.onnx"],
    candidateDevices: ["wasm", "webgpu"],
  },
  q4: {
    size: 23049463,
    weightsSize: 23047336,
    modelKeys: ["model_q4.onnx"],
    candidateDevices: ["wasm", "webgpu"],
  },
  q4f16: {
    size: 15543472,
    weightsSize: 15541345,
    modelKeys: ["model_q4f16.onnx"],
    candidateDevices: ["wasm", "webgpu"],
  },
  bnb4: {
    size: 22079335,
    weightsSize: 22077208,
    modelKeys: ["model_bnb4.onnx"],
    candidateDevices: ["wasm", "webgpu"],
  },
};
const SUPPORTED = {
  fp32: { ...PRECISIONS.fp32, supportedDevices: ["wasm", "webgpu"] },
  uint8: { ...PRECISIONS.uint8, supportedDevices: ["wasm", "webgpu"] },
  q8: { ...PRECISIONS.q8, supportedDevices: ["wasm", "webgpu"] },
  q4: { ...PRECISIONS.q4, supportedDevices: ["wasm", "webgpu"] },
  bnb4: { ...PRECISIONS.bnb4, supportedDevices: ["wasm", "webgpu"] },
};
const RESULTS = [
  {
    precision: "fp32",
    device: "wasm",
    status: "pass",
    runtime: {
      loadMs: 11523.2,
      inferenceMs: 6849.6,
      totalMs: 19818.3,
      imagesPerSecond: 0.146,
      grade: "very-slow",
    },
    quality: { psnrDb: 27.12, grade: "excellent" },
  },
  {
    precision: "fp32",
    device: "webgpu",
    status: "pass",
    runtime: {
      loadMs: 12121.7,
      inferenceMs: 397.7,
      totalMs: 13840.9,
      imagesPerSecond: 2.514,
      grade: "good",
    },
    quality: { psnrDb: 27.12, grade: "excellent" },
  },
  {
    precision: "fp16",
    device: "wasm",
    status: "fail",
    runtime: {
      loadMs: null,
      inferenceMs: null,
      totalMs: 11975.5,
      grade: "failed",
    },
    failure: "onnx-session-error",
  },
  {
    precision: "fp16",
    device: "webgpu",
    status: "fail",
    runtime: {
      loadMs: 15326.2,
      inferenceMs: null,
      totalMs: 17759.7,
      grade: "failed",
    },
    failure: "onnx-shape-mismatch",
  },
  {
    precision: "uint8",
    device: "wasm",
    status: "pass",
    runtime: {
      loadMs: 9572.6,
      inferenceMs: 6931.5,
      totalMs: 17882.5,
      imagesPerSecond: 0.144,
      grade: "very-slow",
    },
    quality: { psnrDb: 27.16, grade: "excellent" },
  },
  {
    precision: "uint8",
    device: "webgpu",
    status: "pass",
    runtime: {
      loadMs: 10696.8,
      inferenceMs: 6415.9,
      totalMs: 18434.4,
      imagesPerSecond: 0.156,
      grade: "very-slow",
    },
    quality: { psnrDb: 27.15, grade: "excellent" },
  },
  {
    precision: "q8",
    device: "wasm",
    status: "pass",
    runtime: {
      loadMs: 15775.6,
      inferenceMs: 6763.4,
      totalMs: 24122.2,
      imagesPerSecond: 0.148,
      grade: "very-slow",
    },
    quality: { psnrDb: 26.99, grade: "excellent" },
  },
  {
    precision: "q8",
    device: "webgpu",
    status: "pass",
    runtime: {
      loadMs: 15365.2,
      inferenceMs: 6206.4,
      totalMs: 23047.8,
      imagesPerSecond: 0.161,
      grade: "very-slow",
    },
    quality: { psnrDb: 26.99, grade: "excellent" },
  },
  {
    precision: "q4",
    device: "wasm",
    status: "pass",
    runtime: {
      loadMs: 9977,
      inferenceMs: 6993.2,
      totalMs: 18393.4,
      imagesPerSecond: 0.143,
      grade: "very-slow",
    },
    quality: { psnrDb: 27.14, grade: "excellent" },
  },
  {
    precision: "q4",
    device: "webgpu",
    status: "pass",
    runtime: {
      loadMs: 9797.6,
      inferenceMs: 372.1,
      totalMs: 11544.2,
      imagesPerSecond: 2.687,
      grade: "good",
    },
    quality: { psnrDb: 27.14, grade: "excellent" },
  },
  {
    precision: "q4f16",
    device: "wasm",
    status: "fail",
    runtime: {
      loadMs: null,
      inferenceMs: null,
      totalMs: 9167.1,
      grade: "failed",
    },
    failure: "onnx-session-error",
  },
  {
    precision: "q4f16",
    device: "webgpu",
    status: "fail",
    runtime: {
      loadMs: 9297.8,
      inferenceMs: 578.2,
      totalMs: 11193.8,
      imagesPerSecond: 1.73,
      grade: "good",
    },
    quality: { psnrDb: 11.06, grade: "failed" },
    failure: "quality-psnr",
  },
  {
    precision: "bnb4",
    device: "wasm",
    status: "pass",
    runtime: {
      loadMs: 9033,
      inferenceMs: 6819.9,
      totalMs: 17391.6,
      imagesPerSecond: 0.147,
      grade: "very-slow",
    },
    quality: { psnrDb: 27.24, grade: "excellent" },
  },
  {
    precision: "bnb4",
    device: "webgpu",
    status: "pass",
    runtime: {
      loadMs: 10378.1,
      inferenceMs: 4919.4,
      totalMs: 16723.3,
      imagesPerSecond: 0.203,
      grade: "slow",
    },
    quality: { psnrDb: 27.24, grade: "excellent" },
  },
];
const MANIFEST = {
  contractVersion: "1.0",
  model: {
    id: ID,
    displayName: "Swin2SR Real-World 4× Super-Resolution",
    provider: "Computer Vision Lab, CAIDAS, University of Würzburg",
    providerUrl: "https://www.informatik.uni-wuerzburg.de/computervision/",
    license: "Apache-2.0",
    licenseUrl: "https://github.com/mv-lab/swin2sr/blob/main/LICENSE",
    lastUpdated: "2023-01-21T12:08:28.000Z",
    sourceRepository: "caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr",
    sourceRevision: "bb13f02e45e88d00b6c202b3fbe6a181af144606",
    repository: REPO,
    artifactRevision: REV,
    artifactLastUpdated: "2025-09-30T19:20:39.000Z",
    task: "image-to-image",
    description:
      "A browser-local Swin2SR model for enlarging real-world RGB images by 4× and reducing common blur, noise, and compression artifacts. It returns raw uint8 RGB pixels plus output dimensions.",
    intendedUses: [
      "Enhancing low-resolution photographs for user-reviewed display.",
      "Prototyping browser image restoration and 4× upscaling.",
      "Benchmarking ONNX super-resolution backends in browsers.",
    ],
    limitations: [
      "The model can hallucinate or alter fine details and must not be used to recover evidentiary, medical, identity, or safety-critical information.",
      "Faces, text, repeated patterns, heavy blur, unusual noise, and out-of-distribution images can produce incorrect or artificial details.",
      "Output is always 4× the decoded input dimensions, increasing memory use by about 16× for pixels.",
      "Quality depends on how closely the degradation resembles BSRGAN training data; clean bicubic inputs are not guaranteed to improve by every fidelity metric.",
    ],
  },
  runtime: {
    engine: { name: "Transformers.js", version: "4.2.0" },
    precisions: SUPPORTED,
  },
  input: {
    description: "One browser-readable RGB image URL.",
    alternatives: [
      {
        name: "image",
        fields: [
          {
            name: "image",
            type: "string(URL)",
            required: true,
            minLength: 1,
            format: "url",
            description:
              "HTTP(S), data, or blob URL readable by the module worker.",
          },
        ],
      },
    ],
  },
  config: { model: [], generation: [] },
  output: {
    description:
      "Raw 4× RGB image pixels. WebAI wraps this native value as { result, runtime }.",
    fields: [
      { name: "data", type: "Uint8Array", required: true },
      { name: "width", type: "integer", required: true, minimum: 1 },
      { name: "height", type: "integer", required: true, minimum: 1 },
      { name: "channels", type: "integer", required: true, enum: [3] },
    ],
    example: {
      data: "Uint8Array(442368)",
      width: 384,
      height: 384,
      channels: 3,
    },
  },
  operations: [
    "checkModelSupports",
    "init",
    "download",
    "generate",
    "clearMemory",
  ],
  benchmark: {
    report: "benchmarks/swin2sr-realworld-sr-x4-64-bsrgan-psnr.json",
      testedAt: "2026-10-23T11:27:48.607Z",
    fixture: {
      file: "swin2sr-portrait-lowres.jpg",
      url: "https://f005.backblazeb2.com/file/hubters-webai-public/assets/benchmarks/swin2sr-portrait-lowres.jpg",
      size: 5059,
      sha1: "86b8839b760031cca032e78fe516dc5bda417a1a",
      width: 96,
      height: 96,
      source:
        "https://www.pexels.com/photo/woman-wearing-pink-turtleneck-sweater-5965592/",
      license: "Pexels License",
      licenseUrl: "https://www.pexels.com/license/",
      preprocessing:
        "Center-crop the 1024×683 source to a square, resize to 384×384, then downsample to 96×96 with macOS sips.",
      expected: {
        file: "swin2sr-portrait-reference.jpg",
        url: "https://f005.backblazeb2.com/file/hubters-webai-public/assets/benchmarks/swin2sr-portrait-reference.jpg",
        size: 38541,
        sha1: "5eac874a7468aa600d6fd805aff177e411362de6",
        width: 384,
        height: 384,
      },
    },
    candidates: PRECISIONS,
    qualityGate:
      "Output must be finite, non-flat 384×384 RGB; PSNR against the checked-in real-photo reference must be at least 18 dB.",
    environment:
      "Isolated headless Google Chrome on macOS; Transformers.js 4.2.0",
    invalidatedBy: [
      "artifact revision change",
      "runtime version change",
      "fixture or quality-gate change",
      "browser/backend behavior change",
    ],
    results: RESULTS,
  },

  samples: [
  {
    "id": "standard",
    "label": "Try sample image",
    "description": "Standard sample for Swin2SR Real-World 4× Super-Resolution.",
    "userInput": {},
    "assets": [
      {
        "field": "image",
        "url": "https://f005.backblazeb2.com/file/hubters-webai-public/assets/benchmarks/swin2sr-portrait-lowres.jpg",
        "file": "model-samples/swin2sr-portrait-lowres.jpg",
        "mediaType": "image/jpeg",
        "size": 5059,
        "sha1": "86b8839b760031cca032e78fe516dc5bda417a1a",
        "encoding": "blob-url",
        "width": 96,
        "height": 96
      }
    ],
    "modelConfig": {},
    "generateConfig": {},
    "source": "https://www.pexels.com/photo/woman-wearing-pink-turtleneck-sweater-5965592/",
    "license": "Pexels License",
    "fixture": {
      "file": "model-samples/swin2sr-portrait-lowres.jpg",
      "url": "https://f005.backblazeb2.com/file/hubters-webai-public/assets/benchmarks/swin2sr-portrait-lowres.jpg",
      "size": 5059,
      "sha1": "86b8839b760031cca032e78fe516dc5bda417a1a"
    },
    "benchmarkReport": "benchmarks/swin2sr-realworld-sr-x4-64-bsrgan-psnr.json"
  }
],
};
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
let model = null;
const state = { initializing: false, initialized: false, generating: false };
function configure(c = {}) {
  env.allowRemoteModels = c.allowRemoteModels ?? true;
  env.allowLocalModels = c.allowLocalModels ?? false;
  env.useBrowserCache = c.useBrowserCache ?? true;
  if (c.remoteHost) env.remoteHost = c.remoteHost;
  if (c.remotePathTemplate) env.remotePathTemplate = c.remotePathTemplate;
  if (c.localModelPath) env.localModelPath = c.localModelPath;
}
function validate(data) {
  const p = data?.precision,
    d = data?.device,
    s = SUPPORTED[p];
  if (!s) throw Error(`Unsupported precision: ${p}`);
  if (!s.supportedDevices.includes(d))
    throw Error(
      `Precision ${p} is only supported on: ${s.supportedDevices.join(", ")}`,
    );
  if (d === "webgpu" && !navigator.gpu) throw Error("WebGPU unavailable");
  return { p, d };
}
async function clear() {
  if (model?.dispose) await model.dispose();
  model = null;
  state.initialized = false;
}
async function load(data, id) {
  const { p, d } = validate(data);
  await clear();
  model = await pipeline("image-to-image", REPO, {
    revision: REV,
    dtype: p,
    device: d,
    progress_callback: (x) =>
      send(id, "downloadProgress", {
        ...x,
        progress: x.status === "ready" ? 100 : x.progress,
      }),
  });
}
async function generate(data) {
  const image = data?.userInput?.image;
  if (typeof image !== "string" || !image.trim())
    throw Error("userInput.image must be a non-empty URL");
  const input = await RawImage.read(image),
    output = await model(input);
  if (
    !(output?.data instanceof Uint8Array) ||
    output.width !== input.width * 4 ||
    output.height !== input.height * 4 ||
    output.channels !== 3
  )
    throw Error("Model returned an invalid 4× RGB image");
  return {
    data: output.data,
    width: output.width,
    height: output.height,
    channels: output.channels,
  };
}
function send(requestId, type, data) {
  self.postMessage({ requestId, type, data });
}
self.addEventListener("message", async ({ data: m = {} }) => {
  const { requestId, type, data } = m;
  try {
    switch (type) {
      case "checkModelSupports":
        configure(data?.workerConfig);
        send(requestId, type, { ...caps, manifest: MANIFEST });
        break;
      case "init":
        if (state.initializing || state.generating)
          throw Error("Worker is busy");
        state.initializing = true;
        try {
          await load(data, requestId);
          state.initialized = true;
          send(requestId, type, { status: "success" });
        } finally {
          state.initializing = false;
        }
        break;
      case "download":
        await load({ ...data, device: data?.device ?? "wasm" }, requestId);
        send(requestId, type, { status: "success" });
        break;
      case "generate":
        if (!state.initialized) throw Error("Call init before generate");
        if (state.generating) throw Error("Generation already in progress");
        state.generating = true;
        try {
          send(requestId, "generated", {
            status: "success",
            result: await generate(data),
          });
        } finally {
          state.generating = false;
        }
        break;
      case "clearMemory":
        await clear();
        send(requestId, type, { status: "success" });
        break;
      default:
        throw Error(`Unknown operation: ${type}`);
    }
  } catch (e) {
    send(requestId, "error", {
      message: e instanceof Error ? e.message : String(e),
      context: type,
    });
  }
});
self.postMessage({
  type: "worker initialized",
  data: { success: true, workerVersion: "v4", modelId: ID },
});
