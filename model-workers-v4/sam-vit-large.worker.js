import {
  AutoProcessor,
  env,
  RawImage,
  SamModel,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0";

const ID = "sam-vit-large";
const REPO = "Xenova/sam-vit-large";
const REV = "bad3c600df2df287ffbd1592849efbad608399e6";

const PRECISIONS = {
  q8: {
    weightsSize: 330163356,
    size: 330166460,
    modelKeys: ["vision_encoder_quantized.onnx", "prompt_encoder_mask_decoder_quantized.onnx"],
    dtype: { vision_encoder: "q8", prompt_encoder_mask_decoder: "q8" },
    candidateDevices: ["wasm", "webgpu"],
  },
  fp16: {
    weightsSize: 626913014,
    size: 626916118,
    modelKeys: ["vision_encoder_fp16.onnx", "prompt_encoder_mask_decoder_fp16.onnx"],
    dtype: { vision_encoder: "fp16", prompt_encoder_mask_decoder: "fp16" },
    candidateDevices: ["wasm", "webgpu"],
  },
  fp32: {
    weightsSize: 1251004231,
    size: 1251007335,
    modelKeys: ["vision_encoder.onnx", "prompt_encoder_mask_decoder.onnx"],
    dtype: { vision_encoder: "fp32", prompt_encoder_mask_decoder: "fp32" },
    candidateDevices: ["wasm", "webgpu"],
  },
};

const SUPPORTED = {
  q8: { ...PRECISIONS.q8, supportedDevices: ["wasm","webgpu"] },
  fp16: { ...PRECISIONS.fp16, supportedDevices: ["webgpu"] },
  fp32: { ...PRECISIONS.fp32, supportedDevices: ["webgpu"] },
};

const MANIFEST = {
  contractVersion: "1.0",
  model: {
    id: ID,
    displayName: "SAM ViT Large — Point-Prompt Segmentation",
    provider: "Meta AI",
    providerUrl: "https://ai.meta.com/research/publications/segment-anything/",
    license: "Apache-2.0",
    licenseUrl: "https://huggingface.co/facebook/sam-vit-large/blob/main/LICENSE",
    lastUpdated: "2024-01-11T19:23:46.000Z",
    sourceRepository: "nielsr/sam-vit-large",
    repository: REPO,
    artifactRevision: REV,
    artifactLastUpdated: "2026-03-18T23:09:28.000Z",
    task: "mask-generation",
    description: "Meta AI’s large Segment Anything vision transformer, exported by Xenova for browser inference, generates candidate object masks from positive or negative point prompts.",
    intendedUses: [
      "Interactive foreground selection and image editing prototypes.",
      "Generating candidate masks around user-selected image points.",
    ],
    limitations: [
      "Mask quality depends on image quality, prompt placement, object boundaries, and scene complexity.",
      "Small, transparent, occluded, or visually ambiguous objects may be missed or merged.",
      "Predicted IoU scores are confidence estimates, not guarantees of accurate segmentation.",
      "Masks may reflect dataset biases and must not be used as the sole basis for safety-critical, medical, identity, or access decisions.",
    ],
  },
  runtime: {
    engine: { name: "Transformers.js", version: "4.2.0" },
    precisions: SUPPORTED,
  },
  input: {
    description: "A browser-readable image URL and one prompt group containing image-space points with matching labels.",
    alternatives: [{
      name: "image URL and points",
      description: "A browser-readable image plus positive or negative image-space point prompts.",
      fields: [
        { name: "image", type: "string(URL)", required: true, description: "A URL readable by the worker." },
        { name: "input_points", type: "number[][][]", required: true, description: "Groups of [x, y] image-space points." },
        { name: "input_labels", type: "number[][]", required: false, description: "Labels matching every point: 1 positive, 0 negative." },
      ],
    }],
  },
  config: { model: [], generation: [] },
  output: {
    description: "The highest-scoring post-processed boolean mask plus every candidate IoU score.",
    fields: [
      { name: "mask", type: "Uint8Array" },
      { name: "width", type: "integer" },
      { name: "height", type: "integer" },
      { name: "scores", type: "number[]" },
      { name: "bestIndex", type: "integer" },
    ],
    example: { mask: "Uint8Array", width: 640, height: 480, scores: [0.9], bestIndex: 0 },
  },
  operations: ["checkModelSupports", "init", "download", "generate", "clearMemory"],
  benchmark: {
    report: "benchmarks/sam-vit-large.json",
    testedAt: "2026-09-12T18:46:29.240Z",
    fixture: {
      file: "slimsam-77-uniform.json",
      url: "https://f005.backblazeb2.com/file/hubters-webai-public/assets/benchmarks/slimsam-77-uniform.json",
      size: 404,
      sha1: "f03c558b1d0dfd0dcb2db0dcaffbd9038aac6cf8",
    },
    note: "Measurements are specific to the tested browser and hardware.",
    environment: "Isolated headless Chrome on macOS; Transformers.js 4.2.0",
    candidates: PRECISIONS,
    qualityGate: "Finite candidate scores, expected output dimensions, selected point inside the best mask, and a non-degenerate foreground ratio.",
    invalidatedBy: ["artifact revision change", "runtime version change", "fixture or quality gate change", "browser/backend change"],
    results: [{"precision":"q8","device":"wasm","status":"pass","runtime":{"megapixelsPerSecond":0.0103,"grade":"very-slow"},"quality":{"grade":"segmentation-pass","pointInside":true}},{"precision":"q8","device":"webgpu","status":"pass","runtime":{"megapixelsPerSecond":0.0118,"grade":"very-slow"},"quality":{"grade":"segmentation-pass","pointInside":true}},{"precision":"fp16","device":"wasm","status":"fail","failure":"Error: Can't create a session. ERROR_CODE: 1, ERROR_MESSAGE: /mnt/vss/_work/1/s/onnxruntime/core/graph/graph_utils.cc:30 int onnxruntime::graph_utils::GetIndexFromName(const Node &, const std::string &, bool) itr != node_args.end() was false. Attempting to get index by a name which does not exist:InsertedPrecisionFreeCast_/mask_decoder/upscale_layer_norm/Constant_output_0for node: /mask_decoder/transformer/layers.0/layer_norm1/Mul/SimplifiedLayerNormFusion/\n"},{"precision":"fp16","device":"webgpu","status":"pass","runtime":{"megapixelsPerSecond":0.3004,"grade":"slow"},"quality":{"grade":"segmentation-pass","pointInside":true}},{"precision":"fp32","device":"wasm","status":"fail","failure":"Error: failed to call OrtRun(). ERROR_CODE: 6, ERROR_MESSAGE: std::bad_alloc"},{"precision":"fp32","device":"webgpu","status":"pass","runtime":{"megapixelsPerSecond":0.2666,"grade":"slow"},"quality":{"grade":"segmentation-pass","pointInside":true}}],
  },
};

const state = { initializing: false, initialized: false, generating: false };
let model = null;
let processor = null;

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

const send = (requestId, type, data) => self.postMessage({ requestId, type, data });

function configure(config = {}) {
  env.allowRemoteModels = config.allowRemoteModels ?? true;
  env.allowLocalModels = config.allowLocalModels ?? false;
  env.useBrowserCache = config.useBrowserCache ?? true;
  if (config.remoteHost) env.remoteHost = config.remoteHost;
  if (config.remotePathTemplate) env.remotePathTemplate = config.remotePathTemplate;
  if (config.localModelPath) env.localModelPath = config.localModelPath;
}

function validateInit(data) {
  const precision = data?.precision;
  const device = data?.device;
  const supported = (data?.benchmark === true ? PRECISIONS : SUPPORTED)[precision];
  if (!supported) throw Error(`Unsupported precision: ${precision}`);
  const devices = supported.supportedDevices ?? supported.candidateDevices;
  if (!devices.includes(device)) throw Error(`Precision ${precision} is only supported on: ${devices.join(", ")}`);
  if (device === "webgpu" && !self.navigator?.gpu) throw Error("WebGPU is unavailable");
  return { precision, device };
}

async function clear() {
  if (model?.dispose) await model.dispose();
  model = null;
  processor = null;
  state.initialized = false;
}

async function load(data, requestId) {
  const { precision, device } = validateInit(data);
  await clear();
  model = await SamModel.from_pretrained(REPO, {
    revision: REV,
    dtype: PRECISIONS[precision].dtype,
    device,
    progress_callback: (value) => send(requestId, "downloadProgress", value),
  });
  processor = await AutoProcessor.from_pretrained(REPO, { revision: REV });
}

function validateInput(data) {
  const { image, input_points: points, input_labels: labels } = data?.userInput ?? {};
  if (typeof image !== "string" || !image.trim()) throw Error("userInput.image must be a non-empty browser-readable URL");
  if (!Array.isArray(points) || !points.length || !points.every((group) => Array.isArray(group) && group.length && group.every((point) => Array.isArray(point) && point.length === 2 && point.every(Number.isFinite)))) {
    throw Error("userInput.input_points must be a non-empty number[][][] of [x, y] points");
  }
  if (labels !== undefined && (!Array.isArray(labels) || labels.length !== points.length || !labels.every((group, index) => Array.isArray(group) && group.length === points[index].length && group.every((label) => label === 0 || label === 1)))) {
    throw Error("userInput.input_labels must match input_points and contain only 0 or 1");
  }
  return { image, points, labels };
}

async function generate(data) {
  const { image, points, labels } = validateInput(data);
  const rawImage = await RawImage.read(image);
  const inputs = await processor(rawImage, { input_points: points, input_labels: labels });
  const outputs = await model(inputs);
  const masks = await processor.post_process_masks(outputs.pred_masks, inputs.original_sizes, inputs.reshaped_input_sizes);
  const tensor = masks[0];
  const scores = Array.from(outputs.iou_scores.data, Number);
  const bestIndex = scores.reduce((best, score, index) => score > scores[best] ? index : best, 0);
  const [, candidates, height, width] = tensor.dims;
  const area = width * height;
  const offset = bestIndex * area;
  const mask = Uint8Array.from(tensor.data.slice(offset, offset + area));
  return { mask, width, height, scores, bestIndex, candidates };
}

self.addEventListener("message", async ({ data: { requestId, type, data } = {} }) => {
  try {
    if (type === "checkModelSupports") {
      configure(data?.workerConfig);
      send(requestId, type, { ...caps, manifest: MANIFEST });
      return;
    }
    if (type === "download") {
      if (state.initializing || state.generating) throw Error("Worker is busy");
      await load(data, requestId);
      await clear();
      send(requestId, type, { status: "success" });
      return;
    }
    if (type === "init") {
      if (state.initializing || state.generating) throw Error("Worker is busy");
      state.initializing = true;
      try {
        await load(data, requestId);
        state.initialized = true;
        send(requestId, type, { status: "success" });
      } finally {
        state.initializing = false;
      }
      return;
    }
    if (type === "generate") {
      if (!state.initialized || !model || !processor) throw Error("Call init before generate");
      if (state.generating) throw Error("Generation already in progress");
      state.generating = true;
      try {
        send(requestId, "generated", { status: "success", result: await generate(data) });
      } finally {
        state.generating = false;
      }
      return;
    }
    if (type === "generateStream") throw Error("Stream generation is not supported for mask generation");
    if (type === "clearMemory") {
      if (state.generating) throw Error("Cannot clear memory while generation is active");
      await clear();
      send(requestId, type, { status: "success" });
      return;
    }
    throw Error(`Unknown operation: ${type}`);
  } catch (error) {
    send(requestId, "error", { message: error instanceof Error ? error.message : String(error), context: type });
  }
});

self.postMessage({ type: "worker initialized", data: { success: true, workerVersion: "v4", modelId: ID } });
