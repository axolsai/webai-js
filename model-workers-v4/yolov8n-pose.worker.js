import {
  AutoModel,
  AutoProcessor,
  RawImage,
  env,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0";

const ID = "yolov8n-pose",
  REPO = "Xenova/yolov8n-pose",
  REV = "da4224085e2f6ce4c9ff9b670e28765194619db2",
  SHARED = 1071;
const LABELS = [
  "nose",
  "left_eye",
  "right_eye",
  "left_ear",
  "right_ear",
  "left_shoulder",
  "right_shoulder",
  "left_elbow",
  "right_elbow",
  "left_wrist",
  "right_wrist",
  "left_hip",
  "right_hip",
  "left_knee",
  "right_knee",
  "left_ankle",
  "right_ankle",
];
const candidate = (weightsSize, suffix) => ({
  weightsSize,
  size: weightsSize + SHARED,
  modelKeys: [`model${suffix}.onnx`],
  candidateDevices: ["wasm", "webgpu"],
});
const PRECISIONS = {
  fp32: candidate(13484153, ""),
  fp16: candidate(6789736, "_fp16"),
  int8: candidate(3755740, "_int8"),
  uint8: candidate(3755780, "_uint8"),
  q8: candidate(3755780, "_quantized"),
  q4: candidate(13484172, "_q4"),
  bnb4: candidate(13484172, "_bnb4"),
};
const SUPPORTED = {
  fp32: { ...PRECISIONS.fp32, supportedDevices: ["wasm", "webgpu"] },
  fp16: { ...PRECISIONS.fp16, supportedDevices: ["wasm", "webgpu"] },
  int8: { ...PRECISIONS.int8, supportedDevices: ["wasm", "webgpu"] },
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
      loadMs: 2838.6,
      inferenceMs: 275.3,
      totalMs: 3208.9,
      imagesPerSecond: 3.632,
      grade: "good",
    },
    quality: {
      topPersonScore: 0.857,
      visibleKeypoints: 13,
      signatureMeanAbsoluteError: 0,
      grade: "excellent",
    },
  },
  {
    precision: "fp32",
    device: "webgpu",
    status: "pass",
    runtime: {
      loadMs: 2807,
      inferenceMs: 706,
      totalMs: 3607.8,
      imagesPerSecond: 1.416,
      grade: "slow",
    },
    quality: {
      topPersonScore: 0.857,
      visibleKeypoints: 13,
      signatureMeanAbsoluteError: 0,
      grade: "excellent",
    },
  },
  {
    precision: "fp16",
    device: "wasm",
    status: "pass",
    runtime: {
      loadMs: 2619.9,
      inferenceMs: 298.5,
      totalMs: 3023.5,
      imagesPerSecond: 3.35,
      grade: "good",
    },
    quality: {
      topPersonScore: 0.857,
      visibleKeypoints: 13,
      signatureMeanAbsoluteError: 0.000141,
      grade: "excellent",
    },
  },
  {
    precision: "fp16",
    device: "webgpu",
    status: "pass",
    runtime: {
      loadMs: 2632.2,
      inferenceMs: 729.5,
      totalMs: 3457,
      imagesPerSecond: 1.371,
      grade: "slow",
    },
    quality: {
      topPersonScore: 0.857,
      visibleKeypoints: 13,
      signatureMeanAbsoluteError: 0.000949,
      grade: "excellent",
    },
  },
  {
    precision: "int8",
    device: "wasm",
    status: "pass",
    runtime: {
      loadMs: 2615.2,
      inferenceMs: 733.5,
      totalMs: 3485.4,
      imagesPerSecond: 1.363,
      grade: "slow",
    },
    quality: {
      topPersonScore: 0.868,
      visibleKeypoints: 13,
      signatureMeanAbsoluteError: 0.011018,
      grade: "good",
    },
  },
  {
    precision: "int8",
    device: "webgpu",
    status: "pass",
    runtime: {
      loadMs: 2685.4,
      inferenceMs: 1025.6,
      totalMs: 3807.1,
      imagesPerSecond: 0.975,
      grade: "very-slow",
    },
    quality: {
      topPersonScore: 0.872,
      visibleKeypoints: 13,
      signatureMeanAbsoluteError: 0.011646,
      grade: "good",
    },
  },
  {
    precision: "uint8",
    device: "wasm",
    status: "pass",
    runtime: {
      loadMs: 2583.4,
      inferenceMs: 310.3,
      totalMs: 2998.6,
      imagesPerSecond: 3.223,
      grade: "good",
    },
    quality: {
      topPersonScore: 0.619,
      visibleKeypoints: 13,
      signatureMeanAbsoluteError: 0.015296,
      grade: "good",
    },
  },
  {
    precision: "uint8",
    device: "webgpu",
    status: "pass",
    runtime: {
      loadMs: 8897.7,
      inferenceMs: 392.4,
      totalMs: 9442,
      imagesPerSecond: 2.548,
      grade: "slow",
    },
    quality: {
      topPersonScore: 0.633,
      visibleKeypoints: 13,
      signatureMeanAbsoluteError: 0.019261,
      grade: "good",
    },
  },
  {
    precision: "q8",
    device: "wasm",
    status: "pass",
    runtime: {
      loadMs: 2761.7,
      inferenceMs: 297.2,
      totalMs: 3241.9,
      imagesPerSecond: 3.365,
      grade: "good",
    },
    quality: {
      topPersonScore: 0.619,
      visibleKeypoints: 13,
      signatureMeanAbsoluteError: 0.015296,
      grade: "good",
    },
  },
  {
    precision: "q8",
    device: "webgpu",
    status: "pass",
    runtime: {
      loadMs: 3049,
      inferenceMs: 389.2,
      totalMs: 3546.4,
      imagesPerSecond: 2.569,
      grade: "slow",
    },
    quality: {
      topPersonScore: 0.633,
      visibleKeypoints: 13,
      signatureMeanAbsoluteError: 0.019261,
      grade: "good",
    },
  },
  {
    precision: "q4",
    device: "wasm",
    status: "pass",
    runtime: {
      loadMs: 2911.1,
      inferenceMs: 276.6,
      totalMs: 3469.5,
      imagesPerSecond: 3.615,
      grade: "good",
    },
    quality: {
      topPersonScore: 0.857,
      visibleKeypoints: 13,
      signatureMeanAbsoluteError: 0,
      grade: "excellent",
    },
  },
  {
    precision: "q4",
    device: "webgpu",
    status: "pass",
    runtime: {
      loadMs: 2837.3,
      inferenceMs: 134.4,
      totalMs: 3067.1,
      imagesPerSecond: 7.44,
      grade: "good",
    },
    quality: {
      topPersonScore: 0.857,
      visibleKeypoints: 13,
      signatureMeanAbsoluteError: 0,
      grade: "excellent",
    },
  },
  {
    precision: "bnb4",
    device: "wasm",
    status: "pass",
    runtime: {
      loadMs: 3494.8,
      inferenceMs: 275.7,
      totalMs: 3909.7,
      imagesPerSecond: 3.627,
      grade: "good",
    },
    quality: {
      topPersonScore: 0.857,
      visibleKeypoints: 13,
      signatureMeanAbsoluteError: 0,
      grade: "excellent",
    },
  },
  {
    precision: "bnb4",
    device: "webgpu",
    status: "pass",
    runtime: {
      loadMs: 3169.7,
      inferenceMs: 136.4,
      totalMs: 3424.2,
      imagesPerSecond: 7.331,
      grade: "good",
    },
    quality: {
      topPersonScore: 0.857,
      visibleKeypoints: 13,
      signatureMeanAbsoluteError: 0,
      grade: "excellent",
    },
  },
];
const CONFIG_FIELDS = [
  {
    name: "confidenceThreshold",
    type: "number",
    default: 0.3,
    minimum: 0,
    maximum: 1,
    description: "Minimum person-box confidence.",
  },
  {
    name: "iouThreshold",
    type: "number",
    default: 0.5,
    minimum: 0,
    maximum: 1,
    description:
      "Intersection-over-union threshold for non-maximum suppression.",
  },
  {
    name: "keypointThreshold",
    type: "number",
    default: 0.3,
    minimum: 0,
    maximum: 1,
    description: "Minimum confidence for a keypoint to be marked visible.",
  },
];
const MANIFEST = {
  contractVersion: "1.0",
  model: {
    id: ID,
    displayName: "YOLOv8 Nano Pose",
    provider: "Ultralytics",
    providerUrl: "https://www.ultralytics.com/",
    license: "AGPL-3.0-only",
    licenseUrl: "https://github.com/ultralytics/ultralytics/blob/main/LICENSE",
    lastUpdated: "2024-04-17T05:32:48.000Z",
    sourceRepository: "ultralytics/assets",
    sourceRevision: "v8.2.0",
    repository: REPO,
    artifactRevision: REV,
    artifactLastUpdated: "2025-08-18T15:41:44.000Z",
    task: "pose-estimation",
    description:
      "The nano YOLOv8 pose model converted to ONNX for browser-local inference. It detects people and returns bounding boxes plus 17 COCO-style body keypoints per detection.",
    intendedUses: [
      "Interactive, user-reviewed human-pose overlays in browser applications.",
      "Fitness, animation, and accessibility prototypes where approximate 2D landmarks are sufficient.",
      "Benchmarking browser pose estimation across ONNX precisions.",
    ],
    limitations: [
      "Small, occluded, blurred, crowded, rotated, or partially visible people can be missed or assigned incorrect landmarks.",
      "Two-dimensional keypoints do not establish depth, identity, intent, health, or biomechanical safety.",
      "Accuracy can vary across body types, clothing, skin tones, camera viewpoints, lighting, and activities represented unevenly in training data.",
      "Do not use outputs for biometric identification, surveillance, medical diagnosis, employment, law enforcement, or other consequential decisions.",
    ],
  },
  runtime: {
    engine: { name: "Transformers.js", version: "4.2.0" },
    precisions: SUPPORTED,
  },
  input: {
    description: "One browser-readable image URL.",
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
            description: "HTTP(S), data, or blob URL readable by the worker.",
          },
        ],
      },
    ],
  },
  config: { model: CONFIG_FIELDS, generation: [] },
  output: {
    description:
      "Detected people with source-image pixel boxes and 17 labeled source-image pixel keypoints.",
    fields: [
      {
        name: "persons",
        type: "Array<{score:number,box:{x1:number,y1:number,x2:number,y2:number},keypoints:Array<{label:string,x:number,y:number,score:number,visible:boolean}>}>",
      },
      { name: "width", type: "integer", required: true, minimum: 1 },
      { name: "height", type: "integer", required: true, minimum: 1 },
    ],
    example: {
      persons: [
        {
          score: 0.9,
          box: { x1: 10, y1: 20, x2: 200, y2: 400 },
          keypoints: [
            { label: "nose", x: 100, y: 50, score: 0.95, visible: true },
          ],
        },
      ],
      width: 640,
      height: 480,
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
    report: "benchmarks/yolov8n-pose.json",
    testedAt: "2026-10-23T13:09:50.699Z",
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
        "Decode RGB, resize directly to 640×640, rescale pixels by 1/255.",
      expected: {
        file: "yolov8n-pose.json",
        url: "https://f005.backblazeb2.com/file/hubters-webai-public/assets/benchmarks/yolov8n-pose.json",
        size: 1901,
        sha1: "91cac63bf855b61c269ac362e4c93ae5df5c8e45",
        description:
          "FP32/WASM normalized top-person box, confidence, and 17-keypoint signature.",
      },
    },
    candidates: PRECISIONS,
    qualityGate:
      "At least one finite person pose with a confidence of 0.5 or greater, at least five visible keypoints, valid source-image coordinates, and a normalized pose signature within 0.08 mean absolute error of the checked-in FP32/WASM reference.",
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
const state = { initializing: false, initialized: false, generating: false };
let model = null,
  processor = null;
const send = (requestId, type, data) =>
  self.postMessage({ requestId, type, data });
function configure(c = {}) {
  env.allowRemoteModels = c.allowRemoteModels ?? true;
  env.allowLocalModels = c.allowLocalModels ?? false;
  env.useBrowserCache = c.useBrowserCache ?? true;
  if (c.remoteHost) env.remoteHost = c.remoteHost;
  if (c.remotePathTemplate) env.remotePathTemplate = c.remotePathTemplate;
  if (c.localModelPath) env.localModelPath = c.localModelPath;
}
function validateInit(data) {
  const precision = data?.precision,
    device = data?.device,
    s = SUPPORTED[precision];
  if (!s) throw Error(`Unsupported precision: ${precision}`);
  if (!s.supportedDevices.includes(device))
    throw Error(
      `Precision ${precision} is only supported on: ${s.supportedDevices.join(", ")}`,
    );
  if (device === "webgpu" && !navigator.gpu) throw Error("WebGPU unavailable");
  return { precision, device };
}
async function clear() {
  if (model?.dispose) await model.dispose();
  model = null;
  processor = null;
  state.initialized = false;
}
async function load(data, id) {
  const { precision, device } = validateInit(data);
  await clear();
  const common = {
    revision: REV,
    progress_callback: (value) =>
      send(id, "downloadProgress", {
        ...value,
        progress: value.status === "ready" ? 100 : value.progress,
      }),
  };
  processor = await AutoProcessor.from_pretrained(REPO, common);
  model = await AutoModel.from_pretrained(REPO, {
    ...common,
    dtype: precision,
    device,
  });
}
function thresholds(data) {
  const values = {
    confidenceThreshold: data?.modelConfig?.confidenceThreshold ?? 0.3,
    iouThreshold: data?.modelConfig?.iouThreshold ?? 0.5,
    keypointThreshold: data?.modelConfig?.keypointThreshold ?? 0.3,
  };
  for (const [name, value] of Object.entries(values))
    if (!Number.isFinite(value) || value < 0 || value > 1)
      throw Error(
        `modelConfig.${name} must be a finite number from 0 through 1`,
      );
  return values;
}
function iou(a, b) {
  const w = Math.max(
      0,
      Math.min(a.box.x2, b.box.x2) - Math.max(a.box.x1, b.box.x1),
    ),
    h = Math.max(
      0,
      Math.min(a.box.y2, b.box.y2) - Math.max(a.box.y1, b.box.y1),
    ),
    intersection = w * h,
    areaA = Math.max(0, a.box.x2 - a.box.x1) * Math.max(0, a.box.y2 - a.box.y1),
    areaB = Math.max(0, b.box.x2 - b.box.x1) * Math.max(0, b.box.y2 - b.box.y1);
  return intersection / (areaA + areaB - intersection || 1);
}
async function generate(data) {
  const image = data?.userInput?.image;
  if (typeof image !== "string" || !image.trim())
    throw Error("userInput.image must be a non-empty URL");
  const { confidenceThreshold, iouThreshold, keypointThreshold } =
      thresholds(data),
    raw = await RawImage.read(image),
    { pixel_values } = await processor(raw),
    { output0 } = await model({ images: pixel_values }),
    [, features, candidates] = output0.dims;
  if (features !== 56)
    throw Error(`Unexpected YOLO pose feature count: ${features}`);
  const values = output0.data,
    sx = raw.width / pixel_values.dims.at(-1),
    sy = raw.height / pixel_values.dims.at(-2),
    found = [];
  for (let i = 0; i < candidates; i++) {
    const value = (c) => Number(values[c * candidates + i]),
      score = value(4);
    if (!Number.isFinite(score) || score < confidenceThreshold) continue;
    const xc = value(0) * sx,
      yc = value(1) * sy,
      w = value(2) * sx,
      h = value(3) * sy,
      keypoints = [];
    for (let k = 0; k < 17; k++) {
      const x = value(5 + k * 3) * sx,
        y = value(6 + k * 3) * sy,
        pointScore = value(7 + k * 3);
      if (![x, y, pointScore].every(Number.isFinite))
        throw Error("Model returned non-finite keypoints");
      keypoints.push({
        label: LABELS[k],
        x,
        y,
        score: pointScore,
        visible: pointScore >= keypointThreshold,
      });
    }
    found.push({
      score,
      box: {
        x1: Math.max(0, xc - w / 2),
        y1: Math.max(0, yc - h / 2),
        x2: Math.min(raw.width, xc + w / 2),
        y2: Math.min(raw.height, yc + h / 2),
      },
      keypoints,
    });
  }
  found.sort((a, b) => b.score - a.score);
  const persons = [];
  for (const pose of found)
    if (!persons.some((kept) => iou(pose, kept) > iouThreshold))
      persons.push(pose);
  return { persons, width: raw.width, height: raw.height };
}
self.addEventListener(
  "message",
  async ({ data: { requestId, type, data } = {} }) => {
    try {
      if (type === "checkModelSupports") {
        configure(data?.workerConfig);
        send(requestId, type, { ...caps, manifest: MANIFEST });
        return;
      }
      if (type === "download") {
        const s = SUPPORTED[data?.precision];
        if (!s) throw Error(`Unsupported precision: ${data?.precision}`);
        await load(
          {
            ...data,
            device: s.supportedDevices.includes("wasm")
              ? "wasm"
              : s.supportedDevices[0],
          },
          requestId,
        );
        await clear();
        send(requestId, type, { status: "success" });
        return;
      }
      if (type === "init") {
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
        return;
      }
      if (type === "generate") {
        if (!state.initialized || !model)
          throw Error("Call init before generate");
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
        return;
      }
      if (type === "generateStream")
        throw Error("Stream generation is not supported by this worker");
      if (type === "clearMemory") {
        if (state.generating)
          throw Error("Cannot clear memory while generation is active");
        await clear();
        send(requestId, type, { status: "success" });
        return;
      }
      throw Error(`Unknown operation: ${type}`);
    } catch (error) {
      send(requestId, "error", {
        message: error instanceof Error ? error.message : String(error),
        context: type,
      });
    }
  },
);
self.postMessage({
  type: "worker initialized",
  data: { success: true, workerVersion: "v4", modelId: ID },
});
