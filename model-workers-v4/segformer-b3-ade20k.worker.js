import { env, pipeline, RawImage } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0";

const ID = "segformer-b3-ade20k";
const REPO = "onnx-community/segformer-b3-finetuned-ade-512-512-ONNX";
const REVISION = "748163478112c6185cd52530d7b133d96c8236be";
const SHARED_BYTES = 7284;
const make = (weightsSize, dtype, suffix) => ({ weightsSize, size: weightsSize + SHARED_BYTES, dtype, modelKeys: [`model${suffix}.onnx`], candidateDevices: ["wasm", "webgpu"] });
const PRECISIONS = {
  fp32: make(190133539, "fp32", ""), fp16: make(95757970, "fp16", "_fp16"), int8: make(49297337, "int8", "_int8"),
  uint8: make(49297441, "uint8", "_uint8"), q8: make(49297337, "q8", "_quantized"), q4: make(77853012, "q4", "_q4"),
  q4f16: make(47922181, "q4f16", "_q4f16"), bnb4: make(77853012, "bnb4", "_bnb4"),
};
// Filled only from benchmark-passing precision/device pairs.
const SUPPORTED = Object.fromEntries(Object.entries(PRECISIONS).map(([precision, value]) => [precision, { ...value, supportedDevices: ["wasm", "webgpu"] }]));
const BENCHMARK_RESULTS = [
  ["fp32","wasm",30140.9,4752.3,"very-slow"],["fp32","webgpu",28649.6,973.1,"slow"],
  ["fp16","wasm",15895.6,5107.5,"very-slow"],["fp16","webgpu",14317.5,1124.9,"slow"],
  ["int8","wasm",10898.8,9320.8,"very-slow"],["int8","webgpu",12340.8,9439.8,"very-slow"],
  ["uint8","wasm",11711.2,4868.9,"very-slow"],["uint8","webgpu",10605.8,4926.4,"very-slow"],
  ["q8","wasm",11630.7,9322.1,"very-slow"],["q8","webgpu",11040.4,9365.6,"very-slow"],
  ["q4","wasm",15844.1,4798.4,"very-slow"],["q4","webgpu",19026.3,961.1,"slow"],
  ["q4f16","wasm",11557.5,5203.8,"very-slow"],["q4f16","webgpu",10842.6,947.2,"slow"],
  ["bnb4","wasm",18473.8,4866.2,"very-slow"],["bnb4","webgpu",15142.8,970.2,"slow"],
].map(([precision,device,loadMs,inferenceMs,grade])=>({precision,device,status:"pass",runtime:{loadMs,inferenceMs,grade},quality:{grade:"scene-segmentation-pass"},cleanup:{modelDisposed:true,cacheClearedBeforeCase:true,cacheClearedAfterCase:true}}));
const LABELS = ["wall","building","sky","floor","tree","ceiling","road","bed","windowpane","grass","cabinet","sidewalk","person","earth","door","table","mountain","plant","curtain","chair","car","water","painting","sofa","shelf","house","sea","mirror","rug","field","armchair","seat","fence","desk","rock","wardrobe","lamp","bathtub","railing","cushion","base","box","column","signboard","chest of drawers","counter","sand","sink","skyscraper","fireplace","refrigerator","grandstand","path","stairs","runway","case","pool table","pillow","screen door","stairway","river","bridge","bookcase","blind","coffee table","toilet","flower","book","hill","bench","countertop","stove","palm","kitchen island","computer","swivel chair","boat","bar","arcade machine","hovel","bus","towel","light","truck","tower","chandelier","awning","streetlight","booth","television receiver","airplane","dirt track","apparel","pole","land","bannister","escalator","ottoman","bottle","buffet","poster","stage","van","ship","fountain","conveyer belt","canopy","washer","plaything","swimming pool","stool","barrel","basket","waterfall","tent","bag","minibike","cradle","oven","ball","food","step","tank","trade name","microwave","pot","animal","bicycle","lake","dishwasher","screen","blanket","sculpture","hood","sconce","vase","traffic light","tray","ashcan","fan","pier","crt screen","plate","monitor","bulletin board","shower","radiator","glass","clock","flag"];
const MANIFEST = {
  contractVersion: "1.0",
  model: {
    id: ID,
    displayName: "SegFormer B3 ADE20K — 150-Class Scene Segmentation",
    provider: "NVIDIA Research",
    providerUrl: "https://github.com/NVlabs/SegFormer",
    license: "LicenseRef-NVIDIA-SegFormer-NonCommercial",
    licenseUrl: "https://github.com/NVlabs/SegFormer/blob/master/LICENSE",
    lastUpdated: "2022-08-06T10:29:16.000Z",
    sourceRepository: "nvidia/segformer-b3-finetuned-ade-512-512",
    sourceRevision: "a820c29fc1e53723079d94ca0e09a14d2657fae6",
    repository: REPO,
    artifactRevision: REVISION,
    artifactLastUpdated: "2026-08-02T11:36:45.000Z",
    task: "image-segmentation",
    description: "NVIDIA's B3-sized SegFormer fine-tuned at 512×512 on ADE20K and converted by ONNX Community for Transformers.js. It returns original-size binary masks for up to 150 indoor and outdoor scene classes.",
    intendedUses: ["Non-commercial research and evaluation of semantic scene segmentation.", "Educational visualization of ADE20K scene classes.", "Prototype browser tools that need coarse per-pixel scene labels."],
    limitations: ["The NVIDIA SegFormer license restricts use to non-commercial research or evaluation and requires preserving its license and notices.", "The fixed ADE20K taxonomy is incomplete and merges visually or functionally distinct objects.", "Predictions can fail on small objects, fine boundaries, unusual viewpoints, low light, occlusion, or domains unlike ADE20K.", "Masks are resized to the source image and can lose boundary detail after the model's 512×512 preprocessing.", "Output must not be used for surveillance, safety-critical navigation, medical analysis, or other high-impact automated decisions."],
  },
  runtime: { engine: { name: "Transformers.js", version: "4.2.0" }, precisions: SUPPORTED },
  input: { description: "One browser-readable image URL for indoor or outdoor scene segmentation.", alternatives: [{ name: "image", fields: [{ name: "image", type: "string(URL)", required: true, description: "An HTTP(S), blob, data, or same-origin image URL readable by the worker." }] }] },
  config: { model: [], generation: [] },
  output: {
    description: "Original image dimensions and one binary byte mask per predicted ADE20K class. WebAI wraps this native value in { result, runtime: { durationMs } }.",
    fields: [{ name: "segments", type: "Array<{label:string,score:number|null,mask:{data:Uint8Array,width:integer,height:integer}}>", required: true }, { name: "width", type: "integer", required: true }, { name: "height", type: "integer", required: true }],
    example: { segments: [{ label: "building", score: null, mask: { data: "Uint8Array", width: 640, height: 480 } }], width: 640, height: 480 },
  },
  operations: ["checkModelSupports", "init", "download", "generate", "clearMemory"],
  benchmark: {
    report: "benchmarks/segformer-b3-ade20k.json",
    testedAt: "2026-09-15T14:19:59.145Z",
    fixture: { file: "modnet-portrait-pexels-5965592.jpg", url: "https://f005.backblazeb2.com/file/hubters-webai-public/assets/benchmarks/modnet-portrait-pexels-5965592.jpg", size: 78124, sha1: "38212739a4a4f0be25d84863cf991d7fb6c6d818", width: 1024, height: 683, source: "https://www.pexels.com/photo/woman-wearing-pink-turtleneck-sweater-5965592/", license: "Pexels License", licenseUrl: "https://www.pexels.com/license/", preprocessing: "RGB decode through the pinned artifact's SegFormer image processor." },
    candidates: PRECISIONS,
    qualityGate: "Finite original-size binary masks with unique known labels; checked-in expected portrait scene labels and area ranges must match the pinned FP32 reference signature.",
    environment: "Isolated headless Chrome on macOS; Transformers.js 4.2.0",
    invalidatedBy: ["artifact revision change", "runtime version change", "fixture or quality-gate change", "browser/backend behavior change"],
    results: BENCHMARK_RESULTS,
  },
};

const state = { initializing: false, initialized: false, generating: false };
let model = null;
const send = (requestId, type, data) => self.postMessage({ requestId, type, data });
const capabilities = { supportedModes: ["webai"], supportedPrecisions: Object.keys(SUPPORTED), supportedPrecisionsDevicesMap: SUPPORTED, doesSupportStreamGeneration: false, externalInterrupt: true, workerVersion: "v4", transformersJsVersion: "4.2.0", cacheModelId: REPO };
function configure(config = {}) { env.allowRemoteModels = config.allowRemoteModels ?? true; env.allowLocalModels = config.allowLocalModels ?? false; env.useBrowserCache = config.useBrowserCache ?? true; if (config.remoteHost) env.remoteHost = config.remoteHost; if (config.remotePathTemplate) env.remotePathTemplate = config.remotePathTemplate; if (config.localModelPath) env.localModelPath = config.localModelPath; }
function validateInit(data) { const precision = data?.precision, device = data?.device, support = SUPPORTED[precision]; if (!support) throw Error(`Unsupported precision: ${precision}`); const devices = support.supportedDevices ?? support.candidateDevices; if (!devices.includes(device)) throw Error(`Precision ${precision} is only supported on: ${devices.join(", ")}`); if (device === "webgpu" && !self.navigator?.gpu) throw Error("WebGPU is unavailable"); return { precision, device }; }
function validateConfigs(data) { for (const key of ["modelConfig", "generateConfig"]) if (data?.[key] !== undefined && (!data[key] || typeof data[key] !== "object" || Array.isArray(data[key]) || Object.keys(data[key]).length)) throw Error(`${key} does not accept options for this model`); }
async function clear() { if (model?.dispose) await model.dispose(); model = null; state.initialized = false; }
async function load(data, requestId) { const { precision, device } = validateInit(data); await clear(); model = await pipeline("image-segmentation", REPO, { revision: REVISION, dtype: precision, device, progress_callback: value => send(requestId, "downloadProgress", value) }); }
async function generate(data) {
  validateConfigs(data);
  const image = data?.userInput?.image;
  if (typeof image !== "string" || !image.trim()) throw Error("userInput.image must be a non-empty URL");
  const raw = await RawImage.read(image), output = await model(raw);
  if (!Array.isArray(output) || !output.length) throw Error("Model returned no segmentation masks");
  const segments = output.map(({ label, score, mask }) => {
    if (!LABELS.includes(label)) throw Error(`Model returned unknown label: ${label}`);
    if (!mask?.data || !Number.isInteger(mask.width) || !Number.isInteger(mask.height)) throw Error(`Model returned an invalid mask for ${label}`);
    const values = Uint8Array.from(mask.data, value => Number(value));
    if (values.length !== mask.width * mask.height || values.some(value => value !== 0 && value !== 255)) throw Error(`Model returned a non-binary mask for ${label}`);
    return { label, score: score == null ? null : Number(score), mask: { data: values, width: mask.width, height: mask.height } };
  });
  return { segments, width: raw.width, height: raw.height };
}
self.addEventListener("message", async ({ data: message = {} }) => {
  const { requestId, type, data } = message;
  try {
    if (type === "checkModelSupports") { configure(data?.workerConfig); send(requestId, type, { ...capabilities, manifest: MANIFEST }); return; }
    if (type === "init" || type === "download") {
      if (state.initializing || state.generating) throw Error("Worker is busy"); state.initializing = true; state.initialized = false;
      try { const precision = data?.precision, device = type === "download" ? SUPPORTED[precision]?.supportedDevices[0] : data?.device; await load({ ...data, precision, device }, requestId); if (type === "download") await clear(); else state.initialized = true; send(requestId, type, { status: "success" }); }
      finally { state.initializing = false; }
      return;
    }
    if (type === "generate") { if (state.initializing || !state.initialized || !model) throw Error("Call init before generate"); if (state.generating) throw Error("Generation already in progress"); state.generating = true; try { send(requestId, "generated", { status: "success", result: await generate(data) }); } finally { state.generating = false; } return; }
    if (type === "generateStream") throw Error("Stream generation is not supported");
    if (type === "clearMemory") { if (state.initializing || state.generating) throw Error("Worker is busy"); state.initializing = true; try { await clear(); send(requestId, type, { status: "success" }); } finally { state.initializing = false; } return; }
    throw Error(`Unknown operation: ${type}`);
  } catch (error) { send(requestId, "error", { message: error instanceof Error ? error.message : String(error), context: type }); }
});
self.postMessage({ type: "worker initialized", data: { success: true, workerVersion: "v4", modelId: ID } });
