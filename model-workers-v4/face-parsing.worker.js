import { env, pipeline, RawImage } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0";

const ID = "face-parsing";
const REPO = "Xenova/face-parsing";
const REVISION = "f25b9b521a8783d4e78e80e026ef4c2a15f821e0";
const SHARED_BYTES = 2812;
const make = (weightsSize, dtype, suffix) => ({ weightsSize, size: weightsSize + SHARED_BYTES, dtype, modelKeys: [`model${suffix}.onnx`], candidateDevices: ["wasm", "webgpu"] });
const PRECISIONS = {
  fp32: make(340316611, "fp32", ""),
  fp16: make(171716570, "fp16", "_fp16"),
  q8: make(89439678, "q8", "_quantized"),
};
// Filled only from benchmark-passing precision/device pairs.
const SUPPORTED = {"fp32":{"weightsSize":340316611,"size":340319423,"dtype":"fp32","modelKeys":["model.onnx"],"candidateDevices":["wasm","webgpu"],"supportedDevices":["wasm","webgpu"]},"fp16":{"weightsSize":171716570,"size":171719382,"dtype":"fp16","modelKeys":["model_fp16.onnx"],"candidateDevices":["wasm","webgpu"],"supportedDevices":["webgpu"]},"q8":{"weightsSize":89439678,"size":89442490,"dtype":"q8","modelKeys":["model_quantized.onnx"],"candidateDevices":["wasm","webgpu"],"supportedDevices":["wasm","webgpu"]}};
const LABELS = ["background", "skin", "nose", "eye_g", "l_eye", "r_eye", "l_brow", "r_brow", "l_ear", "r_ear", "mouth", "u_lip", "l_lip", "hair", "hat", "ear_r", "neck_l", "neck", "cloth"];
const MANIFEST = {
  contractVersion: "1.0",
  model: {
    id: ID,
    displayName: "Face Parsing — 19-Class Semantic Segmentation",
    provider: "Jonathan Dinu",
    providerUrl: "https://huggingface.co/jonathandinu",
    license: "LicenseRef-FaceParsing-NC-Research-Education",
    licenseUrl: "https://huggingface.co/jonathandinu/face-parsing/blob/758b82e15a0178c9db39c1ff666a8b56e3a550c8/README.md#model-description",
    lastUpdated: "2026-02-18T22:38:54.000Z",
    sourceRepository: "jonathandinu/face-parsing",
    sourceRevision: "758b82e15a0178c9db39c1ff666a8b56e3a550c8",
    repository: REPO,
    artifactRevision: REVISION,
    artifactLastUpdated: "2025-08-18T16:00:54.000Z",
    task: "image-segmentation",
    description: "A SegFormer face-parsing model fine-tuned on CelebAMask-HQ and converted by Xenova for browser inference. It returns separate pixel masks for 19 facial, hair, accessory, clothing, neck, and background classes.",
    intendedUses: ["Non-commercial research into face-part segmentation.", "Educational image-editing prototypes that need facial-region masks."],
    limitations: ["The source permits non-commercial research and educational use only; confirm the upstream terms before redistribution or commercial use.", "The model is trained for portrait-like faces and can fail on profiles, occlusion, unusual lighting, small faces, multiple people, illustrations, and out-of-domain images.", "Left/right labels follow model training conventions and may not match the viewer's perspective.", "Small regions such as earrings, lips, and eyebrows may be missing or fragmented.", "Face-region output must not be used for identity, biometric inference, medical conclusions, surveillance, or other high-impact decisions."],
  },
  runtime: { engine: { name: "Transformers.js", version: "4.2.0" }, precisions: SUPPORTED },
  input: { description: "One browser-readable portrait image URL.", alternatives: [{ name: "image", fields: [{ name: "image", type: "string(URL)", required: true, description: "An HTTP(S), blob, data, or same-origin image URL readable by the worker." }] }] },
  config: { model: [], generation: [] },
  output: {
    description: "Original image dimensions and one binary byte mask per predicted face-parsing class. WebAI wraps this native value in { result, runtime: { durationMs } }.",
    fields: [{ name: "segments", type: "Array<{label:string,score:number|null,mask:{data:Uint8Array,width:integer,height:integer}}>", required: true }, { name: "width", type: "integer", required: true }, { name: "height", type: "integer", required: true }],
    example: { segments: [{ label: "skin", score: null, mask: { data: "Uint8Array", width: 640, height: 480 } }], width: 640, height: 480 },
  },
  operations: ["checkModelSupports", "init", "download", "generate", "clearMemory"],
  benchmark: {
    report: "benchmarks/face-parsing.json",
    testedAt: "2026-09-15T11:46:34.148Z",
    fixture: { file: "modnet-portrait-pexels-5965592.jpg", url: "https://f005.backblazeb2.com/file/hubters-webai-public/assets/benchmarks/modnet-portrait-pexels-5965592.jpg", size: 78124, sha1: "38212739a4a4f0be25d84863cf991d7fb6c6d818", width: 1024, height: 683, source: "https://www.pexels.com/photo/woman-wearing-pink-turtleneck-sweater-5965592/", license: "Pexels License", licenseUrl: "https://www.pexels.com/license/", preprocessing: "RGB decode through the pinned artifact's SegFormer image processor." },
    candidates: PRECISIONS,
    qualityGate: "Finite original-size binary masks; unique known labels; the portrait's checked-in expected background, skin, eyeglasses, hair, hat, neck, and clothing regions present; skin and hair each cover 1% to 60% of pixels; non-background union covers 10% to 90%.",
    environment: "Isolated headless Chrome on macOS; Transformers.js 4.2.0",
    invalidatedBy: ["artifact revision change", "runtime version change", "fixture or quality-gate change", "browser/backend behavior change"],
    results: [{"precision":"fp32","device":"wasm","status":"pass","runtime":{"loadMs":14697,"inferenceMs":6142,"totalMs":21042,"grade":"very-slow","megapixelsPerSecond":0.11387040052100293},"quality":{"dimensions":true,"finite":true,"uniqueLabels":true,"requiredLabelsPresent":true,"areas":{"background":0.7762499428074671,"skin":0.05067115437408492,"eye_g":0.018614739659590044,"hair":0.07427165309297218,"hat":0.028283137353587114,"neck":0.00002287701317715959,"cloth":0.05188649569912152},"nonBackgroundRatio":0.22375005719253294,"grade":"face-parsing-pass"},"cleanup":{"modelDisposed":true,"cacheClearedBeforeCase":true,"cacheClearedAfterCase":true,"deletedEntries":3}},{"precision":"fp32","device":"webgpu","status":"pass","runtime":{"loadMs":15167.5,"inferenceMs":820.2000000178814,"totalMs":16193.800000011921,"grade":"slow","megapixelsPerSecond":0.8527090953240093},"quality":{"dimensions":true,"finite":true,"uniqueLabels":true,"requiredLabelsPresent":true,"areas":{"background":0.7762499428074671,"skin":0.05067115437408492,"eye_g":0.018614739659590044,"hair":0.07427165309297218,"hat":0.028283137353587114,"neck":0.00002287701317715959,"cloth":0.05188649569912152},"nonBackgroundRatio":0.22375005719253294,"grade":"face-parsing-pass"},"cleanup":{"modelDisposed":true,"cacheClearedBeforeCase":true,"cacheClearedAfterCase":true,"deletedEntries":3}},{"precision":"fp16","device":"wasm","status":"fail","runtime":{"loadMs":null,"inferenceMs":null,"totalMs":14044.800000011921,"grade":"failed"},"error":"Error: Can't create a session. ERROR_CODE: 1, ERROR_MESSAGE: /mnt/vss/_work/1/s/onnxruntime/core/graph/graph_utils.cc:30 int onnxruntime::graph_utils::GetIndexFromName(const Node &, const std::string &, bool) itr != node_args.end() was false. Attempting to get index by a name which does not exist:InsertedPrecisionFreeCast_/segformer/encoder/block.3.1/layer_norm_2/Constant_output_0for node: /segformer/encoder/patch_embeddings.0/layer_norm/Mul/SimplifiedLayerNormFusion/\n","cleanup":{"modelDisposed":true,"cacheClearedBeforeCase":true,"cacheClearedAfterCase":true,"deletedEntries":4}},{"precision":"fp16","device":"webgpu","status":"pass","runtime":{"loadMs":9184,"inferenceMs":805,"totalMs":10197,"grade":"slow","megapixelsPerSecond":0.8688099378881987},"quality":{"dimensions":true,"finite":true,"uniqueLabels":true,"requiredLabelsPresent":true,"areas":{"background":0.7752605119875549,"skin":0.07139057924597364,"eye_g":0.014512605234260616,"hair":0.07439318722547585,"hat":0.014308141928989751,"neck":0.0000014298133235724744,"cloth":0.05013354456442167},"nonBackgroundRatio":0.22473948801244514,"grade":"face-parsing-pass"},"cleanup":{"modelDisposed":true,"cacheClearedBeforeCase":true,"cacheClearedAfterCase":true,"deletedEntries":4}},{"precision":"q8","device":"wasm","status":"pass","runtime":{"loadMs":6912.300000011921,"inferenceMs":6382.699999988079,"totalMs":13502.59999999404,"grade":"very-slow","megapixelsPerSecond":0.10957619816085767},"quality":{"dimensions":true,"finite":true,"uniqueLabels":true,"requiredLabelsPresent":true,"areas":{"background":0.775972559022694,"skin":0.05380816480600293,"eye_g":0.020390567807467057,"hair":0.0740157165080527,"hat":0.026408652086383603,"neck":0.00014870058565153734,"cloth":0.04925563918374817},"nonBackgroundRatio":0.22402744097730598,"grade":"face-parsing-pass"},"cleanup":{"modelDisposed":true,"cacheClearedBeforeCase":true,"cacheClearedAfterCase":true,"deletedEntries":4}},{"precision":"q8","device":"webgpu","status":"pass","runtime":{"loadMs":7769.399999976158,"inferenceMs":6227.0999999940395,"totalMs":14230.5,"grade":"very-slow","megapixelsPerSecond":0.11231423937317041},"quality":{"dimensions":true,"finite":true,"uniqueLabels":true,"requiredLabelsPresent":true,"areas":{"background":0.7762756794472914,"skin":0.05471609626647145,"eye_g":0.018518942166910687,"hair":0.07415726802708639,"hat":0.026508739019033674,"neck":0.0002516471449487555,"cloth":0.049571627928257686},"nonBackgroundRatio":0.22372432055270863,"grade":"face-parsing-pass"},"cleanup":{"modelDisposed":true,"cacheClearedBeforeCase":true,"cacheClearedAfterCase":true,"deletedEntries":4}}],
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
