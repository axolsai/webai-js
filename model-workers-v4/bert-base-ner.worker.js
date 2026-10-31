import {
  pipeline,
  env,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0";

const ID = "bert-base-ner";
const REPO = "Xenova/bert-base-NER";
const REV = "8e892123e8b7c2c0c2bd1dcb598b7d244c4e53aa";
const SHARED_BYTES = 883882;
const make = (weightsSize, dtype, key) => ({
  weightsSize,
  size: weightsSize + SHARED_BYTES,
  dtype,
  modelKeys: [key],
  candidateDevices: ["wasm", "webgpu"],
});
const PRECISIONS = {
  fp32: make(431172606, "fp32", "model.onnx"),
  fp16: make(215805679, "fp16", "model_fp16.onnx"),
  int8: make(108486236, "int8", "model_int8.onnx"),
  uint8: make(108486273, "uint8", "model_uint8.onnx"),
  q8: make(108952255, "q8", "model_quantized.onnx"),
  q4: make(144540477, "q4", "model_q4.onnx"),
  q4f16: make(93661735, "q4f16", "model_q4f16.onnx"),
  bnb4: make(139232164, "bnb4", "model_bnb4.onnx"),
};
const SUPPORTED = {
  q8: { ...PRECISIONS.q8, supportedDevices: ["wasm", "webgpu"] },
  q4: { ...PRECISIONS.q4, supportedDevices: ["wasm", "webgpu"] },
};

const MANIFEST = {
  contractVersion: "1.0",
  model: {
    id: ID,
    displayName: "BERT Base NER — English Named Entities",
    provider: "dslim",
    providerUrl: "https://huggingface.co/dslim",
    license: "MIT",
    licenseUrl: "https://spdx.org/licenses/MIT.html",
    lastUpdated: "2024-10-08T07:51:39.000Z",
    sourceRepository: "dslim/bert-base-NER",
    sourceRevision: "d1a3e8f13f8c3566299d95fcfc9a8d2382a9affc",
    repository: REPO,
    artifactRevision: REV,
    artifactLastUpdated: "2026-09-14T15:58:49.000Z",
    task: "token-classification",
    description:
      "A cased English BERT model fine-tuned on CoNLL-2003 news text and converted by Xenova for local browser inference. It extracts person, organization, location, and miscellaneous entity spans with confidence scores and character offsets.",
    intendedUses: [
      "Highlight named entities in English news-like text.",
      "Prototype local document indexing, search annotation, and information-extraction workflows.",
      "Provide candidate spans for human-reviewed downstream processing.",
    ],
    limitations: [
      "The model recognizes only PER, ORG, LOC, and MISC; dates, quantities, products, events, and other entity types are not represented consistently.",
      "It was trained on English Reuters news from a limited historical period and may generalize poorly to other languages, domains, emerging names, informal spelling, or unusual capitalization.",
      "Subword tokenization can split, truncate, merge, or omit entity spans; character offsets and labels require application-level review.",
      "Confidence scores are not calibrated guarantees and must not be used alone for consequential decisions about people.",
      "Inputs longer than 512 tokens are truncated, so entities beyond the retained prefix are omitted.",
    ],
  },
  runtime: {
    engine: { name: "Transformers.js", version: "4.2.0" },
    precisions: SUPPORTED,
  },
  input: {
    description:
      "A non-empty English text string. The model preserves JavaScript character offsets for returned spans.",
    alternatives: [
      {
        name: "text",
        fields: [
          { name: "text", type: "string", required: true, minLength: 1 },
        ],
      },
    ],
  },
  config: {
    model: [
      {
        name: "max_length",
        type: "integer",
        default: 512,
        minimum: 8,
        maximum: 512,
        description:
          "Maximum token count including special tokens; longer text is truncated.",
      },
    ],
    generation: [
      {
        name: "aggregation_strategy",
        type: "string",
        default: "simple",
        enum: ["simple", "first", "average", "max", "none"],
        description:
          "How adjacent token labels are merged into entity spans. none returns token-level entities.",
      },
    ],
  },
  output: {
    description:
      "Detected entity spans in source order with normalized labels, confidence scores, and zero-based character offsets.",
    fields: [
      {
        name: "entities",
        type: "Array<{text:string,label:'PER'|'ORG'|'LOC'|'MISC',score:number,start:number,end:number}>",
      },
    ],
    example: {
      entities: [
        { text: "Berlin", label: "LOC", score: 0.999, start: 31, end: 37 },
      ],
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
    report: "benchmarks/bert-base-ner.json",
    testedAt: "2026-10-23T10:53:16.920Z",
    fixture: {
      file: "bert-base-ner.json",
      url: "https://f005.backblazeb2.com/file/hubters-webai-public/assets/benchmarks/bert-base-ner.json",
      size: 676,
      sha1: "a90640ef195e01ddb8f45899678975feaee59853",
      source:
        "NASA Apollo 11 Mission Overview; lightly shortened without changing named entities.",
      license: "United States government work",
      licenseUrl: "https://www.nasa.gov/nasa-brand-center/images-and-media/",
      preprocessing:
        "Pinned cased BERT tokenizer with truncation at max_length; simple BIO aggregation.",
    },
    candidates: PRECISIONS,
    qualityGate:
      "Finite in-range scores and valid ordered offsets; every checked NASA location/person span must be recovered with the expected label, exact normalized text, and score >= 0.8; output must be non-empty and non-degenerate.",
    environment:
      "Isolated headless Google Chrome on macOS; Transformers.js 4.2.0",
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
        status: "fail",
        runtime: {
          loadMs: 51071.199999928474,
          inferenceMs: 184.89999997615814,
          totalMs: 52634.700000047684,
          grade: "slow",
          textsPerSecond: 5.40832882709002,
        },
        cleanup: {
          modelDisposed: true,
          cacheClearedBeforeCase: true,
          cacheClearedAfterCase: true,
          deletedEntries: 4,
        },
        quality: {
          finite: true,
          validOffsets: true,
          nondegenerate: true,
          recall: 0.75,
          grade: "failed",
          expected: [
            { text: "Cape Kennedy", label: "LOC", pass: true },
            { text: "Neil Armstrong", label: "PER", pass: true },
            { text: "Michael Collins", label: "PER", pass: true },
            { text: "Edwin Buzz Aldrin", label: "PER", pass: false },
          ],
        },
      },
      {
        precision: "fp32",
        device: "webgpu",
        status: "fail",
        runtime: {
          loadMs: 56146.5,
          inferenceMs: 79.39999997615814,
          totalMs: 57656.60000002384,
          grade: "good",
          textsPerSecond: 12.594458442068959,
        },
        cleanup: {
          modelDisposed: true,
          cacheClearedBeforeCase: true,
          cacheClearedAfterCase: true,
          deletedEntries: 4,
        },
        quality: {
          finite: true,
          validOffsets: true,
          nondegenerate: true,
          recall: 0.75,
          grade: "failed",
          expected: [
            { text: "Cape Kennedy", label: "LOC", pass: true },
            { text: "Neil Armstrong", label: "PER", pass: true },
            { text: "Michael Collins", label: "PER", pass: true },
            { text: "Edwin Buzz Aldrin", label: "PER", pass: false },
          ],
        },
      },
      {
        precision: "fp16",
        device: "wasm",
        status: "fail",
        runtime: {
          loadMs: null,
          inferenceMs: null,
          totalMs: 29800.399999976158,
          grade: "failed",
        },
        cleanup: {
          modelDisposed: true,
          cacheClearedBeforeCase: true,
          cacheClearedAfterCase: true,
          deletedEntries: 5,
        },
        error:
          "Error: Can't create a session. ERROR_CODE: 1, ERROR_MESSAGE: /mnt/vss/_work/1/s/onnxruntime/core/graph/graph_utils.cc:30 int onnxruntime::graph_utils::GetIndexFromName(const Node &, const std::string &, bool) itr != node_args.end() was false. Attempting to get index by a name which does not exist:InsertedPrecisionFreeCast_/bert/encoder/layer.11/output/LayerNorm/Constant_output_0for node: /bert/embeddings/LayerNorm/Mul/SimplifiedLayerNormFusion/\n",
      },
      {
        precision: "fp16",
        device: "webgpu",
        status: "fail",
        runtime: {
          loadMs: 27688,
          inferenceMs: 68.30000007152557,
          totalMs: 29063.399999976158,
          grade: "good",
          textsPerSecond: 14.641288418049392,
        },
        cleanup: {
          modelDisposed: true,
          cacheClearedBeforeCase: true,
          cacheClearedAfterCase: true,
          deletedEntries: 5,
        },
        quality: {
          finite: true,
          validOffsets: true,
          nondegenerate: true,
          recall: 0.75,
          grade: "failed",
          expected: [
            { text: "Cape Kennedy", label: "LOC", pass: true },
            { text: "Neil Armstrong", label: "PER", pass: true },
            { text: "Michael Collins", label: "PER", pass: true },
            { text: "Edwin Buzz Aldrin", label: "PER", pass: false },
          ],
        },
      },
      {
        precision: "int8",
        device: "wasm",
        status: "fail",
        runtime: {
          loadMs: 19456.799999952316,
          inferenceMs: 194.29999995231628,
          totalMs: 21021.399999976158,
          grade: "slow",
          textsPerSecond: 5.146680392410771,
        },
        cleanup: {
          modelDisposed: true,
          cacheClearedBeforeCase: true,
          cacheClearedAfterCase: true,
          deletedEntries: 5,
        },
        quality: {
          finite: true,
          validOffsets: true,
          nondegenerate: true,
          recall: 0.75,
          grade: "failed",
          expected: [
            { text: "Cape Kennedy", label: "LOC", pass: true },
            { text: "Neil Armstrong", label: "PER", pass: true },
            { text: "Michael Collins", label: "PER", pass: true },
            { text: "Edwin Buzz Aldrin", label: "PER", pass: false },
          ],
        },
      },
      {
        precision: "int8",
        device: "webgpu",
        status: "fail",
        runtime: {
          loadMs: 18314.100000023842,
          inferenceMs: 362.09999990463257,
          totalMs: 19945.100000023842,
          grade: "slow",
          textsPerSecond: 2.7616680482280396,
        },
        cleanup: {
          modelDisposed: true,
          cacheClearedBeforeCase: true,
          cacheClearedAfterCase: true,
          deletedEntries: 5,
        },
        quality: {
          finite: true,
          validOffsets: true,
          nondegenerate: true,
          recall: 0.75,
          grade: "failed",
          expected: [
            { text: "Cape Kennedy", label: "LOC", pass: true },
            { text: "Neil Armstrong", label: "PER", pass: true },
            { text: "Michael Collins", label: "PER", pass: true },
            { text: "Edwin Buzz Aldrin", label: "PER", pass: false },
          ],
        },
      },
      {
        precision: "uint8",
        device: "wasm",
        status: "fail",
        runtime: {
          loadMs: 20562.299999952316,
          inferenceMs: 197.29999995231628,
          totalMs: 22139.100000023842,
          grade: "slow",
          textsPerSecond: 5.068423721447954,
        },
        cleanup: {
          modelDisposed: true,
          cacheClearedBeforeCase: true,
          cacheClearedAfterCase: true,
          deletedEntries: 5,
        },
        quality: {
          finite: true,
          validOffsets: true,
          nondegenerate: true,
          recall: 0.75,
          grade: "failed",
          expected: [
            { text: "Cape Kennedy", label: "LOC", pass: true },
            { text: "Neil Armstrong", label: "PER", pass: true },
            { text: "Michael Collins", label: "PER", pass: true },
            { text: "Edwin Buzz Aldrin", label: "PER", pass: false },
          ],
        },
      },
      {
        precision: "uint8",
        device: "webgpu",
        status: "fail",
        runtime: {
          loadMs: 18230.300000071526,
          inferenceMs: 285.7999999523163,
          totalMs: 19954.700000047684,
          grade: "slow",
          textsPerSecond: 3.4989503154893034,
        },
        cleanup: {
          modelDisposed: true,
          cacheClearedBeforeCase: true,
          cacheClearedAfterCase: true,
          deletedEntries: 5,
        },
        quality: {
          finite: true,
          validOffsets: true,
          nondegenerate: true,
          recall: 0.75,
          grade: "failed",
          expected: [
            { text: "Cape Kennedy", label: "LOC", pass: true },
            { text: "Neil Armstrong", label: "PER", pass: true },
            { text: "Michael Collins", label: "PER", pass: true },
            { text: "Edwin Buzz Aldrin", label: "PER", pass: false },
          ],
        },
      },
      {
        precision: "q8",
        device: "wasm",
        status: "pass",
        runtime: {
          loadMs: 19248.100000023842,
          inferenceMs: 198.39999997615814,
          totalMs: 20881.299999952316,
          grade: "slow",
          textsPerSecond: 5.0403225812508605,
        },
        cleanup: {
          modelDisposed: true,
          cacheClearedBeforeCase: true,
          cacheClearedAfterCase: true,
          deletedEntries: 5,
        },
        quality: {
          finite: true,
          validOffsets: true,
          nondegenerate: true,
          recall: 1,
          grade: "ner-span-pass",
          expected: [
            { text: "Cape Kennedy", label: "LOC", pass: true },
            { text: "Neil Armstrong", label: "PER", pass: true },
            { text: "Michael Collins", label: "PER", pass: true },
            { text: "Edwin Buzz Aldrin", label: "PER", pass: true },
          ],
        },
      },
      {
        precision: "q8",
        device: "webgpu",
        status: "pass",
        runtime: {
          loadMs: 16627.100000023842,
          inferenceMs: 363,
          totalMs: 18244.899999976158,
          grade: "slow",
          textsPerSecond: 2.7548209366391183,
        },
        cleanup: {
          modelDisposed: true,
          cacheClearedBeforeCase: true,
          cacheClearedAfterCase: true,
          deletedEntries: 5,
        },
        quality: {
          finite: true,
          validOffsets: true,
          nondegenerate: true,
          recall: 1,
          grade: "ner-span-pass",
          expected: [
            { text: "Cape Kennedy", label: "LOC", pass: true },
            { text: "Neil Armstrong", label: "PER", pass: true },
            { text: "Michael Collins", label: "PER", pass: true },
            { text: "Edwin Buzz Aldrin", label: "PER", pass: true },
          ],
        },
      },
      {
        precision: "q4",
        device: "wasm",
        status: "pass",
        runtime: {
          loadMs: 20310.100000023842,
          inferenceMs: 302.10000002384186,
          totalMs: 21789.799999952316,
          grade: "slow",
          textsPerSecond: 3.31016219768646,
        },
        cleanup: {
          modelDisposed: true,
          cacheClearedBeforeCase: true,
          cacheClearedAfterCase: true,
          deletedEntries: 5,
        },
        quality: {
          finite: true,
          validOffsets: true,
          nondegenerate: true,
          recall: 1,
          grade: "ner-span-pass",
          expected: [
            { text: "Cape Kennedy", label: "LOC", pass: true },
            { text: "Neil Armstrong", label: "PER", pass: true },
            { text: "Michael Collins", label: "PER", pass: true },
            { text: "Edwin Buzz Aldrin", label: "PER", pass: true },
          ],
        },
      },
      {
        precision: "q4",
        device: "webgpu",
        status: "pass",
        runtime: {
          loadMs: 20363,
          inferenceMs: 91.39999997615814,
          totalMs: 21691.099999904633,
          grade: "good",
          textsPerSecond: 10.940919040053084,
        },
        cleanup: {
          modelDisposed: true,
          cacheClearedBeforeCase: true,
          cacheClearedAfterCase: true,
          deletedEntries: 5,
        },
        quality: {
          finite: true,
          validOffsets: true,
          nondegenerate: true,
          recall: 1,
          grade: "ner-span-pass",
          expected: [
            { text: "Cape Kennedy", label: "LOC", pass: true },
            { text: "Neil Armstrong", label: "PER", pass: true },
            { text: "Michael Collins", label: "PER", pass: true },
            { text: "Edwin Buzz Aldrin", label: "PER", pass: true },
          ],
        },
      },
      {
        precision: "q4f16",
        device: "wasm",
        status: "fail",
        runtime: {
          loadMs: null,
          inferenceMs: null,
          totalMs: 17129.299999952316,
          grade: "failed",
        },
        cleanup: {
          modelDisposed: true,
          cacheClearedBeforeCase: true,
          cacheClearedAfterCase: true,
          deletedEntries: 5,
        },
        error:
          "Error: Can't create a session. ERROR_CODE: 1, ERROR_MESSAGE: /mnt/vss/_work/1/s/onnxruntime/core/graph/graph_utils.cc:30 int onnxruntime::graph_utils::GetIndexFromName(const Node &, const std::string &, bool) itr != node_args.end() was false. Attempting to get index by a name which does not exist:InsertedPrecisionFreeCast_/bert/embeddings/LayerNorm/Constant_output_0for node: /bert/embeddings/LayerNorm/Mul/SimplifiedLayerNormFusion/\n",
      },
      {
        precision: "q4f16",
        device: "webgpu",
        status: "fail",
        runtime: {
          loadMs: 15915.899999976158,
          inferenceMs: 68.89999997615814,
          totalMs: 17249.600000023842,
          grade: "good",
          textsPerSecond: 14.513788103716047,
        },
        cleanup: {
          modelDisposed: true,
          cacheClearedBeforeCase: true,
          cacheClearedAfterCase: true,
          deletedEntries: 5,
        },
        quality: {
          finite: true,
          validOffsets: true,
          nondegenerate: true,
          recall: 0.75,
          grade: "failed",
          expected: [
            { text: "Cape Kennedy", label: "LOC", pass: true },
            { text: "Neil Armstrong", label: "PER", pass: true },
            { text: "Michael Collins", label: "PER", pass: true },
            { text: "Edwin Buzz Aldrin", label: "PER", pass: false },
          ],
        },
      },
      {
        precision: "bnb4",
        device: "wasm",
        status: "fail",
        runtime: {
          loadMs: 21031.100000023842,
          inferenceMs: 234.29999995231628,
          totalMs: 22572.199999928474,
          grade: "slow",
          textsPerSecond: 4.268032437915133,
        },
        cleanup: {
          modelDisposed: true,
          cacheClearedBeforeCase: true,
          cacheClearedAfterCase: true,
          deletedEntries: 5,
        },
        quality: {
          finite: true,
          validOffsets: true,
          nondegenerate: true,
          recall: 0.75,
          grade: "failed",
          expected: [
            { text: "Cape Kennedy", label: "LOC", pass: true },
            { text: "Neil Armstrong", label: "PER", pass: true },
            { text: "Michael Collins", label: "PER", pass: true },
            { text: "Edwin Buzz Aldrin", label: "PER", pass: false },
          ],
        },
      },
      {
        precision: "bnb4",
        device: "webgpu",
        status: "fail",
        runtime: {
          loadMs: 20800.199999928474,
          inferenceMs: 417.3000000715256,
          totalMs: 22650.799999952316,
          grade: "slow",
          textsPerSecond: 2.396357536133715,
        },
        cleanup: {
          modelDisposed: true,
          cacheClearedBeforeCase: true,
          cacheClearedAfterCase: true,
          deletedEntries: 5,
        },
        quality: {
          finite: true,
          validOffsets: true,
          nondegenerate: true,
          recall: 0.75,
          grade: "failed",
          expected: [
            { text: "Cape Kennedy", label: "LOC", pass: true },
            { text: "Neil Armstrong", label: "PER", pass: true },
            { text: "Michael Collins", label: "PER", pass: true },
            { text: "Edwin Buzz Aldrin", label: "PER", pass: false },
          ],
        },
      },
    ],
  },

  samples: [
  {
    "id": "standard",
    "label": "Try sample text",
    "description": "Standard sample for BERT Base NER — English Named Entities.",
    "userInput": {
      "text": "Apollo 11 launched from Cape Kennedy on July 16, 1969, carrying Commander Neil Armstrong, Command Module Pilot Michael Collins and Lunar Module Pilot Edwin Buzz Aldrin into an initial Earth orbit."
    },
    "assets": [],
    "modelConfig": {},
    "generateConfig": {},
    "source": "NASA Apollo 11 Mission Overview; lightly shortened without changing named entities.",
    "license": "United States government work",
    "fixture": {
      "file": "model-samples/bert-base-ner.json",
      "url": "https://f005.backblazeb2.com/file/hubters-webai-public/assets/benchmarks/bert-base-ner.json",
      "size": 676,
      "sha1": "a90640ef195e01ddb8f45899678975feaee59853"
    },
    "benchmarkReport": "benchmarks/bert-base-ner.json"
  }
],
};

const state = { initializing: false, initialized: false, generating: false };
let classifier = null;
const send = (requestId, type, data) =>
  self.postMessage({ requestId, type, data });
const capabilities = {
  supportedModes: ["webai"],
  supportedPrecisions: Object.keys(SUPPORTED),
  supportedPrecisionsDevicesMap: SUPPORTED,
  doesSupportStreamGeneration: false,
  externalInterrupt: true,
  workerVersion: "v4",
  transformersJsVersion: "4.2.0",
  cacheModelId: REPO,
};

function configure(config = {}) {
  env.allowRemoteModels = config.allowRemoteModels ?? true;
  env.allowLocalModels = config.allowLocalModels ?? false;
  env.useBrowserCache = config.useBrowserCache ?? true;
  if (config.remoteHost) env.remoteHost = config.remoteHost;
  if (config.remotePathTemplate)
    env.remotePathTemplate = config.remotePathTemplate;
  if (config.localModelPath) env.localModelPath = config.localModelPath;
}
function validateInit(data, supported = SUPPORTED) {
  const precision = data?.precision,
    device = data?.device,
    entry = supported[precision];
  if (!entry) throw Error(`Unsupported precision: ${precision}`);
  const devices = entry.supportedDevices ?? entry.candidateDevices;
  if (!devices.includes(device))
    throw Error(
      `Precision ${precision} is only supported on: ${devices.join(", ")}`,
    );
  if (device === "webgpu" && !self.navigator?.gpu)
    throw Error("WebGPU is unavailable");
  return { precision, device };
}
function options(data) {
  const modelConfig = data?.modelConfig,
    generateConfig = data?.generateConfig;
  for (const [name, value] of [
    ["modelConfig", modelConfig],
    ["generateConfig", generateConfig],
  ])
    if (
      value !== undefined &&
      (!value || typeof value !== "object" || Array.isArray(value))
    )
      throw Error(`${name} must be an object`);
  for (const key of Object.keys(modelConfig ?? {}))
    if (key !== "max_length")
      throw Error(`Unsupported modelConfig option: ${key}`);
  for (const key of Object.keys(generateConfig ?? {}))
    if (key !== "aggregation_strategy")
      throw Error(`Unsupported generateConfig option: ${key}`);
  const max_length = modelConfig?.max_length ?? 512,
    aggregation_strategy = generateConfig?.aggregation_strategy ?? "simple";
  if (!Number.isInteger(max_length) || max_length < 8 || max_length > 512)
    throw Error("max_length must be an integer from 8 through 512");
  if (
    !["simple", "first", "average", "max", "none"].includes(
      aggregation_strategy,
    )
  )
    throw Error(
      "aggregation_strategy must be simple, first, average, max, or none",
    );
  return { max_length, aggregation_strategy };
}
async function clear() {
  if (classifier?.dispose) await classifier.dispose();
  classifier = null;
  state.initialized = false;
}
async function load(data, requestId, supported = SUPPORTED) {
  const { precision, device } = validateInit(data, supported);
  await clear();
  classifier = await pipeline("token-classification", REPO, {
    revision: REV,
    dtype: precision,
    device,
    progress_callback: (value) => send(requestId, "downloadProgress", value),
  });
}
async function generate(data) {
  const text = data?.userInput?.text;
  if (typeof text !== "string" || !text.trim())
    throw Error("userInput.text must be a non-empty string");
  const { max_length, aggregation_strategy } = options(data);
  const raw = await classifier(text, {
    aggregation_strategy,
    truncation: true,
    max_length,
  });
  let searchFrom = 0;
  const normalized = (value) =>
    value.toLocaleLowerCase("en").replace(/[^a-z0-9]/g, "");
  function align(word) {
    const needle = normalized(word.replace(/##/g, ""));
    for (let start = searchFrom; start < text.length; start++) {
      if (normalized(text[start]) !== needle[0]) continue;
      for (let end = start + 1; end <= text.length; end++) {
        const candidate = normalized(text.slice(start, end));
        if (candidate === needle) return { start, end };
        if (!needle.startsWith(candidate)) break;
      }
    }
    return { start: -1, end: -1 };
  }
  const entities = raw
    .filter((item) => (item.entity_group ?? item.entity) !== "O")
    .map((item) => {
      const label = String(item.entity_group ?? item.entity).replace(
        /^[BI]-/,
        "",
      );
      const word = String(item.word ?? ""),
        score = Number(item.score);
      const reportedStart = Number(item.start),
        reportedEnd = Number(item.end);
      const aligned =
        Number.isInteger(reportedStart) && Number.isInteger(reportedEnd)
          ? { start: reportedStart, end: reportedEnd }
          : align(word);
      const { start, end } = aligned;
      if (
        !["PER", "ORG", "LOC", "MISC"].includes(label) ||
        !word ||
        !Number.isFinite(score) ||
        score < 0 ||
        score > 1 ||
        !Number.isInteger(start) ||
        !Number.isInteger(end) ||
        start < 0 ||
        end <= start ||
        end > text.length
      )
        throw Error(
          `Model returned an invalid entity span: ${JSON.stringify(item)}`,
        );
      searchFrom = end;
      return { text: text.slice(start, end), label, score, start, end };
    });
  return { entities };
}

self.addEventListener("message", async ({ data: message = {} }) => {
  const { requestId, type, data } = message;
  try {
    if (type === "checkModelSupports") {
      configure(data?.workerConfig);
      send(requestId, type, { ...capabilities, manifest: MANIFEST });
      return;
    }
    if (type === "init" || type === "download") {
      if (state.initializing || state.generating) throw Error("Worker is busy");
      state.initializing = true;
      state.initialized = false;
      try {
        const precision = data?.precision,
          device =
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
      if (state.initializing || !state.initialized || !classifier)
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
