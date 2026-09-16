export type WebAIPrecision = "fp32" | "fp16" | "q8" | "int8" | "uint8" | "q4" | "bnb4" | "q4f16";
export type WebAIDevice = "wasm" | "webgpu";

/** Runtime settings passed to model workers. No telemetry is collected. */
export type WebAIWorkerConfig = {
  /** Repository identifier understood by Transformers.js. */
  modelId?: string;
  /** Optional immutable Hugging Face commit/revision. */
  modelRevision?: string;
  /** Optional custom model host, for example https://huggingface.co. */
  remoteHost?: string;
  /** Path template appended to remoteHost by Transformers.js. */
  remotePathTemplate?: string;
  /** Local model root used when allowLocalModels is enabled. */
  localModelPath?: string;
  allowRemoteModels?: boolean;
  allowLocalModels?: boolean;
  useBrowserCache?: boolean;
};

export type WebAIOptions = {
  modelId: string;
  /** Defaults to /workers/{modelId}.worker.js on the application's origin. */
  workerPath?: string;
  workerConfig?: WebAIWorkerConfig;
  /** Maximum time to wait for the worker boot handshake. */
  workerInitTimeoutMs?: number;
};

export type ProgressType = {
  file?: string;
  name?: string;
  loaded?: number;
  total?: number;
  status?: string;
  progress?: number;
};

export type WebAIRuntime = {
  /** Total execution time after leaving the queue, including worker messaging. */
  durationMs: number;
};

export type WebAIResult<T = unknown> = {
  /** The model-specific output, unchanged by the core runtime. */
  result: T;
  runtime: WebAIRuntime;
};

export type WebAIWorkerField = {
  name: string;
  type: string;
  description: string;
  required?: boolean;
  default?: unknown;
  enum?: unknown[];
  minimum?: number;
  maximum?: number;
};

export type WebAIWorkerManifest = {
  contractVersion: "1.0";
  model: {
    id: string;
    displayName: string;
    /** Original model creator, not the ONNX conversion publisher. */
    provider: string;
    /** Authoritative homepage for the original model provider. */
    providerUrl: string;
    /** SPDX license identifier, or LicenseRef-* for a non-standard license. */
    license: string;
    /** Authoritative license text for the original model. */
    licenseUrl: string;
    /** ISO timestamp from the authoritative original model repository. */
    lastUpdated: string;
    sourceRepository: string;
    repository: string;
    artifactRevision: string;
    artifactLastUpdated?: string;
    task: string;
    description: string;
    intendedUses: string[];
    limitations: string[];
  };
  runtime: {
    precisions: SupportedPrecisionsDevicesMapType;
  };
  input: {
    description: string;
    alternatives: Array<{
      name: string;
      description: string;
      fields: WebAIWorkerField[];
    }>;
  };
  config: {
    model: WebAIWorkerField[];
    generation: WebAIWorkerField[];
  };
  output: {
    description: string;
    fields: WebAIWorkerField[];
    example: unknown;
  };
  benchmark?: {
    report: string;
    testedAt: string;
    fixture: {
      file: string;
      url: string;
      size: number;
      sha1: string;
    durationSeconds: number;
    };
    note?: string;
    environment: string;
    invalidatedBy: string[];
    results: Array<{
      precision: WebAIPrecision;
      device: WebAIDevice;
      status: "pass" | "fail";
      runtime: {
        loadMs: number | null;
        inferenceMs: number | null;
        totalMs: number;
        grade: string;
        realTimeFactor?: number;
        loadGrade?: string;
      };
      quality?: { grade: string; wordErrorRate: number; gibberish: boolean };
      failure?: string;
    }>;
  };
  operations: Array<"checkModelSupports" | "init" | "download" | "generate" | "generateStream" | "clearMemory">;
};

export type WhisperTimestamp = [startSeconds: number, endSeconds: number | null];

export type WhisperTranscriptionChunk = {
  text: string;
  timestamp: WhisperTimestamp;
};

export type WhisperTranscriptionResult = {
  result: string;
  /** Present when return_timestamps is "word" or true. */
  chunks?: WhisperTranscriptionChunk[];
};

export type WhisperModelConfig = {
  language?: string;
  task?: "transcribe" | "translate";
  /** "word" returns word-level timestamps; true returns segment timestamps. */
  return_timestamps?: "word" | boolean;
  condition_on_prev_tokens?: boolean;
  no_speech_threshold?: number;
  num_beams?: number;
};

export type WhisperGenerationConfig = {
  chunk_length_s?: number;
  stride_length_s?: number;
};

export type PriorityConfig = {
  precision: WebAIPrecision;
  device: WebAIDevice;
};

export type WebAIPriorities = PriorityConfig[];

export type SupportedPrecisionsDevicesMapType = Partial<Record<WebAIPrecision, {
    supportedDevices: WebAIDevice[];
    /** Estimated complete first download, including shared tokenizer/config files. */
    size: number;
    /** Exact selected ONNX encoder and merged-decoder weight bytes. */
    weightsSize: number;
    modelKeys: string[];
    dtype?: WebAIPrecision | Record<string, WebAIPrecision>;
  }>>;
