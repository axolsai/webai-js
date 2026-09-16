import { QueueManager } from "./managers/webai-queue-manager";
import { ModelManager } from "./managers/webai-model-manager";
import { WorkerManager } from "./managers/webai-worker-manager";
import { checkIsWebGPUAvailable, checkStorageQuota } from "./utils/utils";
import { DEFAULT_WORKER_BASE_PATH, DEFAULT_WORKER_INIT_TIMEOUT_MS } from "./utils/constants";
import {
  WebAIDevice,
  WebAIPrecision,
  WebAIPriorities,
  ProgressType,
  WebAIOptions,
  WebAIResult,
} from "./utils/types";

export class WebAI {
  private _modelId: string;
  private _workerPath?: string;
  private _workerManager: WorkerManager;
  private _modelManager: ModelManager;
  private _queueManager: QueueManager;
  private _isWebGPUAvailable: boolean | null = null;
  private _storageQuota: StorageEstimate | null = null;
  private _isInitialized: boolean = false;
  private _isInitializing: boolean = false;

  private constructor({
    modelId,
    workerPath,
    workerConfig,
    workerInitTimeoutMs = DEFAULT_WORKER_INIT_TIMEOUT_MS,
  }: WebAIOptions) {
    this._modelId = modelId;
    this._workerPath = workerPath;

    const workerUrl = this.#getWorkerPath();

    this._workerManager = new WorkerManager(workerUrl, workerInitTimeoutMs);
    this._modelManager = new ModelManager(this._modelId, this._workerManager, workerConfig);
    this._queueManager = new QueueManager(this._workerManager);
  }

  static async create(options: WebAIOptions): Promise<WebAI> {
    const instance = new WebAI(options);
    await instance.#createInstance();
    return instance;
  }

  async #createInstance(): Promise<void> {
    try {
      await this._workerManager.initWorker();
      this._isWebGPUAvailable = await checkIsWebGPUAvailable();
      this._storageQuota = await checkStorageQuota();
      await this._modelManager.checkModelSupports();
    } catch (error) {
      this._workerManager.terminateWorker();
      throw error;
    }
  }

  async init({
    precision,
    device,
    priorities,
    onDownloadProgress,
    callbackThrottle = 1000
  }: {
    precision?: WebAIPrecision;
    device?: WebAIDevice;
    priorities?: WebAIPriorities;
    onDownloadProgress?: (progress: ProgressType) => void;
    callbackThrottle?: number;
  } = {}): Promise<boolean> {
    if (this._isInitializing) throw new Error("WebAI initialization is already in progress.");
    if (this._queueManager.isGenerating) throw new Error("Cannot initialize while generating.");
    this._isInitializing = true;
    this._isInitialized = false;
    try {
      const result = await this._modelManager.initModel({
        precision,
        device,
        priorities,
        onDownloadProgress,
        callbackThrottle,
        isWebGPUAvailable: this._isWebGPUAvailable,
        storageQuota: this._storageQuota,
      });
      this._isInitialized = result;
      return result;
    } finally {
      this._isInitializing = false;
    }
  }

  async generate<T = unknown>(
    data: { userInput?: any; modelConfig?: object; generateConfig?: object } | any,
  ): Promise<WebAIResult<T>> {
    if (!this._isInitialized) {
      throw new Error("WebAI instance must be initialized before generating. Call init() first.");
    }

    const payload = data?.userInput ? data : { userInput: data };

    return this._queueManager.enqueueGenerate<T>(payload);
  }

  async generateStream(data: {
    userInput: any,
    modelConfig?: object,
    generateConfig?: object,
    onStream: (chunk: any) => void
  }): Promise<void> {
    if (!this._modelManager.doesSupportStreamGeneration) {
      throw new Error(`Stream generation is not supported by model '${this._modelId}'.`);
    }
    if (!this._isInitialized) {
      throw new Error("WebAI instance must be initialized before generating. Call init() first.");
    }

    const { onStream, ...payload } = data;
    return this._queueManager.enqueueGenerateStream(payload, onStream);
  }

  async clearQueue({ interrupt = true }: { interrupt?: boolean } = {}): Promise<void> {
    this._queueManager.clearQueue();
    if (interrupt) {
      await this.interrupt({ clearQueue: false });
    }
  }

  async deleteDownloadedModel({ precision }: { precision?: WebAIPrecision } = {}): Promise<void> {
    if (this._isInitialized) await this.clearMemory();
    if (precision) {
      await this._modelManager.deletePrecisionSpecificModel(precision);
    } else {
      await this._modelManager.clearCache();
    }
    this._isInitialized = false;
  }

  async terminate(): Promise<void> {
    this._queueManager.cancelAll();
    this._workerManager.terminateWorker();
    this._isInitialized = false;
  }

  async interrupt({ clearQueue = true }: { clearQueue?: boolean } = {}): Promise<void> {
    if (clearQueue) {
      this._queueManager.clearQueue();
    }

    await this._handleWorkerInterrupt();
  }

  private async _handleWorkerInterrupt(): Promise<void> {
    try {
      // Termination is the reliable cancellation primitive because a busy inference worker may
      // not service an interrupt message. Recreate it and restore the cached model configuration.
      this._queueManager.cancelCurrent();
      await this._workerManager.restart();
      await this._modelManager.checkModelSupports();

      if (this._isInitialized) {
        await this._modelManager.reinitializeAfterInterrupt();
      }
    } catch (error) {
      this._isInitialized = false;
      throw error;
    }
  }

  async clearMemory(): Promise<void> {
    if (this._queueManager.isGenerating) {
      throw new Error("Cannot clear model memory while generation is active. Interrupt first.");
    }
    await this._workerManager.clearMemory();
    this._isInitialized = false;
  }

  async checkIsModelDownloaded({ precision }: {
    precision: WebAIPrecision;
  }) {
    return this._modelManager.isModelDownloaded(precision);
  }

  async downloadModel({
    precision,
    onDownloadProgress,
    callbackThrottle = 1000
  }: {
    precision: WebAIPrecision;
    onDownloadProgress?: (progress: ProgressType) => void;
    callbackThrottle?: number;
  }) {
    return this._modelManager.downloadModel(
      precision,
      onDownloadProgress,
      callbackThrottle
    );
  }

  #getWorkerPath(): string {
    if (this._workerPath) {
      return this._workerPath;
    }

    const workerFile = `${this._modelId}.worker.js`;
    return `${DEFAULT_WORKER_BASE_PATH}/${workerFile}`;
  }

  get modelId(): string {
    return this._modelId;
  }

  get precision(): WebAIPrecision | null {
    return this._modelManager.precision;
  }

  get device(): WebAIDevice | null {
    return this._modelManager.device;
  }

  get isWebGPUAvailable(): boolean | null {
    return this._isWebGPUAvailable;
  }

  get deviceStorageQuota(): StorageEstimate | null {
    return this._storageQuota;
  }

  get modelSupportedPrecisionsDevicesMap() {
    return this._modelManager.modelSupportedPrecisionsDevicesMap;
  }

  get modelSupportedPrecisions(): WebAIPrecision[] | null {
    return this._modelManager.modelSupportedPrecisions;
  }

  get doesSupportStreamGeneration(): boolean | null {
    return this._modelManager.doesSupportStreamGeneration;
  }

  get externalInterrupt(): boolean | null {
    return this._modelManager.externalInterrupt;
  }

  /** Self-described input, configuration, output, and operation contract. */
  get modelManifest() {
    return this._modelManager.manifest;
  }

  get isGenerating(): boolean {
    return this._queueManager.isGenerating;
  }

  get queueLength(): number {
    return this._queueManager.queueLength;
  }

  get isInitialized(): boolean {
    return this._isInitialized;
  }
}

export default WebAI;
