// src/WebAI.ts
import { ModelManager } from "./managers/webai-model-manager";
import { QueueManager } from "./managers/webai-queue-manager";
import { WorkerManager } from "./managers/webai-worker-manager";
import { REMOTE_WORKER_ENDPOINT, VERSION } from "./utils/constants";
import { WebAIMode, WebAIDevice, WebAIPrecision, WebAIPriorities, ProgressType, AuthRetryOptions, OnAuthCallback } from "./utils/types";
import { checkIsModelDownloaded, checkIsWebGPUAvailable, checkStorageQuota } from "./utils/utils";

export class WebAI {
  public version = VERSION;
  private _modelId: string;
  private _dev: boolean;
  private _workerPath?: string;
  private _onAuth?: OnAuthCallback;
  private _authRetryOptions?: AuthRetryOptions;
  private _isWebGPUAvailable: boolean | null = null;
  private _storageQuota: StorageEstimate | null = null;
  private _isInitialized: boolean = false;

  // Managers
  private _workerManager: WorkerManager;
  private _queueManager: QueueManager;
  private _modelManager: ModelManager;
 
  private constructor({
    modelId,
    dev = false,
    workerPath,
    onAuth,
    authRetryOptions,
  }: {
    modelId: string;
    dev?: boolean;
    workerPath?: string;
    onAuth?: OnAuthCallback;
    authRetryOptions?: AuthRetryOptions;
  }) {
    this._modelId = modelId;
    this._dev = dev;
    this._workerPath = workerPath;
    this._onAuth = onAuth;
    this._authRetryOptions = authRetryOptions;

    // Initialize managers - pass isDev, onAuth, and authRetryOptions to ModelManager
    this._workerManager = new WorkerManager(this.#getWorkerPath(), modelId);
    this._queueManager = new QueueManager();
    this._modelManager = new ModelManager(modelId, this._workerManager, dev, onAuth, authRetryOptions);
  }

  static async create({
    modelId,
    dev = false,
    workerPath,
    onAuth,
    authRetryOptions,
  }: {
    modelId: string;
    dev?: boolean;
    workerPath?: string;
    onAuth?: OnAuthCallback;
    authRetryOptions?: AuthRetryOptions;
  }): Promise<WebAI> {
    const instance = new WebAI({ modelId, dev, workerPath, onAuth, authRetryOptions });
    await instance.#createInstance();
    return instance;
  }

  async #createInstance(): Promise<void> {
    try {
      await this._workerManager.initWorker();
      this._isWebGPUAvailable = await checkIsWebGPUAvailable();
      this._storageQuota = await checkStorageQuota();
      await this._modelManager.checkModelSupports(this._workerManager.worker);
    } catch (error) {
      throw error;
    }
  }

  async init({
    mode = "auto",
    precision,
    device,
    priorities,
    onDownloadProgress,
    callbackThrottle = 1000
  }: {
    mode: WebAIMode;
    precision?: WebAIPrecision;
    device?: WebAIDevice;
    priorities?: WebAIPriorities;
    onDownloadProgress?: (progress: ProgressType) => void;
    callbackThrottle?: number;
  }): Promise<boolean> {
    
    const result = await this._modelManager.initModel({
      mode,
      precision,
      device,
      priorities,
      onDownloadProgress,
      callbackThrottle,
      worker: this._workerManager.worker,
      isWebGPUAvailable: this._isWebGPUAvailable,
      storageQuota: this._storageQuota,
    });

    this._isInitialized = result;
    return result;
  }

  async generate(data: { userInput: any, modelConfig?: object, generateConfig?: object }): Promise<any> {
    if (!this._isInitialized) {
      throw new Error("WebAI instance must be initialized before generating. Call init() first.");
    }

    return this._queueManager.enqueueGenerate(
      data,
      this._workerManager.worker,
      this._modelManager.mode
    );
  }

  async generateStream(data: {
    userInput: any,
    modelConfig?: object,
    generateConfig?: object,
    onStream: (chunk: any) => void
  }): Promise<void> {
    if (!this._isInitialized) {
      throw new Error("WebAI instance must be initialized before generating. Call init() first.");
    }

    return this._queueManager.enqueueGenerateStream(
      data,
      this._workerManager.worker,
      this._modelManager.mode
    );
  }

  clearQueue({ interrupt = true }: { interrupt?: boolean } = {}): void {
    this._queueManager.clearQueue();
    if (interrupt) {
      this.interrupt({clearQueue: false});
    }
  }

  async deleteDownloadedModel({ precision }: { precision?: WebAIPrecision } = {}): Promise<void> {
    if (precision) {
      // Delete specific precision
      await this._modelManager.deletePrecisionSpecificModel(precision);
    } else {
      // Delete everything (original behavior)
      await this._modelManager.clearCache();
    }
  }

  terminate() {
    this.clearQueue();
    this._workerManager.terminateWorker();
    this._isInitialized = false;
  }

  async interrupt({ clearQueue = true }: { clearQueue?: boolean } = {}): Promise<void> {
    // Clear queue first if requested
    if (clearQueue) {
      this._queueManager.clearQueue();
    }

    // Handle interruption based on device type
    if (this._modelManager.device === 'wasm' || this._modelManager.externalInterrupt) {
      await this._handleWasmInterrupt();
    } else {
      this._workerManager.interruptWorker();
    }
  }

  private async _handleWasmInterrupt(): Promise<void> {
    try {
      // Notify queue manager that we're doing a forced interruption
      this._queueManager.handleForcedInterruption();
      
      // Terminate and reload worker
      await this._workerManager.terminateAndReload();
      
      // Re-initialize the model to restore the same state
      if (this._isInitialized && this._modelManager.mode) {
        await this._modelManager.reinitializeAfterInterrupt(
          this._workerManager.worker,
        );
      }
    } catch (error) {
      console.error('Error during interrupt:', error);
      // Even if reinitialization fails, we've successfully interrupted
      // The user can call init() again if needed
    }
  }

  clearMemory() {
    this._workerManager.clearMemory();
    this._isInitialized = false;
  }

  async checkIsModelDownloaded({ precision }: {
    precision: WebAIPrecision;
  }) {
    const modelKeys = this._modelManager.modelSupportedPrecisionsDevicesMap?.[precision]?.modelKeys;
    if (!modelKeys) {
      console.warn("Model does not support this precision", precision);
      return false;
    }
    const isDownloaded = await checkIsModelDownloaded(
      precision,
      modelKeys,
      this._modelId
    );
    return isDownloaded;
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
      this._workerManager.worker, 
      precision, 
      onDownloadProgress, 
      callbackThrottle
    );
  }

  #getWorkerPath(): string {
    // If custom workerPath is provided, use it directly
    if (this._workerPath) {
      return this._workerPath;
    }

    // Otherwise, use the default remote worker endpoint
    const basePath = `${REMOTE_WORKER_ENDPOINT}/models/${this._modelId}/js-worker`;
    const workerFile = `${this._modelId}.worker.js`;
    
    return `${basePath}/${workerFile}`;
 
  }

  get modelId(): string {
    return this._modelId;
  }

  get dev(): boolean {
    return this._dev;
  }

  get mode(): WebAIMode | null {
    return this._modelManager.mode;
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
    return this._modelManager.modelSupportedPrecisionsDevicesMap
  }

  get modelSupportedPrecisions() {
    return this._modelManager.modelSupportedPrecisions;
  }

  get modelSupportedModes() {
    return this._modelManager.modelSupportedModes;
  }

  get isGenerating(): boolean {
    return this._queueManager.isGenerating;
  }

  get queueSize(): number {
    return this._queueManager.queueSize;
  }

  get doesSupportStreamGeneration(): boolean | null{
    return this._modelManager.doesSupportStreamGeneration;
  }

  get isInitialized(): boolean {
    return this._isInitialized;
  }
}