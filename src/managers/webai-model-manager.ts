import {
  WebAIDevice,
  WebAIPrecision,
  SupportedPrecisionsDevicesMapType,
  ProgressType,
  WebAIPriorities,
  PriorityConfig,
  WebAIWorkerConfig,
  WebAIWorkerManifest,
} from "../utils/types";
import { checkIsModelDownloaded, clearModelCache } from "../utils/utils";
import { createWorkerError } from "../utils/errors";
import { WorkerManager } from "./webai-worker-manager";

const CAPABILITY_TIMEOUT_MS = 10_000;
const MODEL_LOAD_TIMEOUT_MS = 15 * 60_000;

export class ModelManager {
  private _modelId: string;
  private _cacheModelId: string;
  private _precision: WebAIPrecision | null = null;
  private _device: WebAIDevice | null = null;
  private _modelSupportedPrecisions: WebAIPrecision[] | null = null;
  private _supportedPrecisionsDevicesMap: SupportedPrecisionsDevicesMapType | null = null;
  private _doesSupportStreamGeneration: boolean | null = null;
  private _externalInterrupt: boolean | null = null;
  private _manifest: WebAIWorkerManifest | null = null;
  private readonly _workerManager: WorkerManager;
  private readonly _workerConfig?: WebAIWorkerConfig;

  constructor(
    modelId: string,
    workerManager: WorkerManager,
    workerConfig?: WebAIWorkerConfig,
  ) {
    this._modelId = modelId;
    this._cacheModelId = modelId;
    this._workerManager = workerManager;
    this._workerConfig = workerConfig;
  }

  async checkModelSupports(): Promise<void> {
    const data = await this._workerManager.request<{
      supportedPrecisions: WebAIPrecision[];
      supportedPrecisionsDevicesMap: SupportedPrecisionsDevicesMapType;
      doesSupportStreamGeneration: boolean;
      externalInterrupt: boolean;
      cacheModelId?: string;
      manifest: WebAIWorkerManifest;
    }>("checkModelSupports", { workerConfig: this._workerConfig }, {
      timeoutMs: CAPABILITY_TIMEOUT_MS,
    });
    if (
      !Array.isArray(data.supportedPrecisions) ||
      !data.supportedPrecisionsDevicesMap ||
      data.manifest?.contractVersion !== "1.0" ||
      !data.manifest.model?.displayName ||
      !data.manifest.model?.description ||
      !data.manifest.model?.intendedUses?.length ||
      !data.manifest.model?.limitations?.length ||
      !data.manifest.model?.provider ||
      !data.manifest.model?.license ||
      !data.manifest.model?.sourceRepository ||
      !data.manifest.model?.lastUpdated ||
      !data.manifest.model?.artifactRevision ||
      !data.manifest.runtime?.precisions ||
      !Array.isArray(data.manifest.benchmark?.results)
    ) {
      throw createWorkerError("Worker returned invalid capability metadata", "message_error");
    }
    this._modelSupportedPrecisions = data.supportedPrecisions;
    this._supportedPrecisionsDevicesMap = data.supportedPrecisionsDevicesMap;
    this._doesSupportStreamGeneration = Boolean(data.doesSupportStreamGeneration);
    this._externalInterrupt = Boolean(data.externalInterrupt);
    this._cacheModelId = data.cacheModelId || this._modelId;
    this._manifest = data.manifest;
  }

  async sendInitMessage(
    precision: WebAIPrecision,
    device: WebAIDevice,
    onDownloadProgress?: (progress: ProgressType) => void,
    callbackThrottle: number = 1000,
  ): Promise<boolean> {
    let lastCallbackTime = 0;
    const data = await this._workerManager.request<{ status: string }>(
      "init",
      { precision, device },
      {
        timeoutMs: MODEL_LOAD_TIMEOUT_MS,
        onProgress: (value) => {
          const now = Date.now();
          if (onDownloadProgress && now - lastCallbackTime >= callbackThrottle) {
            lastCallbackTime = now;
            onDownloadProgress(value as ProgressType);
          }
        },
      },
    );
    if (data.status !== "success") {
      throw createWorkerError("Worker failed to initialize model", "initialization_failed");
    }
    this._device = device;
    this._precision = precision;
    return true;
  }

  async downloadModel(
    precision: WebAIPrecision,
    onDownloadProgress?: (progress: ProgressType) => void,
    callbackThrottle: number = 1000
  ): Promise<boolean> {
    let lastCallbackTime = 0;
    const data = await this._workerManager.request<{ status: string }>(
      "download",
      { precision },
      {
        timeoutMs: MODEL_LOAD_TIMEOUT_MS,
        onProgress: (value) => {
          const now = Date.now();
          if (onDownloadProgress && now - lastCallbackTime >= callbackThrottle) {
            lastCallbackTime = now;
            onDownloadProgress(value as ProgressType);
          }
        },
      },
    );
    return data.status === "success";
  }

  private validatePrecisionAndDevice(
    precision: WebAIPrecision,
    device: WebAIDevice
  ): void {
    if (!this._modelSupportedPrecisions?.includes(precision)) {
      throw createWorkerError(
        `Precision '${precision}' is not supported for this model. Supported precisions are: ${this._modelSupportedPrecisions?.join(", ")}`,
        'initialization_failed'
      );
    }

    const supportedDevicesForPrecision =
      this._supportedPrecisionsDevicesMap?.[precision]?.supportedDevices || [];

    if (!supportedDevicesForPrecision.includes(device)) {
      throw createWorkerError(
        `Device '${device}' is not supported for precision '${precision}'. Supported devices for this precision are: ${supportedDevicesForPrecision.join(", ")}`,
        'initialization_failed'
      );
    }
  }

  private async validateStorageAndWebGPU(
    device: WebAIDevice,
    precision: WebAIPrecision,
    isWebGPUAvailable: boolean | null,
    storageQuota: StorageEstimate | null
  ): Promise<void> {
    if (device === "webgpu" && !isWebGPUAvailable) {
      throw createWorkerError(
        "WebGPU is not available in your current browser. We recommend using the latest version of Chrome. You can find a list of supported browsers at https://caniuse.com/webgpu",
        'initialization_failed'
      );
    }

    const modelSize = this._supportedPrecisionsDevicesMap?.[precision]?.size;
    if (!modelSize) {
      throw createWorkerError(
        `Model size information not available for precision '${precision}'.`,
        'initialization_failed'
      );
    }

    const modelKeys = this._supportedPrecisionsDevicesMap?.[precision]?.modelKeys ?? [];
    const isDownloaded = await checkIsModelDownloaded(
      precision,
      modelKeys,
      this._cacheModelId,
    );
    if (!isDownloaded && storageQuota?.quota && storageQuota.quota > 0) {
      const leftStorage = storageQuota.quota - (storageQuota.usage ?? 0);
      if (modelSize > leftStorage) {
        throw createWorkerError(
          `Not enough storage to download the model. Required: ${modelSize} bytes, Available: ${leftStorage} bytes.`,
          'initialization_failed'
        );
      }
    }
  }

  async initModel({
    precision,
    device,
    priorities,
    onDownloadProgress,
    callbackThrottle = 1000,
    isWebGPUAvailable,
    storageQuota,
  }: {
    precision?: WebAIPrecision;
    device?: WebAIDevice;
    priorities?: WebAIPriorities;
    onDownloadProgress?: (progress: ProgressType) => void;
    callbackThrottle?: number;
    isWebGPUAvailable: boolean | null;
    storageQuota: StorageEstimate | null;
  }): Promise<boolean> {
    if (!this._supportedPrecisionsDevicesMap || !this._modelSupportedPrecisions) {
      throw createWorkerError(
        "Model support information is not available. Make sure to create the instance properly.",
        'initialization_failed'
      );
    }

    if (precision && device) {
      this.validatePrecisionAndDevice(precision, device);
      await this.validateStorageAndWebGPU(device, precision, isWebGPUAvailable, storageQuota);

      const result = await this.#initWebAI(
        precision,
        device,
        onDownloadProgress,
        callbackThrottle
      );

      if (!result) {
        throw createWorkerError("Failed to initialize model", 'initialization_failed');
      }

      return true;
    }

    const defaultPriorities = this.getDefaultPriorities();
    const priorityList = priorities?.length ? priorities : defaultPriorities;

    for (const config of priorityList) {
      if (await this.tryConfiguration(
        config,
        isWebGPUAvailable,
        storageQuota,
        onDownloadProgress,
        callbackThrottle
      )) {
        return true;
      }
    }

    throw createWorkerError(
      "Could not run the model, none of the configurations (precision, device) " +
      "specified in the priorities list works. Please verify the model requirements and the device you are running on.",
      'initialization_failed'
    );
  }

  private getDefaultPriorities(): WebAIPriorities {
    return [
      { precision: "q4", device: "webgpu" },
      { precision: "q4", device: "wasm" },
      { precision: "fp32", device: "webgpu" },
      { precision: "fp32", device: "wasm" },
    ];
  }

  private async tryConfiguration(
    config: PriorityConfig,
    isWebGPUAvailable: boolean | null,
    storageQuota: StorageEstimate | null,
    onDownloadProgress?: (progress: ProgressType) => void,
    callbackThrottle: number = 1000
  ): Promise<boolean> {
    const precision = config.precision;
    const device = config.device;

    if (!this._modelSupportedPrecisions?.includes(precision)) {
      return false;
    }

    const supportedDevices = this._supportedPrecisionsDevicesMap?.[precision]?.supportedDevices || [];
    if (!supportedDevices.includes(device)) {
      return false;
    }

    if (device === "webgpu" && !isWebGPUAvailable) {
      return false;
    }

    const modelSize = this._supportedPrecisionsDevicesMap?.[precision]?.size;
    if (!modelSize) {
      return false;
    }

    const modelKeys = this._supportedPrecisionsDevicesMap?.[precision]?.modelKeys ?? [];
    const isDownloaded = await checkIsModelDownloaded(
      precision,
      modelKeys,
      this._cacheModelId,
    );
    if (!isDownloaded && storageQuota?.quota && storageQuota.quota > 0) {
      const leftStorage = storageQuota.quota - (storageQuota.usage ?? 0);
      if (modelSize > leftStorage) {
        console.warn(`Skipping configuration (${precision}/${device}): Not enough storage space`);
        return false;
      }
    }

    try {
      console.log("Trying configuration:", precision, device);

      const result = await this.#initWebAI(
        precision,
        device,
        onDownloadProgress,
        callbackThrottle
      );

      if (result) {
        return true;
      }
    } catch (error) {
      console.error(`[WebAI ModelManager] Failed to initialize with ${precision}/${device}:`, error);
    }

    return false;
  }

  async #initWebAI(
    precision: WebAIPrecision,
    device: WebAIDevice,
    onDownloadProgress?: (progress: ProgressType) => void,
    callbackThrottle: number = 1000
  ): Promise<boolean> {
    await this.sendInitMessage(
      precision,
      device,
      onDownloadProgress,
      callbackThrottle,
    );
    return true;
  }

  async reinitializeAfterInterrupt(): Promise<void> {
    if (!this._precision || !this._device) {
      throw createWorkerError(
        "Cannot reinitialize: missing required state (precision or device)",
        'initialization_failed'
      );
    }

    try {
      await this.sendInitMessage(this._precision, this._device);

      console.log(`Model reinitialized after interrupt: ${this._precision}/${this._device}`);
    } catch (error) {
      console.error("Failed to reinitialize model after interrupt:", error);

      this._precision = null;
      this._device = null;

      throw error;
    }
  }

  async clearCache(): Promise<void> {
    return clearModelCache(this._cacheModelId);
  }

  async isModelDownloaded(precision: WebAIPrecision): Promise<boolean> {
    const modelKeys = this._supportedPrecisionsDevicesMap?.[precision]?.modelKeys;
    if (!modelKeys) return false;
    return checkIsModelDownloaded(precision, modelKeys, this._cacheModelId);
  }

  async deletePrecisionSpecificModel(precision: WebAIPrecision): Promise<void> {
    if (!this._supportedPrecisionsDevicesMap?.[precision]) {
      console.warn(`Precision ${precision} is not supported for model ${this._modelId}`);
      return;
    }

    const modelKeys = this._supportedPrecisionsDevicesMap[precision].modelKeys;

    const downloadedPrecisions: WebAIPrecision[] = [];

    if (this._modelSupportedPrecisions) {
      for (const supportedPrecision of this._modelSupportedPrecisions) {
        if (supportedPrecision === precision) continue;

        const otherModelKeys = this._supportedPrecisionsDevicesMap?.[supportedPrecision]?.modelKeys;
        if (otherModelKeys) {
          const isDownloaded = await checkIsModelDownloaded(
            supportedPrecision,
            otherModelKeys,
            this._cacheModelId
          );
          if (isDownloaded) {
            downloadedPrecisions.push(supportedPrecision);
          }
        }
      }
    }

    if (downloadedPrecisions.length === 0) {
      console.log(`Deleting all cached files for model ${this._modelId} (only precision: ${precision})`);
      await this.clearCache();
      return;
    }

    console.log(`Deleting precision-specific files for ${this._modelId}, precision: ${precision}`);
    console.log(`Keeping other precisions: ${downloadedPrecisions.join(', ')}`);

    const cache = await caches.open("transformers-cache");

    for (const modelKey of modelKeys) {
      try {
        const cacheKeys = await cache.keys();
        for (const request of cacheKeys) {
          const url = request.url;
          if (url.includes(this._cacheModelId) && url.includes(modelKey)) {
            await cache.delete(request);
          }
        }
      } catch (error) {
        console.warn(`Failed to delete cached files for model key ${modelKey}:`, error);
      }
    }

    console.log(`Successfully deleted model files for precision: ${precision}`);
  }

  get precision(): WebAIPrecision | null {
    return this._precision;
  }

  get device(): WebAIDevice | null {
    return this._device;
  }

  get modelSupportedPrecisions(): WebAIPrecision[] | null {
    return this._modelSupportedPrecisions;
  }

  get modelSupportedPrecisionsDevicesMap(): SupportedPrecisionsDevicesMapType | null {
    return this._supportedPrecisionsDevicesMap;
  }

  get doesSupportStreamGeneration(): boolean | null {
    return this._doesSupportStreamGeneration;
  }

  get externalInterrupt(): boolean | null {
    return this._externalInterrupt;
  }

  get manifest(): WebAIWorkerManifest | null {
    return this._manifest;
  }
}
