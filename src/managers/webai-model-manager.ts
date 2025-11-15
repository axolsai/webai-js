import {
    WebAIMode,
    WebAIDevice,
    WebAIPrecision,
    SupportedPrecisionsDevicesMapType,
    ProgressType,
    WebAIPriorities
  } from "../utils/types";
  import { checkIsModelDownloaded, clearModelCache } from "../utils/utils";
  import { createWorkerError, isWorkerError } from "../utils/errors";
import { WorkerManager } from "./webai-worker-manager";

  
  export class ModelManager {
    private _modelId: string;
    private _workerManager: WorkerManager | null = null; 
    private _mode: WebAIMode | null = null;
    private _precision: WebAIPrecision | null = null;
    private _device: WebAIDevice | null = null;
    private _modelSupportedModes: WebAIMode[] | null = null;
    private _modelSupportedPrecisions: WebAIPrecision[] | null = null;
    private _supportedPrecisionsDevicesMap: SupportedPrecisionsDevicesMapType | null = null;
    private _doesSupportStreamGeneration: boolean | null = null;
    private _externalInterrupt: boolean | null = null;
  
    constructor(modelId: string, workerManager: WorkerManager | null = null) {
      this._modelId = modelId;
      this._workerManager = workerManager;
    }
  
    async checkModelSupports(worker: Worker | null): Promise<void> {
      if (!worker) {
        throw createWorkerError(
          "Worker is not initialized",
          'initialization_failed'
        );
      }
  
      return new Promise<void>((resolve, reject) => {
        const cleanup = () => {
          worker.removeEventListener("message", handleMessage);
          worker.removeEventListener("error", handleError);
        };
  
        const handleMessage = (event: MessageEvent) => {
          if (event.data.type === "error") {
            cleanup();
            reject(createWorkerError(
              event.data.data.message || "Error checking model support",
              'runtime_error'
            ));
            return;
          }
  
          if (event.data.type === "checkModelSupports") {
            cleanup();
            const {
              supportedModes,
              supportedPrecisions,
              supportedPrecisionsDevicesMap,
              doesSupportStreamGeneration,
              externalInterrupt
            } = event.data.data;
            
            this._modelSupportedModes = supportedModes;
            this._modelSupportedPrecisions = supportedPrecisions;
            this._supportedPrecisionsDevicesMap = supportedPrecisionsDevicesMap;
            this._doesSupportStreamGeneration = doesSupportStreamGeneration;
            this._externalInterrupt = externalInterrupt;
            
            resolve();
          }
        };
  
        const handleError = (error: ErrorEvent) => {
          cleanup();
          reject(createWorkerError(
            `Worker error during model support check: ${error.message}`,
            'runtime_error'
          ));
        };
  
        worker.addEventListener("message", handleMessage);
        worker.addEventListener("error", handleError);
  
        try {
          worker.postMessage({ type: "checkModelSupports" });
        } catch (error) {
          cleanup();
          reject(createWorkerError(
            error instanceof Error
              ? `Failed to check model supports: ${error.message}`
              : "Failed to check model supports: Unknown error",
            'runtime_error'
          ));
        }
      });
    }
  
    async sendInitMessage(
      worker: Worker | null,
      precision: WebAIPrecision,
      device: WebAIDevice
    ): Promise<boolean> {
      if (!worker) {
        throw createWorkerError(
          "Worker is not initialized",
          'initialization_failed'
        );
      }
  
      return new Promise((resolve, reject) => {
        const cleanup = () => {
          worker.removeEventListener("message", handleMessage);
          worker.removeEventListener("error", handleError);
        };
  
        const handleMessage = (event: MessageEvent) => {
          if (event.data.type === "error") {
            cleanup();
            reject(createWorkerError(
              event.data.data.message || "Unknown error",
              'runtime_error'
            ));
            return;
          }
  
          if (event.data.type === "init") {
            cleanup();
            if (event.data.data.status === "success") {
              this._device = device;
              this._precision = precision;
              resolve(true);
            } else {
              reject(createWorkerError(
                `Failed to initialize with device ${device} and precision ${precision}`,
                'runtime_error'
              ));
            }
          }
        };
  
        const handleError = (error: ErrorEvent) => {
          cleanup();
          reject(createWorkerError(
            `Worker error during initialization: ${error.message}`,
            'runtime_error'
          ));
        };
  
        worker.addEventListener("message", handleMessage);
        worker.addEventListener("error", handleError);
  
        try {
          worker.postMessage({
            type: "init",
            data: { precision, device }
          });
        } catch (error) {
          cleanup();
          reject(createWorkerError(
            error instanceof Error
              ? `Failed to send init request: ${error.message}`
              : "Failed to send init request: Unknown error",
            'runtime_error'
          ));
        }
      });
    }
  
    async downloadModel(
      worker: Worker | null,
      precision: WebAIPrecision,
      downloadProgressCallback?: (progress: ProgressType) => void,
      callbackThrottle: number = 500,
    ): Promise<boolean> {
      if (!worker) {
        throw createWorkerError(
          "Worker is not initialized",
          'initialization_failed'
        );
      }
  
      const modelKeys = this._supportedPrecisionsDevicesMap?.[precision]?.modelKeys || [];
      
      // Check if model is already downloaded
      const isModelDownloaded = await checkIsModelDownloaded(
        precision,
        modelKeys,
        this._modelId
      );
      
      if (isModelDownloaded) {
        // Model already exists - report 100% if callback provided
        if (downloadProgressCallback) {
          const actualSize = this._supportedPrecisionsDevicesMap?.[precision]?.size || 0;
          downloadProgressCallback({
            file: `${precision} models`,
            name: this._modelId,
            loaded: actualSize,
            total: actualSize,
            status: "downloaded",
            progress: 100
          });
        }
        return true;
      }
  
      // Model needs to be downloaded
      return new Promise((resolve, reject) => {
        let lastProgressUpdate = 0;
        const fileProgress: Record<string, ProgressType> = {};
  
        const cleanup = () => {
          worker.removeEventListener("message", handleMessage);
          worker.removeEventListener("error", handleError);
        };
  
        const handleMessage = async (event: MessageEvent) => {
          if (event.data.type === "error") {
            cleanup();
            reject(createWorkerError(
              event.data.data.message || "Download error",
              'runtime_error'
            ));
            return;
          }
  
          // Handle download progress updates
          if (event.data.type === "downloadProgress") {
            const progressData = event.data.data;
            const filename = progressData.file;
            
            if (!filename) return;
            
            const matchedKey = modelKeys.find(key => filename.includes(key));
            const now = Date.now();
            
            if (matchedKey && (now - lastProgressUpdate > callbackThrottle)) {
              fileProgress[matchedKey] = progressData;
  
              // Only report aggregated progress when all keys have data
              const allKeysPresent = modelKeys.every(key => fileProgress[key] !== undefined);
  
              if (allKeysPresent && downloadProgressCallback) {
                let totalLoaded = 0;
                let totalSize = 0;
  
                modelKeys.forEach(key => {
                  totalLoaded += fileProgress[key].loaded;
                  totalSize += fileProgress[key].total;
                });
  
                const progress = (totalLoaded / totalSize) * 100;
                
                if (progress >= 0 && progress <= 100) {
                  downloadProgressCallback({
                    file: `${precision} models`,
                    name: this._modelId,
                    loaded: totalLoaded,
                    total: totalSize,
                    status: "downloading",
                    progress: progress
                  });
                }
                lastProgressUpdate = now;
              }
            }
            return;
          }
  
          // Handle download completion
          if (event.data.type === "download" && event.data.data.status === "success") {
            if (downloadProgressCallback) {
              const actualSize = this._supportedPrecisionsDevicesMap?.[precision]?.size || 0;
              downloadProgressCallback({
                file: `${precision} models`,
                name: this._modelId,
                loaded: actualSize,
                total: actualSize,
                status: "downloaded",
                progress: 100
              });
            }
  
            cleanup();
  
            // Reload worker after successful download
            if (this._workerManager) {
              try {
                console.log("Reloading worker after successful download...");
                await this._workerManager.reloadWorker();
                
                if (!this._workerManager.worker) {
                  throw createWorkerError(
                    "Worker is null after reload",
                    'initialization_failed'
                  );
                }
                
                // Re-check model supports with the new worker instance
                await this.checkModelSupports(this._workerManager.worker);
                
              } catch (error) {
                console.error("Failed to reload worker after download:", error);
                reject(createWorkerError(
                  "Failed to reload worker after download",
                  'initialization_failed'
                ));
                return;
              }
            }
  
            resolve(true);
          }
        };
  
        const handleError = (error: ErrorEvent) => {
          cleanup();
          reject(createWorkerError(
            `Worker error during download: ${error.message}`,
            'runtime_error'
          ));
        };
  
        worker.addEventListener("message", handleMessage);
        worker.addEventListener("error", handleError);
  
        try {
          worker.postMessage({type: "download", data: {precision}});
        } catch (error) {
          cleanup();
          reject(createWorkerError(
            error instanceof Error
              ? `Failed to send download request: ${error.message}`
              : "Failed to send download request: Unknown error",
            'runtime_error'
          ));
        }
      });
    }
  
    private validateInitConfig(
      mode: WebAIMode,
      precision?: WebAIPrecision,
      device?: WebAIDevice
    ): void {
      // Validate mode support
      if (!this._modelSupportedModes?.includes(mode)) {
        throw createWorkerError(
          `Mode '${mode}' is not supported for this model.`,
          'initialization_failed'
        );
      }
  
      // For webai mode, validate precision and device
      if (mode === "webai") {
        if (!device || !precision) {
          throw createWorkerError(
            "For 'webai' mode, both device and precision must be specified.",
            'initialization_failed'
          );
        }
  
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
    }
  
    private validateStorageAndWebGPU(
      device: WebAIDevice,
      precision: WebAIPrecision,
      isWebGPUAvailable: boolean | null,
      storageQuota: StorageEstimate | null
    ): void {
      // Check WebGPU availability
      if (device === "webgpu" && !isWebGPUAvailable) {
        throw createWorkerError(
          "WebGPU is not available in your current browser. We recommend using the latest version of Chrome. You can find a list of supported browsers at https://caniuse.com/webgpu",
          'initialization_failed'
        );
      }
  
      // Check storage quota
      const modelSize = this._supportedPrecisionsDevicesMap?.[precision]?.size;
      if (!modelSize) {
        throw createWorkerError(
          `Model size information not available for precision '${precision}'.`,
          'initialization_failed'
        );
      }
  
      const leftStorage = (storageQuota?.quota ?? 0) - (storageQuota?.usage ?? 0);
      if (modelSize > leftStorage) {
        throw createWorkerError(
          `Not enough storage to download the model. Required: ${modelSize} bytes, Available: ${leftStorage} bytes.`,
          'initialization_failed'
        );
      }
    }
  
    async initModel({
      mode = "auto",
      precision,
      device,
      priorities,
      onDownloadProgress,
      callbackThrottle = 1000,
      worker,
      isWebGPUAvailable,
      storageQuota,
    }: {
      mode: WebAIMode;
      precision?: WebAIPrecision;
      device?: WebAIDevice;
      priorities?: WebAIPriorities; 
      onDownloadProgress?: (progress: ProgressType) => void;
      callbackThrottle?: number;
      worker: Worker | null;
      isWebGPUAvailable: boolean | null;
      storageQuota: StorageEstimate | null;
    }): Promise<boolean> {
      // Ensure model support information is available
      if (!this._modelSupportedModes || !this._supportedPrecisionsDevicesMap ||
          !this._modelSupportedPrecisions) {
        throw createWorkerError(
          "Model support information is not available. Make sure to create the instance properly.",
          'initialization_failed'
        );
      }
  
      // Handle explicit mode (not auto)
      if (mode !== "auto") {
        this.validateInitConfig(mode, precision, device);
  
        if (mode === "cloud") {
          this._mode = "cloud";
          return true;
        }
  
        // For webai mode
        if (mode === "webai" && device && precision) {
          this.validateStorageAndWebGPU(device, precision, isWebGPUAvailable, storageQuota);
          this._mode = "webai";
  
          const result = await this.#initWebAI(
            worker, 
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
  
        throw createWorkerError(`Unsupported mode: ${mode}`, 'initialization_failed');
      }
  
      // Auto mode - try configurations in priority order
      const defaultPriorities = this.getDefaultPriorities();
      const priorityList = priorities?.length ? priorities : defaultPriorities;
  
      for (const config of priorityList) {
        if (await this.tryConfiguration(
          config, 
          worker, 
          isWebGPUAvailable, 
          storageQuota, 
          onDownloadProgress, 
          callbackThrottle
        )) {
          return true;
        }
      }
  
      // No configuration worked
      throw createWorkerError(
        "Could not run the model, none of the configurations (mode, precision, device) " +
        "specified in the priorities list works. Please verify the model requirements and the device you are running on.",
        'initialization_failed'
      );
    }
  
    private getDefaultPriorities() {
      return [
        { mode: "webai", precision: "q4", device: "webgpu" },
        { mode: "webai", precision: "q4f16", device: "webgpu" },
        { mode: "webai", precision: "bnb4", device: "webgpu" },
        { mode: "webai", precision: "q8", device: "webgpu" },
        { mode: "webai", precision: "int8", device: "webgpu" },
        { mode: "webai", precision: "uint8", device: "webgpu" },
        { mode: "webai", precision: "fp16", device: "webgpu" },
        { mode: "webai", precision: "fp32", device: "webgpu" },
        { mode: "webai", precision: "q4", device: "wasm" },
        { mode: "webai", precision: "q4f16", device: "wasm" },
        { mode: "webai", precision: "bnb4", device: "wasm" },
        { mode: "webai", precision: "q8", device: "wasm" },
        { mode: "webai", precision: "int8", device: "wasm" },
        { mode: "webai", precision: "uint8", device: "wasm" },
        { mode: "webai", precision: "fp16", device: "wasm" },
        { mode: "webai", precision: "fp32", device: "wasm" },
        { mode: "cloud", precision: "", device: "" }
      ];
    }
  
    private async tryConfiguration(
      config: any,
      worker: Worker | null,
      isWebGPUAvailable: boolean | null,
      storageQuota: StorageEstimate | null,
      onDownloadProgress?: (progress: ProgressType) => void,
      callbackThrottle?: number
    ): Promise<boolean> {
      const currentMode = config.mode as WebAIMode;
  
      // Skip unsupported modes
      if (!this._modelSupportedModes?.includes(currentMode)) {
        return false;
      }
  
      if (currentMode === "cloud") {
        this._mode = "cloud";
        return true;
      }
  
      if (currentMode === "webai") {
        const precision = config.precision as WebAIPrecision;
        const device = config.device as WebAIDevice;
  
        // Validate precision support
        if (!this._modelSupportedPrecisions?.includes(precision)) {
          return false;
        }
  
        // Validate device support for this precision
        const supportedDevices = 
          this._supportedPrecisionsDevicesMap?.[precision]?.supportedDevices || [];
        if (!supportedDevices.includes(device)) {
          return false;
        }
  
        // Check WebGPU availability
        if (device === "webgpu" && !isWebGPUAvailable) {
          return false;
        }
  
        // Check storage
        const modelSize = this._supportedPrecisionsDevicesMap?.[precision]?.size;
        if (!modelSize) {
          return false;
        }
  
        const leftStorage = (storageQuota?.quota ?? 0) - (storageQuota?.usage ?? 0);
        if (modelSize > leftStorage) {
          return false;
        }
  
        // Try to initialize with this configuration
        try {
          console.log("Trying configuration:", precision, device);
  
          const result = await this.#initWebAI(
            worker,
            precision,
            device,
            onDownloadProgress,
            callbackThrottle
          );
  
          if (result) {
            this._mode = "webai";
            return true;
          }
        } catch (error) {
          console.warn(`Failed to initialize with ${precision}/${device}:`, error);
        }
      }
  
      return false;
    }
  
    async #initWebAI(
      worker: Worker | null,
      precision: WebAIPrecision,
      device: WebAIDevice,
      onDownloadProgress?: (progress: ProgressType) => void,
      callbackThrottle: number = 1000
    ): Promise<boolean> {
      const modelKeys = this._supportedPrecisionsDevicesMap?.[precision]?.modelKeys || [];
      
      // Check if model is downloaded
      const isModelDownloaded = await checkIsModelDownloaded(precision, modelKeys, this._modelId);
      
      if (isModelDownloaded) {
        // Model exists - just initialize
        await this.sendInitMessage(worker, precision, device);
        return true;
      }
      
      // Model not downloaded - download it
      const downloadSuccess = await this.downloadModel(
        worker,
        precision,
        onDownloadProgress,
        callbackThrottle
      );
      
      if (!downloadSuccess) {
        throw createWorkerError("Failed to download model", 'initialization_failed');
      }
      
      // Worker has already been reloaded in downloadModel
      const freshWorker = this._workerManager?.worker;
      if (!freshWorker) {
        throw createWorkerError(
          "Worker is null after download and reload",
          'initialization_failed'
        );
      }
  
      // Initialize with the fresh worker
      await this.sendInitMessage(freshWorker, precision, device);
      return true;
    }
  
    async reinitializeAfterInterrupt(worker: Worker | null): Promise<void> {
      if (!worker || !this._mode || !this._precision || !this._device) {
        throw createWorkerError(
          "Cannot reinitialize: missing required state (mode, precision, or device)",
          'initialization_failed'
        );
      }
  
      if (this._mode !== "webai") {
        return; // Cloud mode doesn't need reinitialization
      }
  
      try {
        // Verify model is still downloaded
        const modelKeys = this._supportedPrecisionsDevicesMap?.[this._precision]?.modelKeys;
        if (!modelKeys) {
          throw createWorkerError(
            `Model does not support precision ${this._precision}`,
            'initialization_failed'
          );
        }
  
        const isDownloaded = await checkIsModelDownloaded(
          this._precision, 
          modelKeys, 
          this._modelId
        );
        
        if (!isDownloaded) {
          throw createWorkerError(
            "Model is no longer downloaded - full reinitialization required",
            'initialization_failed'
          );
        }
  
        // Re-check model supports and initialize
        await this.checkModelSupports(worker);
        await this.sendInitMessage(worker, this._precision, this._device);
  
        console.log(`Model reinitialized after interrupt: ${this._precision}/${this._device}`);
      } catch (error) {
        console.error("Failed to reinitialize model after interrupt:", error);
        
        // Reset state on failure
        this._mode = null;
        this._precision = null;
        this._device = null;
        
        throw error;
      }
    }
  
    async clearCache(): Promise<void> {
      return clearModelCache(this._modelId);
    }
  
    async deletePrecisionSpecificModel(precision: WebAIPrecision): Promise<void> {
      if (!this._supportedPrecisionsDevicesMap?.[precision]) {
        console.warn(`Precision ${precision} is not supported for model ${this._modelId}`);
        return;
      }
  
      const modelKeys = this._supportedPrecisionsDevicesMap[precision].modelKeys;
      
      // Check if other precisions are downloaded
      const downloadedPrecisions: WebAIPrecision[] = [];
      
      if (this._modelSupportedPrecisions) {
        for (const supportedPrecision of this._modelSupportedPrecisions) {
          if (supportedPrecision === precision) continue;
          
          const otherModelKeys = this._supportedPrecisionsDevicesMap?.[supportedPrecision]?.modelKeys;
          if (otherModelKeys) {
            const isDownloaded = await checkIsModelDownloaded(
              supportedPrecision,
              otherModelKeys,
              this._modelId
            );
            if (isDownloaded) {
              downloadedPrecisions.push(supportedPrecision);
            }
          }
        }
      }
  
      if (downloadedPrecisions.length === 0) {
        // Only precision - delete everything
        console.log(`Deleting all cached files for model ${this._modelId} (only precision: ${precision})`);
        await this.clearCache();
        return;
      }
  
      // Other precisions exist - delete only this precision
      console.log(`Deleting precision-specific files for ${this._modelId}, precision: ${precision}`);
      console.log(`Keeping other precisions: ${downloadedPrecisions.join(', ')}`);
      
      const cache = await caches.open("transformers-cache");
      
      for (const modelKey of modelKeys) {
        try {
          const cacheKeys = await cache.keys();
          for (const request of cacheKeys) {
            const url = request.url;
            if (url.includes(this._modelId) && url.includes(modelKey)) {
              await cache.delete(request);
            }
          }
        } catch (error) {
          console.warn(`Failed to delete cached files for model key ${modelKey}:`, error);
        }
      }
      
      console.log(`Successfully deleted model files for precision: ${precision}`);
    }
  
    // Getters
    get mode(): WebAIMode | null {
      return this._mode;
    }
  
    get precision(): WebAIPrecision | null {
      return this._precision;
    }
  
    get device(): WebAIDevice | null {
      return this._device;
    }
  
    get modelSupportedModes(): WebAIMode[] | null {
      return this._modelSupportedModes;
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
  }