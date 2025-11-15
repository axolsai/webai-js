import { createError, ErrorCategory } from "./errors";

declare global {
    interface Navigator {
      gpu?: {
        requestAdapter(): Promise<GPUAdapter | null>;
      };
    }
  }
  
export async function checkIsWebGPUAvailable(): Promise<boolean> {
    try {
      if (!navigator || !navigator.gpu) {
        return false;
      }
      const adapter = await navigator.gpu.requestAdapter();
      return adapter !== null;
    } catch (error) {
      console.log("Error checking WebGPU availability:", error);
      return false;
    }
  }
  

export async function checkStorageQuota() {
    try {
      if (navigator.storage && navigator.storage.estimate) {
        const estimate = await navigator.storage.estimate();
        return estimate;
      } else {
        console.log("Storage API not supported in this browser");
        return null;
      }
    } catch (error) {
      console.error("Error checking storage quota:", error);
      return null;
    }
  }


export async function checkIsModelDownloaded(precision: string, modelKeys: string[], modelId: string) {
    const cache = await caches.open("transformers-cache");
    const keys = (await cache.keys()).map(key => key.url);
    if (keys.length === 0) {
        return false;
    }
    const isDownloaded = modelKeys.every(modelKey => 
        keys.some(key => key.includes(modelKey) && key.includes("/"+modelId+"/"))
    );
    if (!isDownloaded) {
        return false;
    }
    // const isFingerprintValid = await verifyFingerprint(modelId, precision);
    // if (!isFingerprintValid) {
    //     console.warn(`Fingerprint verification failed for model ${modelId} with precision ${precision}`);
    //     return false;
    // }
    return isDownloaded;
}



export async function clearModelCache(modelId: string, modelKeys?:string[]): Promise<void> {
    if (!modelId) {
      throw createError(
        "Model ID is required to clear cache",
        ErrorCategory.RESOURCE,
        'missing_parameter'
      );
    }
  
    try {
      const cache = await caches.open("transformers-cache");
      const cacheKeys = await cache.keys();
      const keys = cacheKeys.map(key => key.url);
      const allModelKeys = keys.filter(key => key.includes("/"+modelId+"/"));
      await Promise.allSettled(allModelKeys.map(key => cache.delete(new Request(key))));
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : "Unknown error";
      throw createError(
        `Failed to clear model cache: ${errorMessage}`,
        ErrorCategory.RESOURCE,
        'cache_operation_failed'
      );
    }
  
}

export function isLocalPath(path: string): boolean {
    // Check for URL protocols (http://, https://, file://, etc.)
    if (path.includes('://')) {
      return false;
    }
    
    // Check for relative paths (./ or ../)
    if (path.startsWith('./') || path.startsWith('../')) {
      return true;
    }
    
    // Check for absolute Unix paths (/)
    if (path.startsWith('/')) {
      return true;
    }
    
    // Check for Windows absolute paths (C:\, D:\, etc.)
    if (/^[a-zA-Z]:[/\\]/.test(path)) {
      return true;
    }
    
    // Check for Windows network paths (\\server\share)
    if (path.startsWith('\\\\')) {
      return true;
    }
    
    // If none of the above, treat as local path
    return true;
  }