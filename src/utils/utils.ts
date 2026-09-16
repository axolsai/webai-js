import { createError, ErrorCategory } from "./errors";

export async function checkIsWebGPUAvailable(): Promise<boolean> {
    try {
      if (typeof navigator === "undefined" || !navigator.gpu) {
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
      if (typeof navigator !== "undefined" && navigator.storage?.estimate) {
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
    try {
      if (typeof caches === "undefined") return false;
      const cache = await caches.open("transformers-cache");
      const keys = (await cache.keys()).map(key => decodeURIComponent(key.url));
      const modelPath = `/${modelId}/`;
      return modelKeys.every(modelKey =>
        keys.some(key => key.includes(modelKey) && key.includes(modelPath))
      );
    } catch {
      return false;
    }
}



export async function clearModelCache(modelId: string): Promise<void> {
    if (!modelId) {
      throw createError(
        "Model ID is required to clear cache",
        ErrorCategory.RESOURCE,
        'missing_parameter'
      );
    }
  
    try {
      if (typeof caches === "undefined") return;
      const cache = await caches.open("transformers-cache");
      const cacheKeys = await cache.keys();
      const modelPath = `/${modelId}/`;
      const modelRequests = cacheKeys.filter(key =>
        decodeURIComponent(key.url).includes(modelPath)
      );
      const results = await Promise.allSettled(modelRequests.map(key => cache.delete(key)));
      const failed = results.filter(result => result.status === "rejected");
      if (failed.length > 0) throw new Error(`Failed to delete ${failed.length} cache entries`);
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
    if (typeof location !== "undefined") {
      try {
        const url = new URL(path, location.href);
        return url.origin === location.origin || url.protocol === "blob:" || url.protocol === "data:";
      } catch {
        return true;
      }
    }
    return !/^https?:\/\//i.test(path);
  }
