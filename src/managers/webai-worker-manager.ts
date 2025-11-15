// src/worker/WorkerManager.ts (Open Source Version)
import { createWorkerError, isWorkerError } from "../utils/errors";
import { isLocalPath } from "../utils/utils";

export class WorkerManager {
  private _worker: Worker | null = null;
  private _workerPath: string;
  private _modelId: string;
  private _cachedWorkerScript: string | null = null;

  constructor(workerPath: string, modelId: string) {
    this._workerPath = workerPath;
    this._modelId = modelId;
  }

  
  async initWorker(): Promise<void> {
    try {
      // Only fetch if we don't have cached script
      if (!this._cachedWorkerScript) {
        // Check if it's a local path (for direct Worker instantiation) or a URL (for fetch)
        if (isLocalPath(this._workerPath)) {
          // For local paths, create worker directly without fetching
          this._worker = new Worker(this._workerPath, {
            type: "module",
          });

          if (!this._worker) {
            throw createWorkerError(
              "Failed to initialize WebAI worker from local path.",
              'initialization_failed'
            );
          }

          await this.setupWorkerListeners();
          return;
        }

        // For remote URLs, fetch and cache the script
        const workerPath = this._workerPath + "?t=" + Date.now();
        const workerFetchResponse = await fetch(workerPath, {
          cache: "no-store",
        });

        if (!workerFetchResponse.ok) {
          throw createWorkerError(
            `Failed to fetch worker script: ${workerFetchResponse.status} ${workerFetchResponse.statusText}`,
            'initialization_failed'
          );
        }

        // Directly use the script (no decryption needed)
        this._cachedWorkerScript = await workerFetchResponse.text();
        
        if (!this._cachedWorkerScript) {
          throw new Error('Failed to load worker script');
        }
      }

      // Create worker from cached script (for remote URLs)
      const workerBlob = new Blob([this._cachedWorkerScript], {
        type: "application/javascript",
      });
      const objectURL = URL.createObjectURL(workerBlob);
      this._worker = new Worker(objectURL, {
        type: "module",
      });
      URL.revokeObjectURL(objectURL);

      if (!this._worker) {
        throw createWorkerError(
          "Failed to initialize WebAI worker.",
          'initialization_failed'
        );
      }

      await this.setupWorkerListeners();
      
    } catch (error) {
      if (this._worker) {
        this._worker.terminate();
        this._worker = null;
      }
      
      if (isWorkerError(error)) {
        throw error;
      } else {
        if (error instanceof Error) {
          throw createWorkerError(
            `Failed to initialize worker: ${error.message}`,
            'initialization_failed'
          );
        } else {
          throw createWorkerError(
            "Failed to initialize worker: Unknown error",
            'initialization_failed'
          );
        }
      }
    }
  }

  async reloadWorker(): Promise<void> {
    console.log("Terminating existing worker...");
    
    if (this._worker) {
      this._worker.terminate();
      this._worker = null;
    }
    
    // Add a small delay to ensure the worker is fully terminated
    await new Promise(resolve => setTimeout(resolve, 100));
    
    console.log("Initializing new worker...");
    await this.initWorker();
    
    if (!this._worker) {
      throw createWorkerError(
        "Failed to create new worker instance after reload",
        'initialization_failed'
      );
    }
    
    console.log("Worker successfully reloaded");
  }

  private async setupWorkerListeners(): Promise<void> {
    return new Promise<void>((resolve, reject) => {
      const handleError = (event: ErrorEvent) => {
        this._worker?.removeEventListener("error", handleError);
        this._worker?.removeEventListener("messageerror", handleMessageError);
        this._worker?.removeEventListener("message", handleMessage);
        
        reject(createWorkerError(
          `WebAI Worker runtime error: ${event.message || "Unknown error"}`,
          'runtime_error'
        ));
      };

      const handleMessageError = (event: MessageEvent) => {
        this._worker?.removeEventListener("error", handleError);
        this._worker?.removeEventListener("messageerror", handleMessageError);
        this._worker?.removeEventListener("message", handleMessage);
        
        reject(createWorkerError(
          "Failed to deserialize message from worker",
          'message_error'
        ));
      };

      const handleMessage = (event: MessageEvent) => {
        // Check for error message from worker
        if (event.data?.type === "error") {
          this._worker?.removeEventListener("error", handleError);
          this._worker?.removeEventListener("messageerror", handleMessageError);
          this._worker?.removeEventListener("message", handleMessage);
          
          const workerError = this.parseWorkerError(event.data.data);
          console.log(`WebAI Worker initialization error: ${workerError.message}`);
          reject(workerError);
          return;
        }

        // Check for successful initialization
        if (event.data?.type === "worker initialized" && event.data.data.success) {
          console.log(`WebAI worker successfully initialized!`);
          this._worker?.removeEventListener("error", handleError);
          this._worker?.removeEventListener("messageerror", handleMessageError);
          this._worker?.removeEventListener("message", handleMessage);
          
          resolve();
        }
      };

      this._worker?.addEventListener("error", handleError);
      this._worker?.addEventListener("messageerror", handleMessageError);
      this._worker?.addEventListener("message", handleMessage);
    });
  }

  /**
   * Parse worker error data and create appropriate error with context
   */
  private parseWorkerError(errorData: any): Error {
    const errorMessage = errorData?.message || "Unknown worker error";
    const errorContext = errorData?.context || "unknown";
    
    return createWorkerError(
      `${errorContext}: ${errorMessage}`,
      'runtime_error'
    );
  }

  /**
   * Creates a message handler function that handles both success and error cases
   */
  createMessageHandler(
    expectedType: string,
    onSuccess: (data: any) => void,
    onError: (error: Error) => void,
    cleanup: () => void
  ): (event: MessageEvent) => void {
    return (event: MessageEvent) => {
      if (event.data.type === "error") {
        cleanup();
        const workerError = this.parseWorkerError(event.data.data);
        console.log(`WebAI Worker error: ${workerError.message}`);
        onError(workerError);
        return;
      }
      
      if (event.data.type === expectedType) {
        cleanup();
        onSuccess(event.data.data);
      }
    };
  }

  /**
   * Creates a promise-based worker operation handler
   */
  createWorkerPromise<T>(
    messageType: string, 
    messageData: any,
    expectedResponseType: string
  ): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      if (!this._worker) {
        reject(createWorkerError(
          "Worker not initialized",
          'initialization_failed'
        ));
        return;
      }

      const cleanup = () => {
        this._worker?.removeEventListener("message", handler);
      };

      const handler = this.createMessageHandler(
        expectedResponseType,
        (data: T) => resolve(data),
        (error: Error) => reject(error),
        cleanup
      );

      this._worker.addEventListener("message", handler);
      this._worker.postMessage({ type: messageType, data: messageData });
    });
  }

  /**
   * Helper method for operations that expect generic success response
   */
  async sendWorkerMessage(messageType: string, messageData?: any): Promise<any> {
    return this.createWorkerPromise(
      messageType,
      messageData,
      messageType // Expect response type to match message type
    );
  }

  terminateWorker(): void {
    if (this._worker) {
      this._worker.terminate();
      this._worker = null;
      console.log("WebAI worker terminated successfully.");
    }
  }

  interruptWorker(): void {
    if (this._worker) {
      this._worker.postMessage({ type: "interrupt" });
    }
  }

  async terminateAndReload(): Promise<void> {
    if (this._worker) {
      this._worker.terminate();
    }
    this._worker = null;
    await this.initWorker();
  }

  get isWorkerActive(): boolean {
    return this._worker !== null;
  }

  clearMemory(): void {
    if (this._worker) {
      this._worker.postMessage({ type: "clearMemory" });
    }
  }

  /**
   * Clear the cached script (useful if you need to force re-fetch)
   */
  clearCache(): void {
    this._cachedWorkerScript = null;
  }

  get worker(): Worker | null {
    return this._worker;
  }
}