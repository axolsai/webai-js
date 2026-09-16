import { createWorkerError, isWorkerError } from "../utils/errors";
import { isLocalPath } from "../utils/utils";

type PendingRequest = {
  expectedType: string;
  resolve: (value: unknown) => void;
  reject: (reason: Error) => void;
  onProgress?: (data: unknown) => void;
  timeout?: ReturnType<typeof setTimeout>;
  removeAbortListener?: () => void;
};

export type WorkerRequestOptions = {
  expectedType?: string;
  timeoutMs?: number;
  signal?: AbortSignal;
  onProgress?: (data: unknown) => void;
};

export class WorkerManager {
  private _worker: Worker | null = null;
  private readonly _workerPath: string;
  private readonly _initTimeoutMs: number;
  private _cachedWorkerScript: string | null = null;
  private _objectUrl: string | null = null;
  private _requestCounter = 0;
  private readonly _pending = new Map<string, PendingRequest>();

  constructor(workerPath: string, initTimeoutMs: number) {
    this._workerPath = workerPath;
    this._initTimeoutMs = initTimeoutMs;
  }

  async initWorker(): Promise<void> {
    if (this._worker) return;
    try {
      this._worker = await this.createWorker();
      await this.waitForBoot(this._worker);
      this._worker.addEventListener("message", this.handleMessage);
      this._worker.addEventListener("error", this.handleRuntimeError);
      this._worker.addEventListener("messageerror", this.handleMessageError);
    } catch (error) {
      this.disposeWorker();
      if (isWorkerError(error)) throw error;
      throw createWorkerError(
        error instanceof Error
          ? `Failed to initialize worker: ${error.message}`
          : "Failed to initialize worker: Unknown error",
        "initialization_failed",
      );
    }
  }

  private async createWorker(): Promise<Worker> {
    if (isLocalPath(this._workerPath)) {
      return new Worker(this._workerPath, { type: "module" });
    }

    if (!this._cachedWorkerScript) {
      const response = await fetch(this._workerPath, { cache: "default", redirect: "follow" });
      if (!response.ok) {
        throw createWorkerError(
          `Failed to fetch worker script: ${response.status} ${response.statusText}`,
          "initialization_failed",
        );
      }
      this._cachedWorkerScript = await response.text();
      if (!this._cachedWorkerScript.trim()) {
        throw createWorkerError("Worker script was empty", "initialization_failed");
      }
    }

    this._objectUrl = URL.createObjectURL(
      new Blob([this._cachedWorkerScript], { type: "text/javascript" }),
    );
    return new Worker(this._objectUrl, { type: "module" });
  }

  private waitForBoot(worker: Worker): Promise<void> {
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        cleanup();
        reject(createWorkerError(
          `Worker did not initialize within ${this._initTimeoutMs}ms`,
          "worker_timeout",
        ));
      }, this._initTimeoutMs);
      const cleanup = () => {
        clearTimeout(timeout);
        worker.removeEventListener("message", onMessage);
        worker.removeEventListener("error", onError);
        worker.removeEventListener("messageerror", onMessageError);
      };
      const onMessage = (event: MessageEvent) => {
        if (event.data?.type === "worker initialized" && event.data?.data?.success) {
          cleanup();
          resolve();
        } else if (event.data?.type === "error") {
          cleanup();
          reject(this.parseWorkerError(event.data.data));
        }
      };
      const onError = (event: ErrorEvent) => {
        cleanup();
        reject(createWorkerError(event.message || "Worker boot failed", "runtime_error"));
      };
      const onMessageError = () => {
        cleanup();
        reject(createWorkerError("Failed to deserialize worker boot message", "message_error"));
      };
      worker.addEventListener("message", onMessage);
      worker.addEventListener("error", onError);
      worker.addEventListener("messageerror", onMessageError);
    });
  }

  request<T>(type: string, data?: unknown, options: WorkerRequestOptions = {}): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      if (!this._worker) {
        reject(createWorkerError("Worker not initialized", "initialization_failed"));
        return;
      }
      if (options.signal?.aborted) {
        reject(createWorkerError(`Worker operation '${type}' was aborted`, "interrupted"));
        return;
      }

      const requestId = this.createRequestId();
      const pending: PendingRequest = {
        expectedType: options.expectedType ?? type,
        resolve: resolve as (value: unknown) => void,
        reject,
        onProgress: options.onProgress,
      };
      if (options.timeoutMs && options.timeoutMs > 0) {
        pending.timeout = setTimeout(() => {
          this.rejectRequest(requestId, createWorkerError(
            `Worker operation '${type}' timed out after ${options.timeoutMs}ms`,
            "worker_timeout",
          ));
        }, options.timeoutMs);
      }
      if (options.signal) {
        const onAbort = () => this.rejectRequest(
          requestId,
          createWorkerError(`Worker operation '${type}' was aborted`, "interrupted"),
        );
        options.signal.addEventListener("abort", onAbort, { once: true });
        pending.removeAbortListener = () => options.signal?.removeEventListener("abort", onAbort);
      }

      this._pending.set(requestId, pending);
      try {
        this._worker.postMessage({ requestId, type, data });
      } catch (error) {
        this.rejectRequest(requestId, createWorkerError(
          error instanceof Error ? error.message : "Failed to post worker message",
          "message_error",
        ));
      }
    });
  }

  private readonly handleMessage = (event: MessageEvent): void => {
    const { requestId, type, data } = event.data ?? {};
    if (!requestId) return;
    const pending = this._pending.get(requestId);
    if (!pending) return;
    if (type === "downloadProgress" || type === "streamChunk") {
      pending.onProgress?.(data);
    } else if (type === "error") {
      this.rejectRequest(requestId, this.parseWorkerError(data));
    } else if (type === pending.expectedType) {
      this._pending.delete(requestId);
      this.cleanupPending(pending);
      pending.resolve(data);
    }
  };

  private readonly handleRuntimeError = (event: ErrorEvent): void => {
    this.disposeWorker(
      createWorkerError(event.message || "Worker runtime error", "runtime_error"),
    );
  };

  private readonly handleMessageError = (): void => {
    this.disposeWorker(
      createWorkerError("Failed to deserialize worker message", "message_error"),
    );
  };

  private createRequestId(): string {
    if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
      return crypto.randomUUID();
    }
    this._requestCounter += 1;
    return `webai-${Date.now()}-${this._requestCounter}`;
  }

  private parseWorkerError(errorData: unknown): Error {
    const value = errorData as { message?: string; context?: string } | undefined;
    return createWorkerError(
      `${value?.context ?? "worker"}: ${value?.message ?? "Unknown worker error"}`,
      "runtime_error",
    );
  }

  private cleanupPending(pending: PendingRequest): void {
    if (pending.timeout) clearTimeout(pending.timeout);
    pending.removeAbortListener?.();
  }

  private rejectRequest(requestId: string, error: Error): void {
    const pending = this._pending.get(requestId);
    if (!pending) return;
    this._pending.delete(requestId);
    this.cleanupPending(pending);
    pending.reject(error);
  }

  private rejectAll(error: Error): void {
    for (const requestId of [...this._pending.keys()]) this.rejectRequest(requestId, error);
  }

  async restart(): Promise<void> {
    this.disposeWorker(createWorkerError("Worker restarted", "interrupted"));
    await this.initWorker();
  }

  terminateWorker(): void {
    this.disposeWorker(createWorkerError("Worker terminated", "worker_terminated"));
  }

  async clearMemory(): Promise<void> {
    await this.request("clearMemory", undefined, { timeoutMs: this._initTimeoutMs });
  }

  clearScriptCache(): void {
    this._cachedWorkerScript = null;
  }

  private disposeWorker(reason?: Error): void {
    if (reason) this.rejectAll(reason);
    if (this._worker) {
      this._worker.removeEventListener("message", this.handleMessage);
      this._worker.removeEventListener("error", this.handleRuntimeError);
      this._worker.removeEventListener("messageerror", this.handleMessageError);
      this._worker.terminate();
      this._worker = null;
    }
    if (this._objectUrl) {
      URL.revokeObjectURL(this._objectUrl);
      this._objectUrl = null;
    }
  }

  get worker(): Worker | null { return this._worker; }
  get isWorkerActive(): boolean { return this._worker !== null; }
}
