import { createWorkerError } from "../utils/errors";
import type { WebAIResult } from "../utils/types";
import { WorkerManager } from "./webai-worker-manager";

type QueueItem = {
  type: "generate" | "generateStream";
  data: unknown;
  onStream?: (chunk: unknown) => void;
  resolve: (value: unknown) => void;
  reject: (reason: Error) => void;
};

export class QueueManager {
  private readonly _workerManager: WorkerManager;
  private readonly _queue: QueueItem[] = [];
  private _current: QueueItem | null = null;
  private _currentAbortController: AbortController | null = null;

  constructor(workerManager: WorkerManager) {
    this._workerManager = workerManager;
  }

  enqueueGenerate<T = unknown>(data: unknown): Promise<WebAIResult<T>> {
    return this.enqueue({ type: "generate", data }) as Promise<WebAIResult<T>>;
  }

  enqueueGenerateStream(
    data: unknown,
    onStream: (chunk: unknown) => void,
  ): Promise<void> {
    return this.enqueue({ type: "generateStream", data, onStream }) as Promise<void>;
  }

  private enqueue(item: Pick<QueueItem, "type" | "data" | "onStream">): Promise<unknown> {
    return new Promise((resolve, reject) => {
      this._queue.push({ ...item, resolve, reject });
      void this.processNext();
    });
  }

  private async processNext(): Promise<void> {
    if (this._current || this._queue.length === 0) return;
    const item = this._queue.shift();
    if (!item) return;
    this._current = item;
    this._currentAbortController = new AbortController();
    const startedAt = performance.now();

    try {
      const data = await this._workerManager.request<{
        status: string;
        result?: unknown;
      }>(item.type, item.data, {
        expectedType: "generated",
        signal: this._currentAbortController.signal,
        onProgress: item.onStream,
      });
      if (data.status !== "success") {
        throw createWorkerError("Generation failed", "runtime_error");
      }
      item.resolve({
        result: data.result,
        runtime: { durationMs: performance.now() - startedAt },
      });
    } catch (error) {
      item.reject(error instanceof Error
        ? error
        : createWorkerError("Unknown generation failure", "runtime_error"));
    } finally {
      this._current = null;
      this._currentAbortController = null;
      void this.processNext();
    }
  }

  clearQueue(): void {
    for (const item of this._queue.splice(0)) {
      item.reject(createWorkerError("Generation request cancelled", "interrupted"));
    }
  }

  cancelCurrent(): void {
    this._currentAbortController?.abort();
  }

  cancelAll(): void {
    this.clearQueue();
    this.cancelCurrent();
  }

  get isGenerating(): boolean { return this._current !== null; }
  get queueLength(): number { return this._queue.length; }
}
