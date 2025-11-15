// src/queue/QueueManager.ts
import { WebAIMode } from "../utils/types";
import { createWorkerError } from "../utils/errors";

// Define types for the queue items
interface QueueItem {
  type: 'generate' | 'generateStream';
  data: any;
  resolve: (value: any) => void;
  reject: (reason: any) => void;
}

interface ExecutionContext {
  data: any;
  resolve: (value: any) => void;
  reject: (reason: any) => void;
  worker: Worker | null;
  mode: WebAIMode | null;
  isStream: boolean;
  onStream?: (chunk: any) => void;
}

export class QueueManager {
  private _isGenerating: boolean = false;
  private _queue: QueueItem[] = [];
  private _currentExecution: ExecutionContext | null = null; // Track current execution
  
  constructor() {}
  
  async enqueueGenerate(
    data: { userInput: any, modelConfig?: object, generateConfig?: object },
    worker: Worker | null,
    mode: WebAIMode | null
  ): Promise<any> {
    return new Promise((resolve, reject) => {
      this._queue.push({
        type: 'generate',
        data,
        resolve,
        reject
      });

      if (!this._isGenerating) {
        this.processNextInQueue(worker, mode);
      }
    });
  }
  
  async enqueueGenerateStream(
    data: {
      userInput: any,
      modelConfig?: object,
      generateConfig?: object,
      onStream: (chunk: any) => void
    },
    worker: Worker | null,
    mode: WebAIMode | null
  ): Promise<void> {
    return new Promise((resolve, reject) => {
      this._queue.push({
        type: 'generateStream',
        data,
        resolve,
        reject
      });

      if (!this._isGenerating) {
        this.processNextInQueue(worker, mode);
      }
    });
  }

  /**
   * Handle forced interruption (for WASM worker reload)
   * This resolves the current generation with an interruption result
   */
  handleForcedInterruption(): void {
    if (this._currentExecution && this._isGenerating) {
      console.log("WebAI generation is manually interrupted (forced).");
      // Resolve current execution as interrupted (same as WebGPU interrupt response)
      this._currentExecution.resolve(undefined);
      this._currentExecution = null;
    }
    this._isGenerating = false;
  }

  // Process the next item in the queue
  processNextInQueue(worker: Worker | null, mode: WebAIMode | null): void {
    if (this._queue.length === 0) {
      this._isGenerating = false;
      this._currentExecution = null;
      return;
    }

    this._isGenerating = true;
    const nextItem = this._queue.shift()!;

    const isStream = nextItem.type === 'generateStream';
    const { onStream, ...dataWithoutOnStream } = nextItem.data;

    const context: ExecutionContext = {
      data: isStream ? dataWithoutOnStream : nextItem.data,
      resolve: nextItem.resolve,
      reject: nextItem.reject,
      worker,
      mode,
      isStream,
      onStream: isStream ? onStream : undefined
    };

    this._executeGeneration(context);
  }

  // Unified execution method for both generate and generateStream
  private _executeGeneration(context: ExecutionContext): void {
    const { data, resolve, reject, worker, mode, isStream, onStream } = context;

    // Store current execution context for potential forced interruption
    this._currentExecution = context;

    // Validation
    if (!worker) {
      reject(createWorkerError(
        "Worker is not initialized",
        'initialization_failed'
      ));
      this._currentExecution = null;
      this.processNextInQueue(worker, mode);
      return;
    }

    if (!mode) {
      reject(createWorkerError(
        "WebAI is not initialized. Call init() before generate()",
        'initialization_failed'
      ));
      this._currentExecution = null;
      this.processNextInQueue(worker, mode);
      return;
    }

    // Setup cleanup and event handlers
    const cleanup = () => {
      worker?.removeEventListener("message", handleMessage);
      worker?.removeEventListener("error", handleError);
      this._currentExecution = null; // Clear current execution
    };

    const handleMessage = (event: MessageEvent) => {
      // Handle error messages first
      if (event.data.type === "error") {
        const errorMessage = isStream 
          ? "Stream generation error" 
          : "Generation error";
        console.log(`WebAI ${isStream ? 'stream ' : ''}generation error:`, event.data.data.message);
        
        cleanup();
        reject(createWorkerError(
          event.data.data.message || errorMessage,
          'runtime_error'
        ));
        this.processNextInQueue(worker, mode);
        return;
      }

      if (event.data.type === "interrupted") {
        console.log(`WebAI generation is manually interrupted.`);
        cleanup();
        resolve(undefined); // Consistent interrupt response
        this.processNextInQueue(worker, mode);
        return;
      }

      // Handle streaming data (only for generateStream)
      if (event.data.type === "stream" && isStream && onStream) {
        onStream(event.data.data);
        return;
      }

      // Handle completion
      if (event.data.type === "generated") {
        if (isStream) {
          console.log("stream ****", event.data.data.result);
        }
        cleanup();
        resolve(event.data.data.result);
        this.processNextInQueue(worker, mode);
        return;
      } 
      
      if (event.data.type === "generateError") {
        cleanup();
        reject(createWorkerError(
          `Generation error: ${event.data.error}`,
          'runtime_error'
        ));
        this.processNextInQueue(worker, mode);
        return;
      }
    };

    const handleError = (error: ErrorEvent) => {
      const errorMessage = isStream 
        ? `Worker error during stream generation: ${error.message}`
        : `Worker error during generation: ${error.message}`;
      
      cleanup();
      reject(createWorkerError(errorMessage, 'runtime_error'));
      this.processNextInQueue(worker, mode);
    };

    // Attach event listeners
    worker.addEventListener("message", handleMessage);
    worker.addEventListener("error", handleError);

    // Send the message
    try {
      worker.postMessage({
        type: isStream ? "generateStream" : "generate",
        data
      });
    } catch (error) {
      const errorMessage = isStream 
        ? "Failed to send stream generation request"
        : "Failed to send generation request";
      
      cleanup();
      reject(createWorkerError(
        error instanceof Error
          ? `${errorMessage}: ${error.message}`
          : `${errorMessage}: Unknown error`,
        'runtime_error'
      ));
      this.processNextInQueue(worker, mode);
    }
  }
  
  // Get the current queue size
  get queueSize(): number {
    return this._queue.length;
  }

  // Check if generation is in progress
  get isGenerating(): boolean {
    return this._isGenerating;
  }

  // Check if there's a current execution (useful for debugging)
  get hasCurrentExecution(): boolean {
    return this._currentExecution !== null;
  }

  // Get current execution type (useful for debugging)
  get currentExecutionType(): 'generate' | 'generateStream' | null {
    if (!this._currentExecution) return null;
    return this._currentExecution.isStream ? 'generateStream' : 'generate';
  }

  // Clear the queue
  clearQueue(): void {
    // Only clear pending items, don't affect current execution
    this._queue = [];
  }

  // Force clear everything including current execution (emergency use)
  forceReset(): void {
    this._queue = [];
    this._currentExecution = null;
    this._isGenerating = false;
  }
}