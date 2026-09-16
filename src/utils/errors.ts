// Error category enum can still be useful for consistency
export enum ErrorCategory {
    VALIDATION = 'ValidationError',
    NETWORK = 'NetworkError',
    SYSTEM = 'SystemError',
    WORKER = 'WorkerError',
    UNEXPECTED = 'UnexpectedError',
    RESOURCE = 'ResourceError'
  }

  // Helper function to create standardized error messages
  export function createError(
    message: string, 
    category: ErrorCategory, 
    type: string, 
    details: Record<string, any> = {}
  ): Error {
    // Format the error message to include category and type
    const formattedMessage = `[${category}:${type}] ${message}`;
    
    // Create a standard Error with the formatted message
    const error = new Error(formattedMessage);
    
    // Add details as properties to the error object
    Object.assign(error, { details });
    
    return error;
  }
  
  // Helper function to identify error type from message
  export function identifyError(error: any): { category: string, type: string } | null {
    if (!error || !error.message) return null;
    
    // Try to extract category and type from the message format "[category:type] message"
    const match = error.message.match(/^\[(.*?):(.*?)\]/);
    if (match) {
      return {
        category: match[1],
        type: match[2]
      };
    }
    
    return null;
  }
  

  // Worker error types
  export type WorkerErrorType = 
    | 'initialization_failed'
    | 'runtime_error'
    | 'message_error'
    | 'worker_terminated'
    | 'resource_limit_exceeded'
    | 'worker_timeout'
    | 'interrupted'
    |'security_error'
  
  // Helper for creating worker errors
  export function createWorkerError(
    message: string, 
    type: WorkerErrorType, 
    details: Record<string, any> = {}
  ): Error {
    return createError(message, ErrorCategory.WORKER, type, details);
  }
  
  // Network error types
  export type NetworkErrorType =
    | 'request_timeout'
    | 'connection_failed'
    | 'api_error'
    | 'rate_limited'
    | 'server_error'
    | 'client_error';
  
  // Helper for creating network errors
  export function createNetworkError(
    message: string, 
    type: NetworkErrorType, 
    details: Record<string, any> = {}
  ): Error {
    return createError(message, ErrorCategory.NETWORK, type, details);
  }
  
  // Helper function to check if an error is of a specific category and type
  export function isErrorOf(
    error: any, 
    category?: ErrorCategory, 
    type?: string
  ): boolean {
    const errorInfo = identifyError(error);
    if (!errorInfo) return false;
    
    return (!category || errorInfo.category === category) &&
           (!type || errorInfo.type === type);
  }
  

  // Specific helper for worker errors
  export function isWorkerError(error: any): boolean {
    const errorInfo = identifyError(error);
    return errorInfo?.category === ErrorCategory.WORKER;
  }
  
  // Specific helper for network errors
  export function isNetworkError(error: any): boolean {
    const errorInfo = identifyError(error);
    return errorInfo?.category === ErrorCategory.NETWORK;
  }
