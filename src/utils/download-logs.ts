// utils/download-logs.ts
import { API_ENDPOINT } from "./constants";
import { createError, ErrorCategory } from "./errors";

/**
 * Logs the start of a model download
 * @param modelId - The ID of the model being downloaded
 * @param precision - The precision of the model (e.g., "q4", "fp16")
 * @param isDev - Whether this is a dev mode download
 * @param clientIdentifier - Optional client identifier for tracking
 * @param authToken - Optional authentication token
 * @returns The download ID for tracking
 */
export async function logStartDownload(
  modelId: string, 
  precision: string, 
  price: number = 0,
  isDev: boolean = false,
  clientIdentifier?: string,
  authToken?: string
): Promise<string> {
  try {
    const headers: HeadersInit = {
      "Content-Type": "application/json",
    };

    // Add Authorization header if authToken is provided
    if (authToken) {
      headers["Authorization"] = `Bearer ${authToken}`;
    }

    const response = await fetch(`${API_ENDPOINT}/download/start`, {
      method: "POST",
      headers,
      body: JSON.stringify({
        modelId,
        precision,
        price,
        type: isDev ? "dev" : "prod",
        clientIdentifier,
      })
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw createError(
        `Failed to log download start: ${errorData.error || response.statusText}`,
        ErrorCategory.NETWORK,
        'server_error'
      );
    }

    const { downloadId } = await response.json();
    
    if (!downloadId) {
      throw createError(
        "Server did not return a downloadId",
        ErrorCategory.NETWORK,
        'server_error'
      );
    }

    return downloadId;
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : "Unknown error";
    throw createError(
      `Failed to log start download: ${errorMessage}`,
      ErrorCategory.NETWORK,
      'request_failed'
    );
  }
}

/**
 * Logs the successful completion of a model download
 * @param downloadId - The download ID returned from logStartDownload
 */
export async function logEndDownload(downloadId: string): Promise<void> {
  try {
    const response = await fetch(`${API_ENDPOINT}/download/end`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        downloadId,
      })
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw createError(
        `Failed to log download end: ${errorData.error || response.statusText}`,
        ErrorCategory.NETWORK,
        'server_error'
      );
    }
  } catch (error) {
    // Don't throw on logging failures - just log the error
    console.error("Failed to log download completion:", error);
  }
}

/**
 * Logs a failed model download
 * @param downloadId - The download ID returned from logStartDownload
 */
export async function logFailedDownload(downloadId: string): Promise<void> {
  try {
    const response = await fetch(`${API_ENDPOINT}/download/failed`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        downloadId,
      })
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw createError(
        `Failed to log download failure: ${errorData.error || response.statusText}`,
        ErrorCategory.NETWORK,
        'server_error'
      );
    }
  } catch (error) {
    // Don't throw on logging failures - just log the error
    console.error("Failed to log download failure:", error);
  }
}