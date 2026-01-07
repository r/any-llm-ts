/**
 * Error types for any-llm-ts.
 * 
 * Provides structured errors following the patterns from mozilla-ai/any-llm.
 * Now enhanced to handle errors from official provider SDKs.
 */

import { AnyLLMError, AnyLLMErrorCode } from './types.js';

// =============================================================================
// SDK Error Type Guards
// =============================================================================

/**
 * Check if an error is from the OpenAI SDK.
 */
export function isOpenAIError(error: unknown): error is { status: number; message: string; code?: string } {
  return (
    error !== null &&
    typeof error === 'object' &&
    'status' in error &&
    typeof (error as Record<string, unknown>).status === 'number'
  );
}

/**
 * Check if an error is from the Anthropic SDK.
 */
export function isAnthropicError(error: unknown): error is { status: number; message: string; error?: { type: string } } {
  return (
    error !== null &&
    typeof error === 'object' &&
    'status' in error &&
    typeof (error as Record<string, unknown>).status === 'number'
  );
}

/**
 * Check if an error is a connection error (Ollama, Llamafile not running).
 */
export function isConnectionError(error: unknown): boolean {
  if (!(error instanceof Error)) return false;
  const message = error.message.toLowerCase();
  return (
    message.includes('econnrefused') ||
    message.includes('fetch failed') ||
    message.includes('network') ||
    message.includes('connection')
  );
}

// =============================================================================
// Error Classes
// =============================================================================

/**
 * Error thrown when an API key is missing.
 */
export class MissingApiKeyError extends AnyLLMError {
  constructor(provider: string, envKey: string) {
    super(
      `API key not found for provider '${provider}'. ` +
      `Please set the ${envKey} environment variable or pass api_key in the request.`,
      AnyLLMErrorCode.MissingApiKey,
      provider,
    );
    this.name = 'MissingApiKeyError';
  }
}

/**
 * Error thrown when a provider is not supported.
 */
export class UnsupportedProviderError extends AnyLLMError {
  constructor(provider: string, supported: string[]) {
    super(
      `Provider '${provider}' is not supported. ` +
      `Supported providers: ${supported.join(', ')}`,
      AnyLLMErrorCode.UnsupportedProvider,
    );
    this.name = 'UnsupportedProviderError';
  }
}

/**
 * Error thrown when a request to a provider fails.
 */
export class ProviderRequestError extends AnyLLMError {
  constructor(
    provider: string,
    message: string,
    statusCode?: number,
    cause?: Error,
  ) {
    super(
      `Request to ${provider} failed: ${message}`,
      AnyLLMErrorCode.RequestFailed,
      provider,
      statusCode,
      cause,
    );
    this.name = 'ProviderRequestError';
  }
}

/**
 * Error thrown when a provider is rate limited.
 */
export class RateLimitError extends AnyLLMError {
  constructor(
    provider: string,
    retryAfter?: number,
  ) {
    const message = retryAfter 
      ? `Rate limited by ${provider}. Retry after ${retryAfter} seconds.`
      : `Rate limited by ${provider}.`;
    super(message, AnyLLMErrorCode.RateLimited, provider, 429);
    this.name = 'RateLimitError';
  }
}

/**
 * Error thrown when a provider is unavailable.
 */
export class ProviderUnavailableError extends AnyLLMError {
  constructor(provider: string, reason?: string) {
    super(
      reason 
        ? `Provider '${provider}' is unavailable: ${reason}`
        : `Provider '${provider}' is unavailable`,
      AnyLLMErrorCode.ProviderUnavailable,
      provider,
    );
    this.name = 'ProviderUnavailableError';
  }
}

/**
 * Error thrown when a request times out.
 */
export class TimeoutError extends AnyLLMError {
  constructor(provider: string, timeoutMs: number) {
    super(
      `Request to ${provider} timed out after ${timeoutMs}ms`,
      AnyLLMErrorCode.Timeout,
      provider,
    );
    this.name = 'TimeoutError';
  }
}

/**
 * Check if an error is an any-llm error.
 */
export function isAnyLLMError(error: unknown): error is AnyLLMError {
  return error instanceof AnyLLMError;
}

/**
 * Check if an error indicates rate limiting.
 */
export function isRateLimitError(error: unknown): error is RateLimitError {
  return error instanceof RateLimitError;
}

/**
 * Wrap an unknown error into an AnyLLMError.
 * Enhanced to handle errors from official provider SDKs.
 */
export function wrapError(error: unknown, provider: string): AnyLLMError {
  if (error instanceof AnyLLMError) {
    return error;
  }
  
  // Handle SDK errors with status codes
  if (isOpenAIError(error) || isAnthropicError(error)) {
    const { status, message } = error;
    
    if (status === 429) {
      return new RateLimitError(provider);
    }
    
    if (status === 401 || status === 403) {
      return new MissingApiKeyError(provider, `${provider.toUpperCase()}_API_KEY`);
    }
    
    return new ProviderRequestError(provider, message, status);
  }
  
  // Handle connection errors (local providers not running)
  if (isConnectionError(error)) {
    return new ProviderUnavailableError(
      provider,
      `Cannot connect to ${provider}. Is it running?`
    );
  }
  
  if (error instanceof Error) {
    // Check for common error patterns
    const message = error.message.toLowerCase();
    
    if (message.includes('rate limit') || message.includes('too many requests')) {
      return new RateLimitError(provider);
    }
    
    if (message.includes('unauthorized') || message.includes('invalid api key') || message.includes('authentication')) {
      return new MissingApiKeyError(provider, `${provider.toUpperCase()}_API_KEY`);
    }
    
    if (message.includes('timeout') || message.includes('timed out') || error.name === 'AbortError') {
      return new TimeoutError(provider, 0);
    }
    
    return new ProviderRequestError(provider, error.message, undefined, error);
  }
  
  return new ProviderRequestError(
    provider,
    String(error),
  );
}

