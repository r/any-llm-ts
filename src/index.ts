/**
 * any-llm-ts - A unified TypeScript interface for LLM providers
 * 
 * A unified interface for communicating with LLM providers.
 * Supports both remote providers (OpenAI, Anthropic) and local providers (Ollama, llamafile).
 * 
 * Inspired by mozilla-ai/any-llm.
 * 
 * @example
 * ```typescript
 * import { completion, AnyLLM } from 'any-llm-ts';
 * 
 * // Direct function API (stateless, creates new client per call)
 * const response = await completion({
 *   model: 'openai:gpt-4o',
 *   messages: [{ role: 'user', content: 'Hello!' }],
 * });
 * 
 * // Class API (reuses client, better for production)
 * const llm = AnyLLM.create('openai', { apiKey: 'sk-...' });
 * const response = await llm.completion({
 *   model: 'gpt-4o',
 *   messages: [{ role: 'user', content: 'Hello!' }],
 * });
 * ```
 * 
 * @module any-llm-ts
 * @packageDocumentation
 */

// =============================================================================
// Package Info
// =============================================================================

/** Package version */
export const VERSION = '0.1.0';

/** Package name */
export const PACKAGE_NAME = 'any-llm-ts';

// =============================================================================
// Main API
// =============================================================================

export {
  // Direct function API
  completion,
  completionStream,
  listModels,
  checkProvider,
  getSupportedProviders,
  
  // Class-based API
  AnyLLM,
} from './api.js';

// =============================================================================
// Types
// =============================================================================

export type {
  // Provider types
  LLMProviderType,
  ProviderConfig,
  ProviderMetadata,
  
  // Message types
  MessageRole,
  Message,
  MessageWithReasoning,
  MessageContent,
  TextContentPart,
  ImageContentPart,
  ToolCall,
  Tool,
  
  // Request/Response types
  CompletionRequest,
  CompletionChoice,
  CompletionUsage,
  ChatCompletion,
  
  // Streaming types
  ChunkDelta,
  ChunkChoice,
  ChatCompletionChunk,
  
  // Model types
  ModelInfo,
  
  // Utility types
  ParsedModel,
  ProviderStatus,
  
  // Error types
  AnyLLMErrorCode,
} from './types.js';

export { AnyLLMError } from './types.js';

// =============================================================================
// Errors
// =============================================================================

export {
  MissingApiKeyError,
  UnsupportedProviderError,
  ProviderRequestError,
  RateLimitError,
  ProviderUnavailableError,
  TimeoutError,
  isAnyLLMError,
  isRateLimitError,
  wrapError,
} from './errors.js';

// =============================================================================
// Registry (for advanced usage)
// =============================================================================

export {
  registerProvider,
  getProviderConstructor,
  hasProvider,
  getRegisteredProviders,
  createProvider,
  parseModelString,
  resolveProviderAndModel,
  getProviderForModel,
  clearProviderCache,
} from './registry.js';

// =============================================================================
// Providers (for advanced usage)
// =============================================================================

export {
  BaseProvider,
  type ProviderConstructor,
  OllamaProvider,
  OpenAIProvider,
  OpenAICompatibleProvider,
  AnthropicProvider,
  LlamafileProvider,
} from './providers/index.js';

