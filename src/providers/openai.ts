/**
 * OpenAI LLM Provider for any-llm.
 * 
 * Uses the official OpenAI SDK for robust API interactions.
 * 
 * @see https://platform.openai.com/docs/api-reference
 * @see https://github.com/openai/openai-node
 */

import OpenAI from 'openai';
import type {
  APIError,
  RateLimitError as OpenAIRateLimitError,
  AuthenticationError,
} from 'openai';
import { BaseProvider } from './base.js';
import type {
  ChatCompletion,
  ChatCompletionChunk,
  ChunkChoice,
  CompletionRequest,
  Message,
  ModelInfo,
  ProviderConfig,
  Tool,
} from '../types.js';
import { MissingApiKeyError, ProviderRequestError, RateLimitError, TimeoutError } from '../errors.js';

// =============================================================================
// OpenAI Provider Implementation
// =============================================================================

export class OpenAIProvider extends BaseProvider {
  readonly PROVIDER_NAME = 'openai';
  readonly ENV_API_KEY_NAME = 'OPENAI_API_KEY';
  readonly PROVIDER_DOCUMENTATION_URL = 'https://platform.openai.com/docs';
  readonly API_BASE = 'https://api.openai.com/v1';
  
  readonly SUPPORTS_STREAMING = true;
  readonly SUPPORTS_TOOLS = true;
  readonly SUPPORTS_VISION = true;
  readonly SUPPORTS_LIST_MODELS = true;
  readonly SUPPORTS_REASONING = false; // Only o1 models, handled separately
  
  protected client: OpenAI;
  private timeout: number;
  
  constructor(config: ProviderConfig = {}) {
    super(config);
    this.init();
    this.timeout = config.timeout ?? 60000;
    
    // Initialize the OpenAI client
    this.client = new OpenAI({
      apiKey: this.apiKey || '', // SDK will also check OPENAI_API_KEY env
      baseURL: this.baseUrl !== this.API_BASE ? this.baseUrl : undefined,
      timeout: this.timeout,
      maxRetries: config.maxRetries as number ?? 2,
    });
  }
  
  /**
   * Check if OpenAI API is available.
   */
  async isAvailable(): Promise<boolean> {
    if (!this.apiKey && !process.env.OPENAI_API_KEY) return false;
    
    try {
      // Use a lightweight models list call to check availability
      await this.client.models.list({ timeout: 5000 });
      return true;
    } catch {
      return false;
    }
  }
  
  /**
   * List available models.
   */
  async listModels(): Promise<ModelInfo[]> {
    try {
      const response = await this.client.models.list();
      
      return response.data
        .filter(m => m.id.includes('gpt') || m.id.includes('o1') || m.id.includes('o3'))
        .map(model => ({
          id: model.id,
          object: 'model' as const,
          created: model.created,
          owned_by: model.owned_by,
          provider: 'openai',
          supports_tools: this.modelSupportsTools(model.id),
          supports_vision: this.modelSupportsVision(model.id),
        }));
    } catch (error) {
      throw this.handleError(error);
    }
  }
  
  /**
   * Check if a model supports tools.
   */
  private modelSupportsTools(modelId: string): boolean {
    // o1 models don't support tools (as of late 2024)
    if (modelId.startsWith('o1-')) return false;
    return true;
  }
  
  /**
   * Check if a model supports vision.
   */
  private modelSupportsVision(modelId: string): boolean {
    const visionModels = ['gpt-4o', 'gpt-4-turbo', 'gpt-4-vision', 'o1'];
    return visionModels.some(m => modelId.includes(m));
  }
  
  /**
   * Handle SDK errors and convert to our error types.
   */
  protected handleError(error: unknown): never {
    // Handle OpenAI SDK errors
    if (error && typeof error === 'object' && 'status' in error) {
      const apiError = error as APIError;
      
      if (apiError.status === 429) {
        throw new RateLimitError(this.PROVIDER_NAME);
      }
      
      if (apiError.status === 401) {
        throw new MissingApiKeyError(this.PROVIDER_NAME, this.ENV_API_KEY_NAME);
      }
      
      throw new ProviderRequestError(
        this.PROVIDER_NAME,
        apiError.message || `HTTP ${apiError.status}`,
        apiError.status,
      );
    }
    
    // Handle timeout errors
    if (error instanceof Error && error.name === 'AbortError') {
      throw new TimeoutError(this.PROVIDER_NAME, this.timeout);
    }
    
    // Handle generic errors
    if (error instanceof Error) {
      throw new ProviderRequestError(this.PROVIDER_NAME, error.message);
    }
    
    throw new ProviderRequestError(this.PROVIDER_NAME, String(error));
  }
  
  /**
   * Convert our messages to OpenAI SDK format.
   */
  private convertMessages(messages: Message[]): OpenAI.ChatCompletionMessageParam[] {
    return messages.map(msg => {
      if (msg.role === 'system') {
        return {
          role: 'system' as const,
          content: typeof msg.content === 'string' ? msg.content : '',
        };
      }
      
      if (msg.role === 'tool') {
        return {
          role: 'tool' as const,
          content: typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content),
          tool_call_id: msg.tool_call_id || '',
        };
      }
      
      if (msg.role === 'assistant') {
        const assistantMsg: OpenAI.ChatCompletionAssistantMessageParam = {
          role: 'assistant' as const,
          content: typeof msg.content === 'string' ? msg.content : null,
        };
        
        if (msg.tool_calls && msg.tool_calls.length > 0) {
          assistantMsg.tool_calls = msg.tool_calls.map(tc => ({
            id: tc.id,
            type: 'function' as const,
            function: {
              name: tc.function.name,
              arguments: tc.function.arguments,
            },
          }));
        }
        
        return assistantMsg;
      }
      
      // User message
      if (typeof msg.content === 'string') {
        return {
          role: 'user' as const,
          content: msg.content,
        };
      }
      
      // Multimodal user message
      if (Array.isArray(msg.content)) {
        return {
          role: 'user' as const,
          content: msg.content.map(part => {
            if (part.type === 'text') {
              return { type: 'text' as const, text: part.text };
            }
            if (part.type === 'image_url') {
              return {
                type: 'image_url' as const,
                image_url: {
                  url: part.image_url.url,
                  detail: part.image_url.detail,
                },
              };
            }
            return part;
          }),
        };
      }
      
      return {
        role: 'user' as const,
        content: '',
      };
    });
  }
  
  /**
   * Convert our tools to OpenAI SDK format.
   */
  private convertTools(tools: Tool[]): OpenAI.ChatCompletionTool[] {
    return tools.map(tool => ({
      type: 'function' as const,
      function: {
        name: tool.function.name,
        description: tool.function.description,
        parameters: tool.function.parameters as Record<string, unknown>,
      },
    }));
  }
  
  /**
   * Convert OpenAI SDK response to our format.
   */
  private convertResponse(response: OpenAI.ChatCompletion): ChatCompletion {
    return {
      id: response.id,
      object: 'chat.completion',
      created: response.created,
      model: response.model,
      provider: 'openai',
      choices: response.choices.map(choice => ({
        index: choice.index,
        message: {
          role: choice.message.role,
          content: choice.message.content,
          tool_calls: choice.message.tool_calls?.map(tc => ({
            id: tc.id,
            type: 'function' as const,
            function: {
              name: tc.function.name,
              arguments: tc.function.arguments,
            },
          })),
        },
        finish_reason: choice.finish_reason as ChatCompletion['choices'][0]['finish_reason'],
      })),
      usage: response.usage ? {
        prompt_tokens: response.usage.prompt_tokens,
        completion_tokens: response.usage.completion_tokens,
        total_tokens: response.usage.total_tokens,
      } : undefined,
    };
  }
  
  /**
   * Create a chat completion.
   */
  async completion(request: CompletionRequest): Promise<ChatCompletion> {
    try {
      const params: OpenAI.ChatCompletionCreateParamsNonStreaming = {
        model: request.model,
        messages: this.convertMessages(request.messages),
        stream: false,
      };
      
      // Add optional parameters
      if (request.tools && request.tools.length > 0) {
        params.tools = this.convertTools(request.tools);
      }
      if (request.tool_choice !== undefined) {
        params.tool_choice = request.tool_choice as OpenAI.ChatCompletionToolChoiceOption;
      }
      if (request.temperature !== undefined) {
        params.temperature = request.temperature;
      }
      if (request.top_p !== undefined) {
        params.top_p = request.top_p;
      }
      if (request.max_tokens !== undefined) {
        // Use max_completion_tokens for newer models
        if (request.model.startsWith('o1') || request.model.startsWith('o3')) {
          params.max_completion_tokens = request.max_tokens;
        } else {
          params.max_tokens = request.max_tokens;
        }
      }
      if (request.n !== undefined) {
        params.n = request.n;
      }
      if (request.stop !== undefined) {
        params.stop = request.stop;
      }
      if (request.presence_penalty !== undefined) {
        params.presence_penalty = request.presence_penalty;
      }
      if (request.frequency_penalty !== undefined) {
        params.frequency_penalty = request.frequency_penalty;
      }
      if (request.seed !== undefined) {
        params.seed = request.seed;
      }
      if (request.user !== undefined) {
        params.user = request.user;
      }
      if (request.response_format !== undefined) {
        params.response_format = request.response_format as OpenAI.ChatCompletionCreateParams['response_format'];
      }
      
      const response = await this.client.chat.completions.create(params);
      return this.convertResponse(response);
    } catch (error) {
      throw this.handleError(error);
    }
  }
  
  /**
   * Stream a chat completion.
   */
  async *completionStream(request: CompletionRequest): AsyncIterable<ChatCompletionChunk> {
    try {
      const params: OpenAI.ChatCompletionCreateParamsStreaming = {
        model: request.model,
        messages: this.convertMessages(request.messages),
        stream: true,
      };
      
      // Add optional parameters (same as non-streaming)
      if (request.tools && request.tools.length > 0) {
        params.tools = this.convertTools(request.tools);
      }
      if (request.tool_choice !== undefined) {
        params.tool_choice = request.tool_choice as OpenAI.ChatCompletionToolChoiceOption;
      }
      if (request.temperature !== undefined) {
        params.temperature = request.temperature;
      }
      if (request.top_p !== undefined) {
        params.top_p = request.top_p;
      }
      if (request.max_tokens !== undefined) {
        if (request.model.startsWith('o1') || request.model.startsWith('o3')) {
          params.max_completion_tokens = request.max_tokens;
        } else {
          params.max_tokens = request.max_tokens;
        }
      }
      
      const stream = await this.client.chat.completions.create(params);
      
      for await (const chunk of stream) {
        const deltaToolCalls = chunk.choices[0]?.delta?.tool_calls;
        const chunkChoice: ChunkChoice = {
          index: chunk.choices[0]?.index || 0,
          delta: {
            role: chunk.choices[0]?.delta?.role as ChunkChoice['delta']['role'],
            content: chunk.choices[0]?.delta?.content || undefined,
            tool_calls: deltaToolCalls?.map(tc => ({
              id: tc.id,
              type: 'function' as const,
              function: tc.function ? {
                name: tc.function.name || '',
                arguments: tc.function.arguments || '',
              } : undefined,
            })),
          },
          finish_reason: chunk.choices[0]?.finish_reason as ChunkChoice['finish_reason'],
        };
        
        yield {
          id: chunk.id,
          object: 'chat.completion.chunk',
          created: chunk.created,
          model: chunk.model,
          choices: [chunkChoice],
        };
      }
    } catch (error) {
      throw this.handleError(error);
    }
  }
}

/**
 * Base class for OpenAI-compatible providers.
 * Providers like Groq, Together, OpenRouter can extend this.
 */
export class OpenAICompatibleProvider extends OpenAIProvider {
  constructor(
    providerName: string,
    apiBase: string,
    envKeyName: string,
    config: ProviderConfig = {},
  ) {
    super({
      ...config,
      baseUrl: config.baseUrl || apiBase,
    });
    
    // Override the readonly properties via Object.defineProperty
    Object.defineProperty(this, 'PROVIDER_NAME', { value: providerName });
    Object.defineProperty(this, 'API_BASE', { value: apiBase });
    Object.defineProperty(this, 'ENV_API_KEY_NAME', { value: envKeyName });
    
    // Reinitialize client with correct base URL
    this.client = new OpenAI({
      apiKey: this.apiKey || '',
      baseURL: config.baseUrl || apiBase,
      timeout: config.timeout ?? 60000,
      maxRetries: config.maxRetries as number ?? 2,
    });
  }
}
