/**
 * Llamafile LLM Provider for any-llm.
 * 
 * Llamafile provides an OpenAI-compatible API on localhost:8080.
 * This provider uses the OpenAI SDK with a custom baseURL.
 * 
 * @see https://github.com/Mozilla-Ocho/llamafile
 */

import OpenAI from 'openai';
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
import { ProviderRequestError, ProviderUnavailableError, TimeoutError } from '../errors.js';

// =============================================================================
// Llamafile Provider Implementation
// =============================================================================

export class LlamafileProvider extends BaseProvider {
  readonly PROVIDER_NAME = 'llamafile';
  readonly ENV_API_KEY_NAME = 'LLAMAFILE_API_KEY'; // Optional
  readonly PROVIDER_DOCUMENTATION_URL = 'https://github.com/Mozilla-Ocho/llamafile';
  readonly API_BASE = 'http://localhost:8080/v1';
  
  readonly SUPPORTS_STREAMING = true;
  readonly SUPPORTS_TOOLS = true;
  readonly SUPPORTS_VISION = false; // Depends on model
  readonly SUPPORTS_LIST_MODELS = true;
  readonly SUPPORTS_REASONING = false;
  
  private client: OpenAI;
  private timeout: number;
  
  constructor(config: ProviderConfig = {}) {
    super(config);
    this.init();
    this.timeout = config.timeout ?? 60000;
    
    // Initialize the OpenAI client pointed at llamafile
    this.client = new OpenAI({
      apiKey: this.apiKey || 'not-needed', // Llamafile doesn't require an API key
      baseURL: this.baseUrl,
      timeout: this.timeout,
      maxRetries: config.maxRetries as number ?? 2,
    });
  }
  
  /**
   * Llamafile doesn't require an API key.
   */
  protected requiresApiKey(): boolean {
    return false;
  }
  
  /**
   * Check if llamafile is available.
   */
  async isAvailable(): Promise<boolean> {
    try {
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
      const response = await this.client.models.list({ timeout: 5000 });
      
      return response.data.map(model => ({
        id: model.id,
        object: 'model' as const,
        created: model.created,
        owned_by: model.owned_by || 'llamafile',
        provider: 'llamafile',
        supports_tools: true,
      }));
    } catch (error) {
      if (error instanceof Error && (error.message.includes('ECONNREFUSED') || error.message.includes('fetch failed'))) {
        throw new ProviderUnavailableError(this.PROVIDER_NAME, 'Llamafile is not running');
      }
      throw new ProviderUnavailableError(this.PROVIDER_NAME, 'Failed to list models');
    }
  }
  
  /**
   * Handle SDK errors and convert to our error types.
   */
  private handleError(error: unknown): never {
    // Handle connection errors
    if (error instanceof Error && (error.message.includes('ECONNREFUSED') || error.message.includes('fetch failed'))) {
      throw new ProviderUnavailableError(this.PROVIDER_NAME, 'Llamafile is not running. Start your llamafile first.');
    }
    
    // Handle timeout errors
    if (error instanceof Error && error.name === 'AbortError') {
      throw new TimeoutError(this.PROVIDER_NAME, this.timeout);
    }
    
    // Handle OpenAI SDK errors
    if (error && typeof error === 'object' && 'status' in error) {
      const apiError = error as { status: number; message?: string };
      throw new ProviderRequestError(
        this.PROVIDER_NAME,
        apiError.message || `HTTP ${apiError.status}`,
        apiError.status,
      );
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
      
      // User message - flatten multimodal to text only for llamafile
      let content = '';
      if (typeof msg.content === 'string') {
        content = msg.content;
      } else if (Array.isArray(msg.content)) {
        content = msg.content
          .filter(p => p.type === 'text')
          .map(p => (p as { text: string }).text)
          .join(' ');
      }
      
      return {
        role: 'user' as const,
        content,
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
      provider: 'llamafile',
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
        model: request.model || 'default',
        messages: this.convertMessages(request.messages),
        stream: false,
      };
      
      if (request.tools && request.tools.length > 0) {
        params.tools = this.convertTools(request.tools);
      }
      if (request.max_tokens !== undefined) {
        params.max_tokens = request.max_tokens;
      }
      if (request.temperature !== undefined) {
        params.temperature = request.temperature;
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
        model: request.model || 'default',
        messages: this.convertMessages(request.messages),
        stream: true,
      };
      
      if (request.tools && request.tools.length > 0) {
        params.tools = this.convertTools(request.tools);
      }
      if (request.max_tokens !== undefined) {
        params.max_tokens = request.max_tokens;
      }
      if (request.temperature !== undefined) {
        params.temperature = request.temperature;
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
          id: chunk.id || `llamafile-${Date.now()}`,
          object: 'chat.completion.chunk',
          created: chunk.created || Math.floor(Date.now() / 1000),
          model: chunk.model || request.model || 'default',
          choices: [chunkChoice],
        };
      }
    } catch (error) {
      throw this.handleError(error);
    }
  }
}
