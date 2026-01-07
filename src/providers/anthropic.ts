/**
 * Anthropic LLM Provider for any-llm.
 * 
 * Uses the official Anthropic SDK for robust API interactions.
 * 
 * @see https://docs.anthropic.com/en/api/messages
 * @see https://github.com/anthropics/anthropic-sdk-typescript
 */

import Anthropic from '@anthropic-ai/sdk';
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
// Anthropic Provider Implementation
// =============================================================================

export class AnthropicProvider extends BaseProvider {
  readonly PROVIDER_NAME = 'anthropic';
  readonly ENV_API_KEY_NAME = 'ANTHROPIC_API_KEY';
  readonly PROVIDER_DOCUMENTATION_URL = 'https://docs.anthropic.com';
  readonly API_BASE = 'https://api.anthropic.com';
  
  readonly SUPPORTS_STREAMING = true;
  readonly SUPPORTS_TOOLS = true;
  readonly SUPPORTS_VISION = true;
  readonly SUPPORTS_LIST_MODELS = false; // Anthropic doesn't have a models endpoint
  readonly SUPPORTS_REASONING = true; // Claude 3.5 has extended thinking
  
  private client: Anthropic;
  private timeout: number;
  
  constructor(config: ProviderConfig = {}) {
    super(config);
    this.init();
    this.timeout = config.timeout ?? 60000;
    
    // Initialize the Anthropic client
    this.client = new Anthropic({
      apiKey: this.apiKey, // SDK will also check ANTHROPIC_API_KEY env
      baseURL: this.baseUrl !== this.API_BASE ? this.baseUrl : undefined,
      timeout: this.timeout,
      maxRetries: config.maxRetries as number ?? 2,
    });
  }
  
  /**
   * Check if Anthropic API is available.
   */
  async isAvailable(): Promise<boolean> {
    if (!this.apiKey && !process.env.ANTHROPIC_API_KEY) return false;
    
    // Anthropic doesn't have a simple health check endpoint
    // Validate API key format
    const key = this.apiKey || process.env.ANTHROPIC_API_KEY || '';
    return key.startsWith('sk-ant-');
  }
  
  /**
   * List available models (static list since Anthropic doesn't have an API).
   */
  async listModels(): Promise<ModelInfo[]> {
    return [
      {
        id: 'claude-sonnet-4-20250514',
        object: 'model',
        owned_by: 'anthropic',
        provider: 'anthropic',
        supports_tools: true,
        supports_vision: true,
      },
      {
        id: 'claude-3-5-sonnet-20241022',
        object: 'model',
        owned_by: 'anthropic',
        provider: 'anthropic',
        supports_tools: true,
        supports_vision: true,
      },
      {
        id: 'claude-3-5-haiku-20241022',
        object: 'model',
        owned_by: 'anthropic',
        provider: 'anthropic',
        supports_tools: true,
        supports_vision: true,
      },
      {
        id: 'claude-3-opus-20240229',
        object: 'model',
        owned_by: 'anthropic',
        provider: 'anthropic',
        supports_tools: true,
        supports_vision: true,
      },
    ];
  }
  
  /**
   * Handle SDK errors and convert to our error types.
   */
  private handleError(error: unknown): never {
    // Handle Anthropic SDK errors
    if (error && typeof error === 'object' && 'status' in error) {
      const apiError = error as { status: number; message?: string };
      
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
   * Convert messages to Anthropic format.
   * Handles system messages separately as Anthropic requires.
   */
  private convertMessages(messages: Message[]): { system?: string; messages: Anthropic.MessageParam[] } {
    let systemPrompt: string | undefined;
    const anthropicMessages: Anthropic.MessageParam[] = [];
    
    for (const msg of messages) {
      if (msg.role === 'system') {
        // Anthropic handles system as a separate parameter
        if (typeof msg.content === 'string') {
          systemPrompt = (systemPrompt ? systemPrompt + '\n' : '') + msg.content;
        }
        continue;
      }
      
      // Map 'tool' role to user with tool_result
      if (msg.role === 'tool') {
        anthropicMessages.push({
          role: 'user',
          content: [{
            type: 'tool_result',
            tool_use_id: msg.tool_call_id || '',
            content: typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content),
          }],
        });
        continue;
      }
      
      // Handle assistant messages
      if (msg.role === 'assistant') {
        const content: Anthropic.ContentBlockParam[] = [];
        
        // Add text content if present
        if (typeof msg.content === 'string' && msg.content) {
          content.push({ type: 'text', text: msg.content });
        }
        
        // Add tool use blocks
        if (msg.tool_calls && msg.tool_calls.length > 0) {
          for (const tc of msg.tool_calls) {
            content.push({
              type: 'tool_use',
              id: tc.id,
              name: tc.function.name,
              input: JSON.parse(tc.function.arguments),
            });
          }
        }
        
        anthropicMessages.push({
          role: 'assistant',
          content: content.length > 0 ? content : [{ type: 'text', text: '' }],
        });
        continue;
      }
      
      // Handle user messages
      if (typeof msg.content === 'string') {
        anthropicMessages.push({
          role: 'user',
          content: msg.content,
        });
      } else if (Array.isArray(msg.content)) {
        const contentBlocks: Anthropic.ContentBlockParam[] = [];
        
        for (const part of msg.content) {
          if (part.type === 'text') {
            contentBlocks.push({ type: 'text', text: part.text });
          } else if (part.type === 'image_url') {
            const url = part.image_url.url;
            if (url.startsWith('data:')) {
              // Parse data URL
              const match = url.match(/^data:([^;]+);base64,(.+)$/);
              if (match) {
                contentBlocks.push({
                  type: 'image',
                  source: {
                    type: 'base64',
                    media_type: match[1] as 'image/jpeg' | 'image/png' | 'image/gif' | 'image/webp',
                    data: match[2],
                  },
                });
              }
            }
          }
        }
        
        if (contentBlocks.length > 0) {
          anthropicMessages.push({
            role: 'user',
            content: contentBlocks,
          });
        }
      } else {
        anthropicMessages.push({
          role: 'user',
          content: '',
        });
      }
    }
    
    return { system: systemPrompt, messages: anthropicMessages };
  }
  
  /**
   * Convert tools to Anthropic format.
   */
  private convertTools(tools: Tool[]): Anthropic.Tool[] {
    return tools.map(tool => ({
      name: tool.function.name,
      description: tool.function.description || '',
      input_schema: tool.function.parameters as Anthropic.Tool['input_schema'],
    }));
  }
  
  /**
   * Convert tool choice to Anthropic format.
   */
  private convertToolChoice(choice: CompletionRequest['tool_choice']): Anthropic.ToolChoice | undefined {
    if (!choice) return undefined;
    if (choice === 'none') return undefined;
    if (choice === 'auto') return { type: 'auto' };
    if (choice === 'required') return { type: 'any' };
    if (typeof choice === 'object' && choice.type === 'function') {
      return { type: 'tool', name: choice.function.name };
    }
    return undefined;
  }
  
  /**
   * Convert Anthropic response to OpenAI-compatible format.
   */
  private convertResponse(response: Anthropic.Message): ChatCompletion {
    let textContent = '';
    const toolCalls: Message['tool_calls'] = [];
    
    for (const block of response.content) {
      if (block.type === 'text') {
        textContent += block.text;
      } else if (block.type === 'tool_use') {
        toolCalls.push({
          id: block.id,
          type: 'function',
          function: {
            name: block.name,
            arguments: JSON.stringify(block.input),
          },
        });
      }
    }
    
    // Determine finish reason
    let finishReason: 'stop' | 'length' | 'tool_calls' | 'content_filter' | null = 'stop';
    if (response.stop_reason === 'tool_use') {
      finishReason = 'tool_calls';
    } else if (response.stop_reason === 'max_tokens') {
      finishReason = 'length';
    }
    
    return {
      id: response.id,
      object: 'chat.completion',
      created: Math.floor(Date.now() / 1000),
      model: response.model,
      provider: 'anthropic',
      choices: [{
        index: 0,
        message: {
          role: 'assistant',
          content: textContent || null,
          tool_calls: toolCalls.length > 0 ? toolCalls : undefined,
        },
        finish_reason: finishReason,
      }],
      usage: {
        prompt_tokens: response.usage.input_tokens,
        completion_tokens: response.usage.output_tokens,
        total_tokens: response.usage.input_tokens + response.usage.output_tokens,
      },
    };
  }
  
  /**
   * Create a chat completion.
   */
  async completion(request: CompletionRequest): Promise<ChatCompletion> {
    try {
      const { system, messages } = this.convertMessages(request.messages);
      
      const params: Anthropic.MessageCreateParamsNonStreaming = {
        model: request.model,
        messages,
        max_tokens: request.max_tokens ?? 4096, // Anthropic requires max_tokens
      };
      
      // Add system prompt
      if (system) {
        params.system = system;
      }
      
      // Add optional parameters
      if (request.tools && request.tools.length > 0) {
        params.tools = this.convertTools(request.tools);
      }
      if (request.tool_choice !== undefined) {
        const toolChoice = this.convertToolChoice(request.tool_choice);
        if (toolChoice) {
          params.tool_choice = toolChoice;
        }
      }
      if (request.temperature !== undefined) {
        params.temperature = request.temperature;
      }
      if (request.top_p !== undefined) {
        params.top_p = request.top_p;
      }
      if (request.stop) {
        params.stop_sequences = Array.isArray(request.stop) ? request.stop : [request.stop];
      }
      
      const response = await this.client.messages.create(params);
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
      const { system, messages } = this.convertMessages(request.messages);
      
      const params: Anthropic.MessageCreateParamsStreaming = {
        model: request.model,
        messages,
        max_tokens: request.max_tokens ?? 4096,
        stream: true,
      };
      
      if (system) {
        params.system = system;
      }
      if (request.tools && request.tools.length > 0) {
        params.tools = this.convertTools(request.tools);
      }
      if (request.tool_choice !== undefined) {
        const toolChoice = this.convertToolChoice(request.tool_choice);
        if (toolChoice) {
          params.tool_choice = toolChoice;
        }
      }
      if (request.temperature !== undefined) {
        params.temperature = request.temperature;
      }
      if (request.top_p !== undefined) {
        params.top_p = request.top_p;
      }
      
      const stream = await this.client.messages.create(params);
      
      let messageId = '';
      let model = request.model;
      const created = Math.floor(Date.now() / 1000);
      
      // Track current tool use for streaming
      let currentToolUse: { id: string; name: string; arguments: string } | null = null;
      
      for await (const event of stream) {
        if (event.type === 'message_start') {
          const msgStart = event as Anthropic.MessageStreamEvent & { type: 'message_start'; message: Anthropic.Message };
          messageId = msgStart.message.id;
          model = msgStart.message.model;
        } else if (event.type === 'content_block_start') {
          const blockStart = event as Anthropic.ContentBlockStartEvent;
          if (blockStart.content_block.type === 'tool_use') {
            currentToolUse = {
              id: blockStart.content_block.id,
              name: blockStart.content_block.name,
              arguments: '',
            };
          }
        } else if (event.type === 'content_block_delta') {
          const deltaEvent = event as Anthropic.ContentBlockDeltaEvent;
          const chunkChoice: ChunkChoice = {
            index: 0,
            delta: {},
            finish_reason: null,
          };
          
          if (deltaEvent.delta.type === 'text_delta') {
            chunkChoice.delta.content = deltaEvent.delta.text;
          } else if (deltaEvent.delta.type === 'input_json_delta') {
            if (currentToolUse) {
              currentToolUse.arguments += deltaEvent.delta.partial_json;
            }
          }
          
          if (chunkChoice.delta.content) {
            yield {
              id: messageId || `anthropic-${Date.now()}`,
              object: 'chat.completion.chunk',
              created,
              model,
              choices: [chunkChoice],
            };
          }
        } else if (event.type === 'content_block_stop') {
          if (currentToolUse) {
            yield {
              id: messageId || `anthropic-${Date.now()}`,
              object: 'chat.completion.chunk',
              created,
              model,
              choices: [{
                index: 0,
                delta: {
                  tool_calls: [{
                    id: currentToolUse.id,
                    type: 'function',
                    function: {
                      name: currentToolUse.name,
                      arguments: currentToolUse.arguments,
                    },
                  }],
                },
                finish_reason: null,
              }],
            };
            currentToolUse = null;
          }
        } else if (event.type === 'message_delta') {
          const msgDelta = event as Anthropic.MessageDeltaEvent;
          let finishReason: ChunkChoice['finish_reason'] = 'stop';
          if (msgDelta.delta.stop_reason === 'tool_use') {
            finishReason = 'tool_calls';
          } else if (msgDelta.delta.stop_reason === 'max_tokens') {
            finishReason = 'length';
          }
          
          yield {
            id: messageId || `anthropic-${Date.now()}`,
            object: 'chat.completion.chunk',
            created,
            model,
            choices: [{
              index: 0,
              delta: {},
              finish_reason: finishReason,
            }],
          };
        }
      }
    } catch (error) {
      throw this.handleError(error);
    }
  }
}
