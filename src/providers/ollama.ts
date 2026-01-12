/**
 * Ollama LLM Provider for any-llm.
 * 
 * Uses the official Ollama SDK for robust API interactions.
 * Ollama runs locally and provides access to open-source models.
 * 
 * @see https://github.com/ollama/ollama
 * @see https://github.com/ollama/ollama-js
 */

import { Ollama } from 'ollama';
import type {
  ChatRequest,
  ChatResponse,
  Message as OllamaMessage,
  Tool as OllamaTool,
  ToolCall as OllamaToolCall,
  ListResponse,
} from 'ollama';
import { BaseProvider } from './base.js';
import type {
  ChatCompletion,
  ChatCompletionChunk,
  ChunkChoice,
  CompletionChoice,
  CompletionRequest,
  Message,
  ModelInfo,
  ProviderConfig,
  Tool,
} from '../types.js';
import { ProviderRequestError, ProviderUnavailableError, TimeoutError } from '../errors.js';

// Minimum Ollama version for tool calling support
const MINIMUM_TOOL_VERSION = '0.3.0';

// =============================================================================
// Ollama Provider Implementation
// =============================================================================

export class OllamaProvider extends BaseProvider {
  readonly PROVIDER_NAME = 'ollama';
  readonly ENV_API_KEY_NAME = 'OLLAMA_API_KEY'; // Optional
  readonly PROVIDER_DOCUMENTATION_URL = 'https://github.com/ollama/ollama';
  readonly API_BASE = 'http://localhost:11434';
  
  readonly SUPPORTS_STREAMING = true;
  readonly SUPPORTS_TOOLS = true;
  readonly SUPPORTS_VISION = true;
  readonly SUPPORTS_LIST_MODELS = true;
  readonly SUPPORTS_REASONING = true;
  
  private client: Ollama;
  private timeout: number;
  private version: string | null = null;
  private toolsSupported: boolean | null = null;
  
  constructor(config: ProviderConfig = {}) {
    super(config);
    this.init();
    this.timeout = config.timeout ?? 120000; // Ollama can be slow
    
    // Initialize the Ollama client
    this.client = new Ollama({
      host: this.baseUrl,
    });
  }
  
  /**
   * Ollama doesn't require an API key.
   */
  protected requiresApiKey(): boolean {
    return false;
  }
  
  /**
   * Compare semantic versions.
   */
  private compareVersions(a: string, b: string): number {
    const partsA = a.replace(/^v/, '').split('.').map(p => parseInt(p, 10) || 0);
    const partsB = b.replace(/^v/, '').split('.').map(p => parseInt(p, 10) || 0);
    
    const maxLen = Math.max(partsA.length, partsB.length);
    for (let i = 0; i < maxLen; i++) {
      const numA = partsA[i] || 0;
      const numB = partsB[i] || 0;
      if (numA !== numB) return numA - numB;
    }
    return 0;
  }
  
  /**
   * Check if Ollama is available.
   */
  async isAvailable(): Promise<boolean> {
    try {
      // Use list() to check if Ollama is running
      const response = await this.client.list();
      
      // Check version for tool support if we haven't already
      if (this.version === null) {
        try {
          // The SDK doesn't expose version directly, but we can infer from capabilities
          // For now, assume tools are supported if we can connect
          this.toolsSupported = true;
        } catch {
          this.toolsSupported = true; // Assume support if we can't determine
        }
      }
      
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
      const response = await this.client.list();
      
      return response.models.map(model => ({
        id: model.name,
        object: 'model' as const,
        owned_by: 'ollama',
        provider: 'ollama',
        supports_tools: this.modelSupportsTools(model.name),
        supports_vision: this.modelSupportsVision(model.name),
      }));
    } catch (error) {
      if (error instanceof Error) {
        throw new ProviderUnavailableError(this.PROVIDER_NAME, error.message);
      }
      throw new ProviderUnavailableError(this.PROVIDER_NAME, 'Failed to list models');
    }
  }
  
  /**
   * Check if a model supports tools (heuristic).
   * 
   * Native tool calling in Ollama requires:
   * - Ollama v0.3.0+ (server-side)
   * - A model that was fine-tuned for tool calling
   * 
   * Models with confirmed tool calling support:
   * - llama3.1, llama3.2, llama3.3 (8B+)
   * - mistral-nemo, mistral-large, mixtral (NOT mistral:7b-instruct which is older)
   * - qwen2.5
   * - phi4
   * - command-r, command-r-plus
   * - granite3
   */
  private modelSupportsTools(modelName: string): boolean {
    const name = modelName.toLowerCase();
    
    // Exclude older models that don't support tools
    const noToolSupport = [
      'mistral:7b-instruct',  // Old Mistral before tool support
      'mistral:7b',           // Base mistral without instruction tuning for tools
      'llama2',               // Llama 2 doesn't have tool support
      'codellama',            // Code Llama doesn't have tool support
    ];
    if (noToolSupport.some(m => name.includes(m))) {
      return false;
    }
    
    // Models with confirmed tool calling support
    const toolModels = [
      'llama3.1', 'llama3.2', 'llama3.3',  // Llama 3.1+ has native tool support
      'mistral-nemo', 'mistral-large', 'mixtral',  // Newer Mistral models
      'qwen2.5',  // Qwen 2.5 has tool support
      'phi4',     // Phi-4 has tool support
      'granite3', // Granite 3 has tool support
      'command-r', // Command R models have tool support
    ];
    return toolModels.some(m => name.includes(m));
  }
  
  /**
   * Check if a model supports vision (heuristic).
   */
  private modelSupportsVision(modelName: string): boolean {
    const visionModels = [
      'llava',
      'bakllava',
      'llama3.2-vision',
      'moondream',
    ];
    return visionModels.some(m => modelName.toLowerCase().includes(m));
  }
  
  /**
   * Handle SDK errors and convert to our error types.
   */
  private handleError(error: unknown): never {
    // Handle timeout errors
    if (error instanceof Error && (error.name === 'AbortError' || error.message.includes('timeout'))) {
      throw new TimeoutError(this.PROVIDER_NAME, this.timeout);
    }
    
    // Handle connection errors
    if (error instanceof Error && (error.message.includes('ECONNREFUSED') || error.message.includes('fetch failed'))) {
      throw new ProviderUnavailableError(this.PROVIDER_NAME, 'Ollama is not running. Start it with: ollama serve');
    }
    
    // Handle generic errors
    if (error instanceof Error) {
      throw new ProviderRequestError(this.PROVIDER_NAME, error.message);
    }
    
    throw new ProviderRequestError(this.PROVIDER_NAME, String(error));
  }
  
  /**
   * Convert messages to Ollama format.
   */
  private convertMessages(messages: Message[]): OllamaMessage[] {
    return messages.map(msg => {
      const ollamaMsg: OllamaMessage = {
        role: msg.role,
        content: '',
      };
      
      // Handle content
      if (typeof msg.content === 'string') {
        ollamaMsg.content = msg.content;
      } else if (Array.isArray(msg.content)) {
        // Handle multimodal content
        const textParts: string[] = [];
        const images: string[] = [];
        
        for (const part of msg.content) {
          if (part.type === 'text') {
            textParts.push(part.text);
          } else if (part.type === 'image_url') {
            const url = part.image_url.url;
            if (url.startsWith('data:image/')) {
              // Extract base64 data
              const base64 = url.split(',')[1];
              if (base64) images.push(base64);
            }
          }
        }
        
        ollamaMsg.content = textParts.join(' ');
        if (images.length > 0) {
          ollamaMsg.images = images;
        }
      } else if (msg.content === null) {
        ollamaMsg.content = '';
      }
      
      // Handle tool calls
      if (msg.tool_calls && msg.tool_calls.length > 0) {
        ollamaMsg.tool_calls = msg.tool_calls.map(tc => ({
          function: {
            name: tc.function.name,
            arguments: JSON.parse(tc.function.arguments),
          },
        }));
      }
      
      return ollamaMsg;
    });
  }
  
  /**
   * Convert tools to Ollama format.
   */
  private convertTools(tools: Tool[]): OllamaTool[] {
    return tools.map(tool => ({
      type: 'function',
      function: {
        name: tool.function.name,
        description: tool.function.description || '',
        parameters: tool.function.parameters as OllamaTool['function']['parameters'],
      },
    }));
  }
  
  /**
   * Convert Ollama response to our format.
   */
  private convertResponse(response: ChatResponse, model: string): ChatCompletion {
    const message: Message = {
      role: response.message.role as Message['role'],
      content: response.message.content || null,
    };
    
    // Convert tool calls
    if (response.message.tool_calls && response.message.tool_calls.length > 0) {
      message.tool_calls = response.message.tool_calls.map((tc, i) => ({
        id: `call_${i}`,
        type: 'function' as const,
        function: {
          name: tc.function.name,
          arguments: JSON.stringify(tc.function.arguments),
        },
      }));
    }
    
    // Determine finish reason
    let finishReason: CompletionChoice['finish_reason'] = 'stop';
    if (message.tool_calls && message.tool_calls.length > 0) {
      finishReason = 'tool_calls';
    }
    
    return {
      id: `ollama-${Date.now()}`,
      object: 'chat.completion',
      created: Math.floor(Date.now() / 1000),
      model: response.model || model,
      provider: 'ollama',
      choices: [{
        index: 0,
        message,
        finish_reason: finishReason,
      }],
      usage: {
        prompt_tokens: response.prompt_eval_count || 0,
        completion_tokens: response.eval_count || 0,
        total_tokens: (response.prompt_eval_count || 0) + (response.eval_count || 0),
      },
    };
  }
  
  /**
   * Create a chat completion.
   */
  async completion(request: CompletionRequest): Promise<ChatCompletion> {
    try {
      const chatRequest: ChatRequest & { stream: false } = {
        model: request.model,
        messages: this.convertMessages(request.messages),
        stream: false,
      };
      
      // Add tools if provided
      if (request.tools && request.tools.length > 0) {
        if (this.toolsSupported === false) {
          throw new ProviderRequestError(
            this.PROVIDER_NAME,
            `Ollama version ${this.version} does not support tool calling. ` +
            `Upgrade to ${MINIMUM_TOOL_VERSION} or later.`
          );
        }
        chatRequest.tools = this.convertTools(request.tools);
        console.log(`[OllamaProvider] Passing ${chatRequest.tools.length} tools to Ollama:`, 
          chatRequest.tools.map(t => t.function.name).join(', '));
      } else {
        console.log('[OllamaProvider] No tools passed to Ollama');
      }
      
      // Add options
      chatRequest.options = {};
      if (request.temperature !== undefined) {
        chatRequest.options.temperature = request.temperature;
      }
      if (request.top_p !== undefined) {
        chatRequest.options.top_p = request.top_p;
      }
      if (request.max_tokens !== undefined) {
        chatRequest.options.num_predict = request.max_tokens;
      }
      if (request.stop) {
        chatRequest.options.stop = Array.isArray(request.stop) ? request.stop : [request.stop];
      }
      if (request.seed !== undefined) {
        chatRequest.options.seed = request.seed;
      }
      
      const response = await this.client.chat(chatRequest);
      
      // Debug logging for tool calls - log to stderr (bridge log) as well
      const toolCallsDebug = {
        model: response.model,
        hasToolCalls: !!response.message?.tool_calls?.length,
        toolCallCount: response.message?.tool_calls?.length || 0,
        toolNames: response.message?.tool_calls?.map(tc => tc.function.name) || [],
        contentLength: response.message?.content?.length || 0,
      };
      console.log('[OllamaProvider] Response summary:', JSON.stringify(toolCallsDebug));
      process.stderr.write(`[OllamaProvider] Response: ${JSON.stringify(toolCallsDebug)}\n`);
      
      return this.convertResponse(response, request.model);
    } catch (error) {
      // Re-throw our errors
      if (error instanceof ProviderRequestError || error instanceof ProviderUnavailableError || error instanceof TimeoutError) {
        throw error;
      }
      throw this.handleError(error);
    }
  }
  
  /**
   * Stream a chat completion.
   */
  async *completionStream(request: CompletionRequest): AsyncIterable<ChatCompletionChunk> {
    try {
      const chatRequest: ChatRequest & { stream: true } = {
        model: request.model,
        messages: this.convertMessages(request.messages),
        stream: true,
      };
      
      // Add tools if provided
      if (request.tools && request.tools.length > 0) {
        if (this.toolsSupported === false) {
          throw new ProviderRequestError(
            this.PROVIDER_NAME,
            `Ollama version ${this.version} does not support tool calling.`
          );
        }
        chatRequest.tools = this.convertTools(request.tools);
      }
      
      // Add options
      chatRequest.options = {};
      if (request.temperature !== undefined) {
        chatRequest.options.temperature = request.temperature;
      }
      if (request.top_p !== undefined) {
        chatRequest.options.top_p = request.top_p;
      }
      if (request.max_tokens !== undefined) {
        chatRequest.options.num_predict = request.max_tokens;
      }
      
      const stream = await this.client.chat(chatRequest);
      
      const id = `ollama-${Date.now()}`;
      const created = Math.floor(Date.now() / 1000);
      
      for await (const chunk of stream) {
        const chunkChoice: ChunkChoice = {
          index: 0,
          delta: {
            content: chunk.message?.content || '',
          },
          finish_reason: chunk.done ? 'stop' : null,
        };
        
        // Include tool calls in delta if present
        if (chunk.message?.tool_calls && chunk.message.tool_calls.length > 0) {
          chunkChoice.delta.tool_calls = chunk.message.tool_calls.map((tc, i) => ({
            id: `call_${i}`,
            type: 'function' as const,
            function: {
              name: tc.function.name,
              arguments: JSON.stringify(tc.function.arguments),
            },
          }));
          if (chunk.done) {
            chunkChoice.finish_reason = 'tool_calls';
          }
        }
        
        yield {
          id,
          object: 'chat.completion.chunk' as const,
          created,
          model: chunk.model || request.model,
          choices: [chunkChoice],
        };
      }
    } catch (error) {
      // Re-throw our errors
      if (error instanceof ProviderRequestError || error instanceof ProviderUnavailableError || error instanceof TimeoutError) {
        throw error;
      }
      throw this.handleError(error);
    }
  }
}
