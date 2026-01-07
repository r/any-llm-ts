/**
 * Tests for the high-level API functions.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  completion,
  completionStream,
  listModels,
  checkProvider,
  getSupportedProviders,
  AnyLLM,
} from '../api.js';
import { registerProvider, clearProviderCache, getRegisteredProviders } from '../registry.js';
import { BaseProvider } from '../providers/base.js';
import type { ChatCompletion, ChatCompletionChunk, CompletionRequest, ModelInfo } from '../types.js';

// Create a mock provider for testing
class MockProvider extends BaseProvider {
  readonly PROVIDER_NAME = 'mock';
  readonly ENV_API_KEY_NAME = 'MOCK_API_KEY';
  readonly PROVIDER_DOCUMENTATION_URL = 'https://mock.test';
  readonly API_BASE = 'https://api.mock.test';
  readonly SUPPORTS_LIST_MODELS = true;
  
  private mockResponse: ChatCompletion;
  private mockStreamChunks: ChatCompletionChunk[];
  private mockModels: ModelInfo[];
  private mockAvailable: boolean;
  
  constructor(config: any = {}) {
    super(config);
    this.init();
    
    this.mockResponse = config.mockResponse || {
      id: 'mock-123',
      object: 'chat.completion',
      created: Date.now(),
      model: 'mock-model',
      provider: 'mock',
      choices: [{
        index: 0,
        message: { role: 'assistant', content: 'Mock response' },
        finish_reason: 'stop',
      }],
      usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
    };
    
    this.mockStreamChunks = config.mockStreamChunks || [
      {
        id: 'mock-stream-123',
        object: 'chat.completion.chunk',
        created: Date.now(),
        model: 'mock-model',
        choices: [{ index: 0, delta: { role: 'assistant', content: 'Hello' }, finish_reason: null }],
      },
      {
        id: 'mock-stream-123',
        object: 'chat.completion.chunk',
        created: Date.now(),
        model: 'mock-model',
        choices: [{ index: 0, delta: { content: ' world!' }, finish_reason: 'stop' }],
      },
    ];
    
    this.mockModels = config.mockModels || [
      { id: 'mock-model-1', object: 'model', provider: 'mock' },
      { id: 'mock-model-2', object: 'model', provider: 'mock' },
    ];
    
    this.mockAvailable = config.mockAvailable ?? true;
  }
  
  protected requiresApiKey(): boolean {
    return false;
  }
  
  async isAvailable(): Promise<boolean> {
    return this.mockAvailable;
  }
  
  async listModels(): Promise<ModelInfo[]> {
    return this.mockModels;
  }
  
  async completion(request: CompletionRequest): Promise<ChatCompletion> {
    return { ...this.mockResponse, model: request.model };
  }
  
  async *completionStream(request: CompletionRequest): AsyncIterable<ChatCompletionChunk> {
    for (const chunk of this.mockStreamChunks) {
      yield { ...chunk, model: request.model };
    }
  }
}

describe('API Functions', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    clearProviderCache();
    // Register mock provider
    registerProvider('mock', MockProvider);
  });
  
  afterEach(() => {
    vi.resetAllMocks();
  });
  
  describe('getSupportedProviders', () => {
    it('should return list of registered providers', () => {
      const providers = getSupportedProviders();
      
      expect(Array.isArray(providers)).toBe(true);
      expect(providers).toContain('mock');
      expect(providers).toContain('openai');
      expect(providers).toContain('anthropic');
      expect(providers).toContain('ollama');
      expect(providers).toContain('llamafile');
    });
  });
  
  describe('completion', () => {
    it('should make completion with provider:model format', async () => {
      const result = await completion({
        model: 'mock:mock-model',
        messages: [{ role: 'user', content: 'Hello' }],
      });
      
      expect(result.provider).toBe('mock');
      expect(result.model).toBe('mock-model');
      expect(result.choices[0].message.content).toBe('Mock response');
    });
    
    it('should make completion with separate provider parameter', async () => {
      const result = await completion({
        model: 'mock-model',
        provider: 'mock',
        messages: [{ role: 'user', content: 'Hello' }],
      });
      
      expect(result.model).toBe('mock-model');
    });
    
    it('should pass API key and base URL overrides', async () => {
      const result = await completion({
        model: 'mock:mock-model',
        messages: [{ role: 'user', content: 'Hello' }],
        api_key: 'custom-key',
        api_base: 'https://custom.api.test',
      });
      
      expect(result).toBeDefined();
    });
  });
  
  describe('completionStream', () => {
    it('should stream completion chunks', async () => {
      const chunks: ChatCompletionChunk[] = [];
      
      for await (const chunk of completionStream({
        model: 'mock:mock-model',
        messages: [{ role: 'user', content: 'Hello' }],
      })) {
        chunks.push(chunk);
      }
      
      expect(chunks.length).toBe(2);
      expect(chunks[0].choices[0].delta.content).toBe('Hello');
      expect(chunks[1].choices[0].delta.content).toBe(' world!');
      expect(chunks[1].choices[0].finish_reason).toBe('stop');
    });
  });
  
  describe('listModels', () => {
    it('should list models from provider', async () => {
      const models = await listModels('mock');
      
      expect(models).toHaveLength(2);
      expect(models.map(m => m.id)).toContain('mock-model-1');
      expect(models.map(m => m.id)).toContain('mock-model-2');
    });
  });
  
  describe('checkProvider', () => {
    it('should return available status when provider is available', async () => {
      const status = await checkProvider('mock');
      
      expect(status.provider).toBe('mock');
      expect(status.available).toBe(true);
      expect(status.models).toHaveLength(2);
    });
    
    it('should return unavailable status for unknown provider', async () => {
      const status = await checkProvider('unknown-provider');
      
      expect(status.available).toBe(false);
      expect(status.error).toBeDefined();
    });
  });
});

describe('AnyLLM Class', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    clearProviderCache();
    registerProvider('mock', MockProvider);
  });
  
  describe('create', () => {
    it('should create instance for provider', () => {
      const llm = AnyLLM.create('mock');
      
      expect(llm).toBeInstanceOf(AnyLLM);
      expect(llm.name).toBe('mock');
    });
  });
  
  describe('static methods', () => {
    it('should list supported providers', () => {
      const providers = AnyLLM.getSupportedProviders();
      
      expect(providers).toContain('mock');
    });
    
    it('should check if provider exists', () => {
      expect(AnyLLM.hasProvider('mock')).toBe(true);
      expect(AnyLLM.hasProvider('nonexistent')).toBe(false);
    });
    
    it('should parse model strings', () => {
      const result = AnyLLM.parseModelString('openai:gpt-4o');
      
      expect(result).toEqual({
        provider: 'openai',
        model: 'gpt-4o',
      });
    });
    
    it('should return null for invalid model strings', () => {
      const result = AnyLLM.parseModelString('just-a-model');
      
      expect(result).toBeNull();
    });
  });
  
  describe('instance methods', () => {
    let llm: AnyLLM;
    
    beforeEach(() => {
      llm = AnyLLM.create('mock');
    });
    
    it('should check availability', async () => {
      const available = await llm.isAvailable();
      
      expect(available).toBe(true);
    });
    
    it('should list models', async () => {
      const models = await llm.listModels();
      
      expect(models).toHaveLength(2);
    });
    
    it('should make completion', async () => {
      const result = await llm.completion({
        model: 'mock-model',
        messages: [{ role: 'user', content: 'Hello' }],
      });
      
      expect(result.choices[0].message.content).toBe('Mock response');
    });
    
    it('should stream completion', async () => {
      const chunks: ChatCompletionChunk[] = [];
      
      for await (const chunk of llm.completionStream({
        model: 'mock-model',
        messages: [{ role: 'user', content: 'Hello' }],
      })) {
        chunks.push(chunk);
      }
      
      expect(chunks.length).toBe(2);
    });
  });
});

describe('Model String Parsing', () => {
  beforeEach(() => {
    clearProviderCache();
    registerProvider('mock', MockProvider);
  });
  
  it('should parse provider:model format', async () => {
    const result = await completion({
      model: 'mock:my-model',
      messages: [{ role: 'user', content: 'Hi' }],
    });
    
    expect(result.model).toBe('my-model');
  });
  
  it('should handle model with colons', async () => {
    // Some model names have colons (e.g., ollama models like llama3.2:latest)
    const result = await completion({
      model: 'mock:model:with:colons',
      messages: [{ role: 'user', content: 'Hi' }],
    });
    
    expect(result.model).toBe('model:with:colons');
  });
  
  it('should use provider param when model has no prefix', async () => {
    const result = await completion({
      model: 'simple-model',
      provider: 'mock',
      messages: [{ role: 'user', content: 'Hi' }],
    });
    
    expect(result.model).toBe('simple-model');
  });
});

describe('Provider Configuration', () => {
  beforeEach(() => {
    clearProviderCache();
    registerProvider('mock', MockProvider);
  });
  
  it('should pass config to provider', async () => {
    // The mock provider accepts config, verify it's passed through
    const result = await completion({
      model: 'mock:test-model',
      messages: [{ role: 'user', content: 'Hi' }],
      temperature: 0.5,
      max_tokens: 100,
    });
    
    expect(result).toBeDefined();
  });
});

