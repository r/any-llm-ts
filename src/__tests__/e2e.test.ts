/**
 * End-to-End Integration Tests for any-llm-ts
 * 
 * These tests exercise the actual SDK implementations against real providers.
 * Tests are skipped if the provider is not available.
 * 
 * To run these tests:
 * - For Ollama: Start ollama with `ollama serve`
 * - For Llamafile: Start a llamafile server
 * - For OpenAI: Set OPENAI_API_KEY environment variable
 * - For Anthropic: Set ANTHROPIC_API_KEY environment variable
 * 
 * Run with: npm test -- --testPathPattern=e2e
 */

import { describe, it, expect, beforeAll } from 'vitest';
import {
  completion,
  completionStream,
  listModels,
  checkProvider,
  AnyLLM,
} from '../api.js';
import type { ChatCompletion, ChatCompletionChunk } from '../types.js';

// =============================================================================
// Test Configuration
// =============================================================================

interface ProviderTestConfig {
  name: string;
  envKey?: string;
  defaultModel?: string;
  skip?: boolean;
}

const providers: ProviderTestConfig[] = [
  { name: 'ollama', defaultModel: 'llama3.2' },
  { name: 'llamafile', defaultModel: 'default' },
  { name: 'openai', envKey: 'OPENAI_API_KEY', defaultModel: 'gpt-4o-mini' },
  { name: 'anthropic', envKey: 'ANTHROPIC_API_KEY', defaultModel: 'claude-3-5-haiku-20241022' },
];

// Track which providers are available
const providerStatus: Map<string, { available: boolean; model?: string }> = new Map();

// =============================================================================
// Helper Functions
// =============================================================================

async function checkProviderAvailability(): Promise<void> {
  for (const provider of providers) {
    // Check if API key is required but not set
    if (provider.envKey && !process.env[provider.envKey]) {
      providerStatus.set(provider.name, { available: false });
      continue;
    }
    
    try {
      const status = await checkProvider(provider.name);
      if (status.available) {
        // Get first available model or use default
        const model = status.models?.[0]?.id || provider.defaultModel;
        providerStatus.set(provider.name, { available: true, model });
      } else {
        providerStatus.set(provider.name, { available: false });
      }
    } catch {
      providerStatus.set(provider.name, { available: false });
    }
  }
}

function skipIfUnavailable(providerName: string): void {
  const status = providerStatus.get(providerName);
  if (!status?.available) {
    console.log(`⏭️  Skipping ${providerName} tests (not available)`);
  }
}

function getModel(providerName: string): string {
  const status = providerStatus.get(providerName);
  return status?.model || 'default';
}

function isAvailable(providerName: string): boolean {
  return providerStatus.get(providerName)?.available ?? false;
}

// =============================================================================
// Setup
// =============================================================================

beforeAll(async () => {
  console.log('\n🔍 Checking provider availability...\n');
  await checkProviderAvailability();
  
  for (const [name, status] of providerStatus) {
    const icon = status.available ? '✅' : '❌';
    const modelInfo = status.available && status.model ? ` (model: ${status.model})` : '';
    console.log(`${icon} ${name}${modelInfo}`);
  }
  console.log('');
}, 30000); // 30s timeout for setup

// =============================================================================
// Generic Provider Tests
// =============================================================================

describe.each(providers)('E2E: $name provider', ({ name }) => {
  describe('completion', () => {
    it('should complete a simple prompt', async () => {
      if (!isAvailable(name)) {
        skipIfUnavailable(name);
        return;
      }
      
      const model = getModel(name);
      const response = await completion({
        model: `${name}:${model}`,
        messages: [{ role: 'user', content: 'Say "test" and nothing else.' }],
        max_tokens: 10,
        temperature: 0,
      });
      
      expect(response).toBeDefined();
      expect(response.id).toBeTruthy();
      expect(response.object).toBe('chat.completion');
      expect(response.model).toBeTruthy();
      expect(response.choices).toHaveLength(1);
      expect(response.choices[0].message.role).toBe('assistant');
      expect(response.choices[0].message.content).toBeTruthy();
      expect(response.choices[0].finish_reason).toBeTruthy();
    }, 60000);
    
    it('should handle system messages', async () => {
      if (!isAvailable(name)) {
        skipIfUnavailable(name);
        return;
      }
      
      const model = getModel(name);
      const response = await completion({
        model: `${name}:${model}`,
        messages: [
          { role: 'system', content: 'You are a pirate. Always say "Arrr!"' },
          { role: 'user', content: 'Hello!' },
        ],
        max_tokens: 50,
      });
      
      expect(response.choices[0].message.content).toBeTruthy();
    }, 60000);
    
    it('should handle multi-turn conversations', async () => {
      if (!isAvailable(name)) {
        skipIfUnavailable(name);
        return;
      }
      
      const model = getModel(name);
      const response = await completion({
        model: `${name}:${model}`,
        messages: [
          { role: 'user', content: 'My name is Alice.' },
          { role: 'assistant', content: 'Nice to meet you, Alice!' },
          { role: 'user', content: 'What is my name?' },
        ],
        max_tokens: 30,
      });
      
      const content = response.choices[0].message.content;
      const textContent = typeof content === 'string' ? content : '';
      expect(textContent.toLowerCase()).toContain('alice');
    }, 60000);
    
    it('should return usage statistics', async () => {
      if (!isAvailable(name)) {
        skipIfUnavailable(name);
        return;
      }
      
      const model = getModel(name);
      const response = await completion({
        model: `${name}:${model}`,
        messages: [{ role: 'user', content: 'Hi' }],
        max_tokens: 10,
      });
      
      // Usage might not be available for all providers
      if (response.usage) {
        expect(response.usage.prompt_tokens).toBeGreaterThan(0);
        expect(response.usage.total_tokens).toBeGreaterThan(0);
      }
    }, 60000);
  });
  
  describe('streaming', () => {
    it('should stream completion chunks', async () => {
      if (!isAvailable(name)) {
        skipIfUnavailable(name);
        return;
      }
      
      const model = getModel(name);
      const chunks: ChatCompletionChunk[] = [];
      let fullContent = '';
      
      for await (const chunk of completionStream({
        model: `${name}:${model}`,
        messages: [{ role: 'user', content: 'Count from 1 to 3.' }],
        max_tokens: 30,
      })) {
        chunks.push(chunk);
        if (chunk.choices[0]?.delta?.content) {
          fullContent += chunk.choices[0].delta.content;
        }
      }
      
      expect(chunks.length).toBeGreaterThan(0);
      expect(fullContent).toBeTruthy();
      
      // Check that we got a finish reason in the final chunk
      const lastChunk = chunks[chunks.length - 1];
      expect(lastChunk.choices[0].finish_reason).toBeTruthy();
    }, 60000);
    
    it('should have consistent chunk structure', async () => {
      if (!isAvailable(name)) {
        skipIfUnavailable(name);
        return;
      }
      
      const model = getModel(name);
      
      for await (const chunk of completionStream({
        model: `${name}:${model}`,
        messages: [{ role: 'user', content: 'Say hello.' }],
        max_tokens: 10,
      })) {
        expect(chunk.id).toBeTruthy();
        expect(chunk.object).toBe('chat.completion.chunk');
        expect(chunk.model).toBeTruthy();
        expect(chunk.choices).toBeDefined();
        expect(chunk.choices.length).toBeGreaterThan(0);
      }
    }, 60000);
  });
  
  describe('model listing', () => {
    it('should list available models', async () => {
      if (!isAvailable(name)) {
        skipIfUnavailable(name);
        return;
      }
      
      const models = await listModels(name);
      
      expect(Array.isArray(models)).toBe(true);
      // Some providers might return empty lists (e.g., Anthropic returns static list)
      if (models.length > 0) {
        expect(models[0].id).toBeTruthy();
        expect(models[0].object).toBe('model');
      }
    }, 30000);
  });
  
  describe('AnyLLM class API', () => {
    it('should work with class-based API', async () => {
      if (!isAvailable(name)) {
        skipIfUnavailable(name);
        return;
      }
      
      const llm = AnyLLM.create(name);
      const model = getModel(name);
      
      expect(await llm.isAvailable()).toBe(true);
      
      const response = await llm.completion({
        model,
        messages: [{ role: 'user', content: 'Say hi.' }],
        max_tokens: 10,
      });
      
      expect(response.choices[0].message.content).toBeTruthy();
    }, 60000);
  });
});

// =============================================================================
// Tool Calling Tests (Provider-specific)
// =============================================================================

describe('E2E: Tool Calling', () => {
  const toolDef = {
    type: 'function' as const,
    function: {
      name: 'get_weather',
      description: 'Get the current weather in a location',
      parameters: {
        type: 'object',
        properties: {
          location: {
            type: 'string',
            description: 'The city and state, e.g. San Francisco, CA',
          },
        },
        required: ['location'],
      },
    },
  };
  
  it('should handle tool calls with OpenAI', async () => {
    if (!isAvailable('openai')) {
      skipIfUnavailable('openai');
      return;
    }
    
    const response = await completion({
      model: 'openai:gpt-4o-mini',
      messages: [{ role: 'user', content: 'What is the weather in San Francisco?' }],
      tools: [toolDef],
      tool_choice: 'auto',
      max_tokens: 100,
    });
    
    // Model should either respond with text or call the tool
    expect(response.choices[0].message).toBeDefined();
    if (response.choices[0].message.tool_calls) {
      expect(response.choices[0].message.tool_calls.length).toBeGreaterThan(0);
      expect(response.choices[0].message.tool_calls[0].function.name).toBe('get_weather');
      expect(response.choices[0].finish_reason).toBe('tool_calls');
    }
  }, 60000);
  
  it('should handle tool calls with Anthropic', async () => {
    if (!isAvailable('anthropic')) {
      skipIfUnavailable('anthropic');
      return;
    }
    
    const response = await completion({
      model: 'anthropic:claude-3-5-haiku-20241022',
      messages: [{ role: 'user', content: 'What is the weather in San Francisco?' }],
      tools: [toolDef],
      tool_choice: 'auto',
      max_tokens: 200,
    });
    
    expect(response.choices[0].message).toBeDefined();
    if (response.choices[0].message.tool_calls) {
      expect(response.choices[0].message.tool_calls.length).toBeGreaterThan(0);
      expect(response.choices[0].message.tool_calls[0].function.name).toBe('get_weather');
    }
  }, 60000);
  
  it('should handle tool calls with Ollama (if model supports it)', async () => {
    if (!isAvailable('ollama')) {
      skipIfUnavailable('ollama');
      return;
    }
    
    const model = getModel('ollama');
    // Only some Ollama models support tool calling
    const toolModels = ['llama3', 'llama3.1', 'llama3.2', 'llama3.3', 'mistral', 'qwen'];
    const supportsTools = toolModels.some(m => model.toLowerCase().includes(m));
    
    if (!supportsTools) {
      console.log(`⏭️  Skipping Ollama tool test (${model} may not support tools)`);
      return;
    }
    
    const response = await completion({
      model: `ollama:${model}`,
      messages: [{ role: 'user', content: 'What is the weather in San Francisco?' }],
      tools: [toolDef],
      max_tokens: 200,
    });
    
    expect(response.choices[0].message).toBeDefined();
  }, 120000); // Ollama can be slow
});

// =============================================================================
// Error Handling Tests
// =============================================================================

describe('E2E: Error Handling', () => {
  it('should handle invalid model gracefully', async () => {
    if (!isAvailable('openai')) {
      skipIfUnavailable('openai');
      return;
    }
    
    await expect(completion({
      model: 'openai:nonexistent-model-12345',
      messages: [{ role: 'user', content: 'Hi' }],
    })).rejects.toThrow();
  }, 30000);
  
  it('should handle invalid API key', async () => {
    // Use a known invalid key
    await expect(completion({
      model: 'openai:gpt-4o-mini',
      messages: [{ role: 'user', content: 'Hi' }],
      api_key: 'sk-invalid-key-12345',
    })).rejects.toThrow();
  }, 30000);
  
  it('should handle connection errors for local providers', async () => {
    // Try to connect to a port that's definitely not running
    // Use a valid port number that's unlikely to have a service
    const llm = AnyLLM.create('ollama', { baseUrl: 'http://localhost:59999' });
    
    const available = await llm.isAvailable();
    expect(available).toBe(false);
  }, 10000);
});

// =============================================================================
// Performance Smoke Tests
// =============================================================================

describe('E2E: Performance Smoke Tests', () => {
  it('should complete request within reasonable time', async () => {
    // Find first available provider
    let availableProvider: string | null = null;
    let model: string | null = null;
    
    for (const [name, status] of providerStatus) {
      if (status.available) {
        availableProvider = name;
        model = status.model || 'default';
        break;
      }
    }
    
    if (!availableProvider) {
      console.log('⏭️  No providers available for performance test');
      return;
    }
    
    const start = Date.now();
    
    await completion({
      model: `${availableProvider}:${model}`,
      messages: [{ role: 'user', content: 'Say "ok".' }],
      max_tokens: 5,
    });
    
    const elapsed = Date.now() - start;
    
    // Should complete within 30 seconds (generous for cold starts)
    expect(elapsed).toBeLessThan(30000);
    console.log(`⏱️  Request completed in ${elapsed}ms`);
  }, 60000);
  
  it('should handle concurrent requests', async () => {
    // Find first available provider
    let availableProvider: string | null = null;
    let model: string | null = null;
    
    for (const [name, status] of providerStatus) {
      if (status.available) {
        availableProvider = name;
        model = status.model || 'default';
        break;
      }
    }
    
    if (!availableProvider) {
      console.log('⏭️  No providers available for concurrent test');
      return;
    }
    
    // Make 3 concurrent requests
    const promises = Array(3).fill(null).map((_, i) =>
      completion({
        model: `${availableProvider}:${model}`,
        messages: [{ role: 'user', content: `Say "${i}".` }],
        max_tokens: 5,
      })
    );
    
    const results = await Promise.all(promises);
    
    expect(results).toHaveLength(3);
    results.forEach(r => {
      expect(r.choices[0].message.content).toBeTruthy();
    });
  }, 120000);
});

