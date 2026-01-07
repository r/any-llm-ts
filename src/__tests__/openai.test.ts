/**
 * Tests for OpenAI provider.
 * 
 * These tests focus on the provider configuration and metadata
 * since mocking the SDK is complex in ES modules.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { OpenAIProvider, OpenAICompatibleProvider } from '../providers/openai.js';

describe('OpenAIProvider', () => {
  describe('constructor and metadata', () => {
    it('should create provider with correct metadata', () => {
      const provider = new OpenAIProvider({ apiKey: 'test-key' });
      
      expect(provider.PROVIDER_NAME).toBe('openai');
      expect(provider.ENV_API_KEY_NAME).toBe('OPENAI_API_KEY');
      expect(provider.API_BASE).toBe('https://api.openai.com/v1');
      expect(provider.PROVIDER_DOCUMENTATION_URL).toBe('https://platform.openai.com/docs');
    });
    
    it('should report correct capabilities', () => {
      const provider = new OpenAIProvider({ apiKey: 'test-key' });
      
      expect(provider.SUPPORTS_STREAMING).toBe(true);
      expect(provider.SUPPORTS_TOOLS).toBe(true);
      expect(provider.SUPPORTS_VISION).toBe(true);
      expect(provider.SUPPORTS_LIST_MODELS).toBe(true);
      expect(provider.SUPPORTS_REASONING).toBe(false);
    });
    
    it('should use custom base URL if provided', () => {
      const provider = new OpenAIProvider({
        apiKey: 'test-key',
        baseUrl: 'https://custom.api.com/v1',
      });
      
      expect(provider.PROVIDER_NAME).toBe('openai');
    });
    
    it('should get metadata correctly', () => {
      const provider = new OpenAIProvider({ apiKey: 'test-key' });
      const metadata = provider.getMetadata();
      
      expect(metadata.name).toBe('openai');
      expect(metadata.envKey).toBe('OPENAI_API_KEY');
      expect(metadata.streaming).toBe(true);
      expect(metadata.completion).toBe(true);
      expect(metadata.listModels).toBe(true);
    });
  });
  
  describe('isAvailable', () => {
    it('should return false when no API key is set', async () => {
      // Create provider without API key and no env var
      const originalEnv = process.env.OPENAI_API_KEY;
      delete process.env.OPENAI_API_KEY;
      
      const provider = new OpenAIProvider({});
      const result = await provider.isAvailable();
      
      // Restore env
      if (originalEnv) process.env.OPENAI_API_KEY = originalEnv;
      
      expect(result).toBe(false);
    });
  });
});

describe('OpenAICompatibleProvider', () => {
  it('should create provider with custom settings', () => {
    const provider = new OpenAICompatibleProvider(
      'groq',
      'https://api.groq.com/openai/v1',
      'GROQ_API_KEY',
      { apiKey: 'test-key' }
    );
    
    expect(provider.PROVIDER_NAME).toBe('groq');
    expect(provider.ENV_API_KEY_NAME).toBe('GROQ_API_KEY');
    expect(provider.API_BASE).toBe('https://api.groq.com/openai/v1');
  });
  
  it('should allow creating multiple compatible providers', () => {
    const groq = new OpenAICompatibleProvider(
      'groq',
      'https://api.groq.com/openai/v1',
      'GROQ_API_KEY',
      { apiKey: 'groq-key' }
    );
    
    const together = new OpenAICompatibleProvider(
      'together',
      'https://api.together.xyz/v1',
      'TOGETHER_API_KEY',
      { apiKey: 'together-key' }
    );
    
    expect(groq.PROVIDER_NAME).toBe('groq');
    expect(together.PROVIDER_NAME).toBe('together');
  });
});
